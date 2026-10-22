"""Teacher-only prompt calibration and real continuations through frozen collector.

History is reconstructed by replaying saved student assistant messages through
the original collector. Every visible history message must exactly match before
the teacher takes over; rejected calls are reconstructed without env execution.
No target, reward, or task-outcome field is added to model inputs.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class ReplayMismatch(RuntimeError):
    pass


class ReplayClient:
    """Keep original action guards, projection, repeat detection and total budget."""
    def __init__(self, client, prefix, preflight=False):
        self.client = client
        self.prefix = deepcopy(prefix)
        self.preflight = preflight
        self.replay_verified = False
        self.replayed_assistant_turns = 0
        self.teacher_assistant_turns = 0
        self.first_teacher_tool_calls = None
        self.last_context_tokens = None
        self.last_context_event = None

    def project_observation(self, *args, **kwargs):
        return self.client.project_observation(*args, **kwargs)

    def complete(self, messages, tools):
        if not self.replay_verified:
            count = len(messages)
            if count > len(self.prefix) or messages != self.prefix[:count]:
                difference = next((i for i, (a,b) in enumerate(zip(messages,self.prefix)) if a != b), count)
                raise ReplayMismatch(f"Visible history differs at message {difference}")
            if count < len(self.prefix):
                saved = self.prefix[count]
                if saved["role"] != "assistant":
                    raise ReplayMismatch("Collector did not reach the expected assistant boundary")
                self.replayed_assistant_turns += 1
                self.last_context_tokens = self.last_context_event = None
                return deepcopy(saved)
            self.replay_verified = True
        if self.preflight:
            return {"role":"assistant", "content":"Replay boundary verified", "tool_calls":[]}
        answer = self.client.complete(messages, tools)
        self.last_context_tokens = self.client.last_context_tokens
        self.last_context_event = self.client.last_context_event
        self.teacher_assistant_turns += 1
        if self.first_teacher_tool_calls is None:
            self.first_teacher_tool_calls = deepcopy(answer.get("tool_calls") or [])
        return answer


def call_signature(calls):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--pilot",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prompt",type=Path,required=True)
    parser.add_argument("--official-metadata",type=Path,required=True)
    parser.add_argument("--teacher-url",default="http://127.0.0.1:5800/v1")
    parser.add_argument("--shop-url",default="http://127.0.0.1:5700")
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    status={"phase":"preparing","started_at":datetime.now(timezone.utc).isoformat(),"completed_records":0}
    import os
    status["pid"]=os.getpid()
    def snapshot():
        status["elapsed_minutes"]=round((time.monotonic()-start)/60,2)
        write(args.output/"status.json",status)
    snapshot()
    try:
        manifest=read(args.pilot/"task-manifest.json")
        tasks=rows(args.pilot/"tasks.jsonl")
        ids=manifest["task_ids"]
        assert [t["task_id"] for t in tasks]==ids and len(set(ids))==50
        assert sha(args.pilot/"tasks.jsonl")==manifest["benchmark_sha256"]
        for kind,path in [("train","data/sft/train.jsonl"),("development","data/sft/validation.jsonl")]:
            assert sha(args.workspace/path)==manifest["sha256"][kind]
        official=read(args.official_metadata)
        for f in official["files"]:
            assert sha(args.model/f["name"])==f["sha256"],f["name"]
        source=args.output/"source"
        shutil.copytree(args.pilot/"teacher/source/src",source/"src",ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(args.pilot/"teacher/source/run_pilot.py",source/"run_pilot.py")
        shutil.copy2(__file__,source/"run_calibration.py")
        shutil.copy2(args.prompt,source/"calibration_prompt.txt")
        sys.path.insert(0,str(source/"src"))
        from shopping_grpo.evaluation.blind_guard import validate_canonical_blind_asset,guard_blind_final
        guard,blind_ids=validate_canonical_blind_asset()
        assert guard["task_sha256"]==manifest["sha256"]["final"] and not set(ids)&blind_ids
        guard_blind_final([args.pilot/"tasks.jsonl"],allowed=False)
        baseline_config=read(args.pilot/"teacher/run_config.json")
        for name,h in baseline_config["source_sha256"].items():
            assert sha(source/name)==h,name
        from shopping_grpo.evaluation.rollout import OpenAIChatClient,collect_for_task,append_jsonl,_is_infrastructure_failure
        from shopping_grpo.evaluation.summary import summarize_trajectories,_is_strict_success
        spec=importlib.util.spec_from_file_location("frozen_pilot",source/"run_pilot.py")
        pilot_module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot_module)
        from transformers import AutoProcessor
        processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
        appendix=args.prompt.read_text(encoding="utf-8")

        class CalibratedTransport(pilot_module.TeacherTransport):
            def __init__(self, appendix):
                super().__init__(processor,args.teacher_url,"qwen38-teacher")
                self.appendix=appendix

            def render(self,messages,tools):
                adjusted=deepcopy(messages)
                if self.appendix:
                    if adjusted[0]["role"]!="system":
                        raise ValueError("Expected a system message")
                    adjusted[0]["content"]+=self.appendix
                return super().render(adjusted,tools)

        def make_client(extra):
            transport=CalibratedTransport(extra)
            assert hashlib.sha256(processor.chat_template.encode()).hexdigest()==baseline_config["chat_template_sha256"]
            assert transport.eos_token_ids==baseline_config["stop_token_ids"]
            return OpenAIChatClient(model="qwen38-teacher",base_url="local-transport",api_key="LOCAL",
                temperature=0.0,top_p=1.0,max_tokens=512,context_window=24576,context_safety_margin=512,
                context_compaction_enable=False,observation_token_budget=1536,
                observation_detail_token_budget=4096,observation_generic_token_budget=768,
                observation_search_top_k=20,token_counter=transport.count_chat_tokens,
                observation_token_counter=transport.count_text_tokens,transport=transport)

        calibrated=make_client(appendix)
        baseline=make_client("")
        config={"task_manifest":manifest,"official_metadata":official,"teacher_configuration":baseline_config,
            "calibration_prompt_sha256":sha(args.prompt),"calibration_prompt":appendix,
            "independent_variable":"Teacher system appendix only; model, template, decoding and environment frozen",
            "source_sha256":{str(f.relative_to(source)):sha(f) for f in source.rglob("*.py")},
            "continuation_budget":"35 total executed steps including the replayed student prefix",
            "preflight_only":args.preflight_only,
            "client_packages":{n:importlib.metadata.version(n) for n in ["torch","transformers","peft","accelerate"]}}
        assert config["client_packages"]==baseline_config["packages"]
        write(args.output/"run_config.json",config)
        student_rows=rows(args.pilot/"student-v2/trajectories.jsonl")
        probes=rows(args.pilot/"probe/state-probes.jsonl")
        failure_ids=[r["task_id"] for r in student_rows if not _is_strict_success(r)]
        failure_probes=[p for p in probes if p["kind"]=="student_failure_state"]
        assert [p["task_id"] for p in failure_probes]==failure_ids and len(failure_ids)==4
        task_by_id={t["task_id"]:t for t in tasks}
        if not args.preflight_only:
            results=[]
            for task in tasks:
                status.update(phase="calibrated_rollouts",active_task_id=task["task_id"])
                snapshot()
                before=time.monotonic()
                r=collect_for_task(task,client=calibrated,base_url=args.shop_url,max_steps=35)
                r.update(actor="teacher_calibrated",elapsed_seconds=round(time.monotonic()-before,3))
                append_jsonl(args.output/"calibrated-trajectories.jsonl",[r])
                results.append(r)
                summary=summarize_trajectories(ids,results)
                write(args.output/"calibrated-summary.json",summary)
                status.update(completed_records=len(results),completed_rollouts=len(results),strict_successes=summary["strict_successes"])
                snapshot()
                print("CALIBRATED",len(results),50,task["task_id"],r["status"],_is_strict_success(r),flush=True)
                if _is_infrastructure_failure(r) or (r.get("error") and r["error"]["type"]!="ContextBudgetError"):
                    raise RuntimeError("Unexpected or infrastructure failure; see saved trajectory")
        continuations=[]
        for variant,client in ([("preflight",baseline)] if args.preflight_only else [("baseline",baseline),("calibrated",calibrated)]):
            for probe in failure_probes:
                status.update(phase="replay_preflight" if args.preflight_only else "student_state_continuations",active_variant=variant,active_task_id=probe["task_id"])
                snapshot()
                replay=ReplayClient(client,probe["prefix"],preflight=args.preflight_only)
                before=time.monotonic()
                r=collect_for_task(task_by_id[probe["task_id"]],client=replay,base_url=args.shop_url,max_steps=35)
                r.update(actor="teacher_"+variant,elapsed_seconds=round(time.monotonic()-before,3),
                    continuation={"source_student_trajectory_sha256":sha(args.pilot/"student-v2/trajectories.jsonl"),
                        "source_probe_task_id":probe["task_id"],"source_message_index":probe["message_index"],
                        "prefix_sha256":hashlib.sha256(json.dumps(probe["prefix"],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                        "visible_history_exactly_reproduced":replay.replay_verified,
                        "replayed_assistant_turns":replay.replayed_assistant_turns,
                        "teacher_assistant_turns":replay.teacher_assistant_turns,
                        "first_teacher_tool_calls":replay.first_teacher_tool_calls,
                        "baseline_first_action_matches_saved_probe":call_signature(replay.first_teacher_tool_calls or [])==call_signature(probe["teacher_completion"].get("tool_calls") or []) if variant=="baseline" else None,
                        "preflight_only":args.preflight_only})
                append_jsonl(args.output/"continuations.jsonl",[r])
                continuations.append(r)
                status.update(completed_continuations=len(continuations),completed_records=status["completed_records"]+1)
                snapshot()
                print("CONTINUATION",variant,probe["task_id"],r["status"],_is_strict_success(r),"replay",replay.replay_verified,flush=True)
                if not replay.replay_verified or _is_infrastructure_failure(r) or (r.get("error") and r["error"]["type"]!="ContextBudgetError"):
                    raise RuntimeError("State reconstruction or inference failed; do not count as recovery")
        if not args.preflight_only:
            old=read(args.pilot/"teacher/summary.json")
            new=read(args.output/"calibrated-summary.json")
            a,b=map(set,[old["strict_success_task_ids"],new["strict_success_task_ids"]])
            summary={"baseline":old,"calibrated":new,"gained":sorted(b-a),"lost":sorted(a-b),
                "both_success":sorted(a&b),"both_fail":sorted(set(ids)-a-b),
                "percentage_point_change":100*(len(b)-len(a))/50,
                "continuations":{variant:summarize_trajectories(failure_ids,[r for r in continuations if r["actor"]=="teacher_"+variant]) for variant in ["baseline","calibrated"]},
                "scope":"Training-only prompt calibration on the previously inspected 50 tasks; not held-out evidence",
                "baseline_reused_from":str(args.pilot/"teacher"),"new_full_rollouts":50,"new_continuations":8}
            write(args.output/"paired-results.json",summary)
        status.update(phase="completed",finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        status.update(phase="failed",error=f"{type(e).__name__}: {e}")
        raise
    finally:
        snapshot()


if __name__=="__main__":
    main()
