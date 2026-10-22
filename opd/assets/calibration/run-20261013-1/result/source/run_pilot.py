"""Run the frozen training-only E1/FP8 teacher pilot with the existing collector.

Student generation uses the saved Transformers/PEFT setup. Teacher generation
can use a separately installed local vLLM server, with the official tokenizer
and explicitly disabled thinking. Saved-state probes measure action legality;
they are not counted as recovered tasks without executing a continuation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


class TeacherTransport:
    def __init__(self, processor, base_url, model_name, model=None):
        from shopping_grpo.evaluation.local_inference import chat_eos_token_ids
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.model = model
        self.eos_token_ids = chat_eos_token_ids(self.tokenizer)

    def render(self, messages, tools):
        from shopping_grpo.training.sft.dataset import normalize_messages_for_chat_template
        normalized = normalize_messages_for_chat_template(messages)
        if normalized is None:
            raise ValueError("Invalid JSON tool arguments in history")
        return self.processor.apply_chat_template(normalized, tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def count_chat_tokens(self, messages, tools):
        return len(self.tokenizer(self.render(messages, tools), add_special_tokens=False)["input_ids"])

    def count_text_tokens(self, text):
        return len(self.tokenizer(str(text), add_special_tokens=False)["input_ids"])

    def post(self, payload, timeout=600):
        request = Request(self.base_url + "/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def __call__(self, url, payload, headers, timeout):
        from shopping_grpo.evaluation.local_inference import parse_qwen_completion
        del url, headers
        prompt = self.render(payload["messages"], payload["tools"])
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        started = time.monotonic()
        if self.model is not None:
            import torch
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=int(payload["max_tokens"]),
                    do_sample=False, use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.eos_token_ids)
            generated = output[0, len(prompt_ids):]
            message = parse_qwen_completion(self.tokenizer.decode(generated, skip_special_tokens=False))
            message["generation_metadata"].update(actor="teacher", prompt_tokens=len(prompt_ids),
                completion_tokens=int(generated.shape[0]), elapsed_seconds=round(time.monotonic()-started,4),
                max_tokens_reached=generated.shape[0] >= int(payload["max_tokens"]), backend="transformers_fp8")
            return {"choices": [{"message": message}], "usage": {
                "prompt_tokens": len(prompt_ids), "completion_tokens": int(generated.shape[0])}}
        response = self.post({"model": self.model_name, "prompt": prompt_ids,
            "max_tokens": int(payload["max_tokens"]), "temperature": 0.0, "top_p": 1.0,
            "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "stop_token_ids": self.eos_token_ids, "skip_special_tokens": False,
            "spaces_between_special_tokens": False, "seed": 42}, timeout=max(timeout, 600))
        choice = response["choices"][0]
        message = parse_qwen_completion(choice["text"])
        usage = response["usage"]
        message["generation_metadata"].update(actor="teacher", prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"], elapsed_seconds=round(time.monotonic()-started,4),
            max_tokens_reached=choice.get("finish_reason") == "length", backend="vllm_completions")
        return {"choices": [{"message": message}], "usage": usage}

    def score_smoke(self, messages, tools, completion):
        """Check probability feedback for the actual student tokens, not teacher samples."""
        raw = (completion.get("generation_metadata") or {}).get("raw_completion")
        if not raw:
            raise ValueError("Student completion lacks raw token text")
        prompt_ids = self.tokenizer(self.render(messages, tools), add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(raw, add_special_tokens=False)["input_ids"]
        if not response_ids:
            raise ValueError("Empty student completion")
        all_ids = prompt_ids + response_ids
        if self.model is not None:
            import torch
            inputs = torch.tensor([all_ids[:-1]], device="cuda")
            with torch.inference_mode():
                result = self.model(input_ids=inputs, use_cache=False, logits_to_keep=len(response_ids))
                logits = result.logits[0].float()
                targets = torch.tensor(response_ids, device="cuda")
                values = -torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                scored = values.cpu().tolist()
            if not all(math.isfinite(value) for value in scored):
                raise ValueError("Non-finite teacher probabilities for student tokens")
            return {"student_tokens": len(scored), "student_token_ids": response_ids,
                    "teacher_token_logprobs": scored, "mean_teacher_nll": -sum(scored)/len(scored),
                    "backend": "transformers_fp8_forward", "local_token_id_mapping": True,
                    "scope": "Teacher-forced student-text scoring smoke check; not a full-vocabulary KL implementation"}
        response = self.post({"model": self.model_name, "prompt": all_ids,
            "echo": True, "max_tokens": 0, "logprobs": 1, "prompt_logprobs": 1,
            "temperature": 0.0, "top_p": 1.0, "return_tokens_as_token_ids": True})
        probabilities = response["choices"][0].get("logprobs") or {}
        values = probabilities.get("token_logprobs") or []
        if len(values) != len(all_ids):
            raise ValueError("Teacher scoring returned an unexpected number of tokens")
        scored = values[len(prompt_ids):]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scored):
            raise ValueError("Non-finite or missing teacher probabilities for student tokens")
        tokens = probabilities.get("tokens") or []
        if tokens != [f"token_id:{token_id}" for token_id in all_ids]:
            raise ValueError("Teacher endpoint token-ID mapping does not match the local tokenizer")
        return {"student_tokens": len(scored), "student_token_ids": response_ids,
                "teacher_token_logprobs": scored, "mean_teacher_nll": -sum(scored)/len(scored),
                "endpoint_token_alignment": True,
                "scope": "Scoring capability smoke check; not a full-vocabulary KL implementation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor", choices=["student", "teacher", "probe"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--student-model", type=Path)
    parser.add_argument("--student-results", type=Path)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:5800/v1")
    parser.add_argument("--teacher-backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--teacher-name", default="qwen38-teacher")
    parser.add_argument("--shop-url", default="http://127.0.0.1:5700")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"phase": "preparing", "actor": args.actor, "pid": os.getpid(),
              "started_at": datetime.now(timezone.utc).isoformat(), "completed_tasks": 0}
    write(args.output / "status.json", status)
    try:
        manifest = json.loads((args.pilot / "task-manifest.json").read_text(encoding="utf-8"))
        paths = {"train": args.workspace / "data/sft/train.jsonl",
                 "development": args.workspace / "data/sft/validation.jsonl",
                 "final": args.workspace / "data/evaluation/tasks.jsonl"}
        for name, path in paths.items():
            if name == "final" and not path.exists():
                # Remote SFT workspace deliberately does not contain blind task text.
                continue
            assert sha(path) == manifest["sha256"][name], f"Dataset changed: {name}"
        benchmark = args.pilot / "tasks.jsonl"
        assert sha(benchmark) == manifest["benchmark_sha256"]
        tasks = read_rows(benchmark)
        expected_ids = [row["task_id"] for row in tasks]
        assert expected_ids == manifest["task_ids"] and len(set(expected_ids)) == len(tasks) == 50
        source = args.output / "source"
        shutil.copytree(args.workspace / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(__file__, source / "run_pilot.py")
        sys.path.insert(0, str(source / "src"))
        from shopping_grpo.evaluation.blind_guard import guard_blind_final, validate_canonical_blind_asset
        from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_for_task, append_jsonl, _is_infrastructure_failure
        from shopping_grpo.evaluation.summary import summarize_trajectories, _is_strict_success
        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
        blind_guard, blind_ids = validate_canonical_blind_asset()
        assert blind_guard["task_sha256"] == manifest["sha256"]["final"]
        assert not (set(expected_ids) & blind_ids)
        guard_blind_final([benchmark], allowed=False)
        import torch
        from transformers import AutoProcessor
        config = {"actor": args.actor, "model": str(args.model.resolve()),
                  "task_manifest": manifest, "source_sha256": {
                      str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*.py"))},
                  "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "accelerate"]}}
        if args.actor == "student":
            if not args.adapter or not torch.cuda.is_available():
                raise RuntimeError("Student needs its E1 adapter and a live CUDA GPU")
            from transformers import AutoModelForMultimodalLM
            from peft import PeftModel
            from shopping_grpo.evaluation.local_inference import TransformersTransport
            torch.manual_seed(42)
            processor = AutoProcessor.from_pretrained(args.adapter, local_files_only=True)
            base = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                        attn_implementation="sdpa", local_files_only=True)
            model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).to("cuda").eval()
            transport = TransformersTransport(model, processor)
            model_name = "qwen35-sft"
            config.update(adapter=str(args.adapter.resolve()), adapter_sha256=sha(args.adapter / "adapter_model.safetensors"),
                          gpu=torch.cuda.get_device_name(), backend="transformers_peft")
            assert config["adapter_sha256"] == "f191ccd7ad9270e2b9aa8ddc12e4ff8a0bd542ee22e9a008774542b25367f1c6"
        else:
            processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
            model = None
            if args.teacher_backend == "transformers":
                if not torch.cuda.is_available():
                    raise RuntimeError("FP8 teacher needs a live CUDA GPU")
                from transformers import AutoModelForMultimodalLM
                torch.manual_seed(42)
                model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                    device_map="cuda", attn_implementation="sdpa", local_files_only=True).eval()
                model.requires_grad_(False)
                if getattr(model.hf_quantizer.quantization_config, "dequantize", True):
                    raise RuntimeError("FP8 loading silently dequantized; refuse an unrecorded BF16 run")
                config.update(gpu=torch.cuda.get_device_name(), gpu_capability=torch.cuda.get_device_capability(),
                              kernels=importlib.metadata.version("kernels"),
                              peak_torch_memory_gib=round(torch.cuda.max_memory_allocated()/1024**3,3))
            transport = TeacherTransport(processor, args.teacher_url, args.teacher_name, model=model)
            model_name = args.teacher_name
            config.update(backend="transformers_fp8" if model is not None else "vllm_completions", teacher_url=args.teacher_url,
                          model_config_sha256=sha(args.model / "config.json"), teacher_enable_thinking=False)
            if args.student_model:
                student_processor = AutoProcessor.from_pretrained(args.student_model, local_files_only=True)
                equality = processor.tokenizer.get_vocab() == student_processor.tokenizer.get_vocab()
                config["student_token_id_mapping_identical"] = equality
                if not equality:
                    raise RuntimeError("Teacher/student token IDs differ; OPD needs explicit alignment")
            if model is None:
                with urlopen(args.teacher_url + "/models", timeout=30) as response:
                    models = json.load(response)
                assert any(item["id"] == args.teacher_name for item in models["data"])
        config["chat_template_sha256"] = hashlib.sha256(processor.chat_template.encode()).hexdigest()
        config["stop_token_ids"] = transport.eos_token_ids
        write(args.output / "run_config.json", config)
        client = OpenAIChatClient(model=model_name, base_url="local-transport", api_key="LOCAL",
            temperature=0.0, top_p=1.0, max_tokens=512, context_window=24576,
            context_safety_margin=512, context_compaction_enable=False,
            observation_token_budget=1536, observation_detail_token_budget=4096,
            observation_generic_token_budget=768, observation_search_top_k=20,
            token_counter=transport.count_chat_tokens, observation_token_counter=transport.count_text_tokens,
            transport=transport)
        if args.preflight_only:
            status.update(phase="completed", configuration_only=True)
            return
        if args.actor == "probe":
            if not args.student_results:
                raise ValueError("--student-results is required for saved-state probes")
            student_rows = read_rows(args.student_results / "trajectories.jsonl")
            assert [row["task_id"] for row in student_rows] == expected_ids
            probes = []
            for row in student_rows:
                # All failed tasks plus the first five successful tasks, for a control.
                if _is_strict_success(row) and sum(p["kind"] == "success_prebuy_control" for p in probes) >= 5:
                    continue
                messages = row["messages"]
                assistants = [i for i,m in enumerate(messages) if m["role"] == "assistant"]
                if not assistants:
                    continue
                blocked = [i+1 for i,m in enumerate(messages[:-1]) if m.get("runtime_action_guard")]
                # Prefer the first response after a guard rejection; otherwise the final action.
                index = next((i for i in blocked if messages[i]["role"] == "assistant"), assistants[-1])
                prefix, student_completion = messages[:index], messages[index]
                kind = "success_prebuy_control" if _is_strict_success(row) else "student_failure_state"
                teacher_completion = client.complete(prefix, SHOP_TOOL_SCHEMAS)
                from shopping_grpo.environment.actions import action_reject_reason
                latest = next((m["content"] for m in reversed(prefix) if m["role"] == "tool" and not m.get("runtime_action_guard")), "")
                if not latest:
                    from shopping_grpo.environment.observation import render_structured_observation
                    initial = row["initial_result"]
                    latest = render_structured_observation(initial["observation_state"]) if initial.get("observation_state") else initial.get("observation", initial.get("instruction", ""))
                calls = teacher_completion.get("tool_calls", [])
                reason = "no_tool_call" if not calls else ("multiple_tool_calls" if len(calls) != 1 else None)
                if reason is None:
                    function = calls[0]["function"]
                    reason = action_reject_reason(function["name"], json.loads(function["arguments"]), latest)
                probe = {"task_id": row["task_id"], "message_index": index, "kind": kind,
                         "prefix": prefix, "student_completion": student_completion,
                         "teacher_completion": teacher_completion, "teacher_action_reject_reason": reason,
                         "teacher_action_legal": reason is None, "continuation_executed": False,
                         "student_token_scoring": transport.score_smoke(prefix, SHOP_TOOL_SCHEMAS, student_completion)}
                append_jsonl(args.output / "state-probes.jsonl", [probe])
                probes.append(probe)
                status.update(phase="probing", completed_tasks=len(probes), active_task_id=row["task_id"])
                write(args.output / "status.json", status)
            write(args.output / "probe-summary.json", {"states": len(probes),
                  "state_kinds": dict(Counter(p["kind"] for p in probes)),
                  "legal_teacher_actions": sum(p["teacher_action_legal"] for p in probes),
                  "scope": "Action legality and probability scoring only; recovery success not measured"})
        else:
            results = []
            for task in tasks:
                status.update(phase="evaluating", active_task_id=task["task_id"])
                write(args.output / "status.json", status)
                task_start = time.monotonic()
                trajectory = collect_for_task(task, client=client, base_url=args.shop_url, max_steps=35)
                trajectory.update(actor=args.actor, elapsed_seconds=round(time.monotonic()-task_start,3))
                append_jsonl(args.output / "trajectories.jsonl", [trajectory])
                results.append(trajectory)
                summary = summarize_trajectories(expected_ids, results)
                write(args.output / "summary.json", summary)
                status.update(completed_tasks=len(results), strict_successes=summary["strict_successes"],
                              elapsed_minutes=round((time.monotonic()-started)/60,2))
                if torch.cuda.is_available():
                    status["peak_torch_memory_gib"] = round(torch.cuda.max_memory_allocated()/1024**3,3)
                write(args.output / "status.json", status)
                print("TASK", args.actor, len(results), len(tasks), task["task_id"], trajectory["status"],
                      "strict", _is_strict_success(trajectory), "seconds", trajectory["elapsed_seconds"], flush=True)
                if _is_infrastructure_failure(trajectory) or (trajectory.get("error") and trajectory["error"]["type"] != "ContextBudgetError"):
                    raise RuntimeError("Pilot stopped on infrastructure or unexpected inference failure")
        status.update(phase="completed", finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        status.update(phase="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["elapsed_minutes"] = round((time.monotonic()-started)/60,2)
        write(args.output / "status.json", status)


if __name__ == "__main__":
    main()
