"""Independent CPU preparation, student rollout, teacher scoring, and updates."""
import argparse
from collections import Counter
from copy import deepcopy
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from common import (append, checkpoint_audit, choose_training_ids, extract_teacher_scores,
                    importance_loss, now, read, rows, sha, write, verify_initial_adapter, manifest_training_ids)


E1_SHA = "f191ccd7ad9270e2b9aa8ddc12e4ff8a0bd542ee22e9a008774542b25367f1c6"
DATA_SHA = {"train": "8c3a6ff0033f6ea672af609891e747d60652ddc17e8d3c8eacb19e9d96dd9477",
            "development": "9525cc2fb04a1d8d38ae2db959397da908dde3fea766f580fdcf77d1239533cc"}


def prepare(args):
    if args.training_tasks <= 0 or args.tasks_per_update <= 0 or args.training_tasks % args.tasks_per_update:
        raise ValueError("Training task count must be a positive whole number of fresh batches")
    workspace, run = args.workspace.resolve(), args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / "run_config.json").exists()
    reference = args.reference.resolve()
    cal = args.calibration.resolve()
    cal_result = cal / "run-20261013-1/result"
    cal_config = read(cal_result / "run_config.json")
    ref_config = read(reference / "run_config.json")
    initial_sha = verify_initial_adapter(args.adapter, ref_config, args.expected_parent_sha256 or E1_SHA)
    for key, file in [("train", "train.jsonl"), ("development", "validation.jsonl")]:
        assert sha(workspace / "data/sft" / file) == DATA_SHA[key]
    source = run / "source"
    shutil.copytree(reference / "source/src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for file in (source / "src").rglob("*.py"):
        relative = file.relative_to(source / "src")
        assert sha(file) == sha(workspace / "src" / relative), relative
    shutil.copytree(Path(__file__).parent, source / "opd-pilot", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(cal_result / "source/run_pilot.py", source / "run_pilot.py")
    shutil.copy2(cal / "calibration_prompt.txt", source / "calibration_prompt.txt")
    assert sha(source / "calibration_prompt.txt") == cal_config["calibration_prompt_sha256"]
    # Evaluation implementation is frozen from E1. Only reference descriptions
    # change from its inherited uniform-SFT label to the actual E1 reference.
    eval_text = (reference / "source/round_task_eval.py").read_text(encoding="utf-8")
    eval_text = eval_text.replace("saved uniform-SFT", "saved E1-SFT")
    eval_text = eval_text.replace("Original-data uniform SFT", "E1 segment-weighted SFT")
    (source / "evaluate_candidate.py").write_text(eval_text, encoding="utf-8")
    sys.path.insert(0, str(source / "src"))
    from shopping_grpo.evaluation.blind_guard import guard_blind_final, validate_canonical_blind_asset
    from shopping_grpo.training.sft.dataset import build_supervised_example
    from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
    blind, final_ids = validate_canonical_blind_asset()
    assert blind["task_sha256"] == cal_config["task_manifest"]["sha256"]["final"]
    training = rows(workspace / "data/sft/train.jsonl")
    development = rows(workspace / "data/sft/validation.jsonl")
    task_prefix = []
    prefix_config = None
    if args.task_prefix_run:
        prefix_config = read(args.task_prefix_run / "run_config.json")
        assert prefix_config["initial_adapter_sha256"] == initial_sha
        assert prefix_config["dataset_sha256"] == DATA_SHA
        assert prefix_config["sampling_seed"] == args.sampling_seed
        assert prefix_config["selection_seed"] == args.selection_seed
        assert prefix_config["tasks_per_update"] == args.tasks_per_update
        assert prefix_config["sampled_turns_per_task"] == 2
        assert prefix_config["calibration_prompt_sha256"] == cal_config["calibration_prompt_sha256"]
        task_prefix = prefix_config["training_task_ids"]
    selected = choose_training_ids([r["task_id"] for r in training],
                                  [r["task_id"] for r in development], final_ids,
                                  cal_config["task_manifest"]["task_ids"], args.selection_seed,
                                  args.training_tasks, prefix_ids=task_prefix)
    if args.training_task_manifest:
        if args.task_prefix_run:
            raise ValueError('choose a frozen task manifest or a prefix run, not both')
        selected = manifest_training_ids(args.training_task_manifest, [r['task_id'] for r in training],
            [r['task_id'] for r in development], final_ids, cal_config['task_manifest']['task_ids'],
            args.selection_seed, args.training_tasks)
    assert len(training) == 800 and len(development) == 200
    by_id = {r["task_id"]: r for r in training}
    (run / "input").mkdir()
    for name in ["benchmark.jsonl", "run_config.json", "sft-summary.json", "sft-trajectories.jsonl"]:
        shutil.copy2(reference / name, run / "input" / name)
    assert sha(run / "input/benchmark.jsonl") == "624b6ee0513cfeb0e2e7efe2a21db7658d020419ceb7a6946937403e0cac5046"
    selected_file = run / "input/training-tasks.jsonl"
    for task_id in selected:
        append(selected_file, {"task_id": task_id})
    guard_blind_final([selected_file], allowed=False)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.adapter, local_files_only=True)
    teacher_processor = AutoProcessor.from_pretrained(args.teacher, local_files_only=True)
    assert processor.tokenizer.get_vocab() == teacher_processor.tokenizer.get_vocab()
    assert all(processor.tokenizer.convert_tokens_to_ids(x) == teacher_processor.tokenizer.convert_tokens_to_ids(x)
               for x in ["<|im_end|>", "<tool_call>", "</tool_call>"])
    assert sha(args.teacher / "chat_template.jinja") == cal_config["teacher_configuration"]["chat_template_sha256"]
    for entry in cal_config["official_metadata"]["files"]:
        assert sha(args.teacher / entry["name"]) == entry["sha256"]
    anchors = []
    for task_id in selected:
        row = by_id[task_id]
        assert row["tools"] == SHOP_TOOL_SCHEMAS
        indices = [i for i, m in enumerate(row["messages"]) if m["role"] == "assistant"]
        index = random.Random(args.selection_seed + task_id).choice(indices)
        messages = row["messages"][:index + 1]
        example = build_supervised_example(messages, row["tools"], processor.tokenizer,
                                           max_length=28672, chat_template=processor, loss_mode="smartad")
        if example is None:
            raise ValueError(f"Original SFT anchor cannot be rendered: {task_id}")
        # Only the chosen last assistant turn is supervised in this pilot.
        # Locate its start using the same common-prefix rule as the SFT code.
        from shopping_grpo.training.sft.dataset import normalize_messages_for_chat_template
        normalized = normalize_messages_for_chat_template(messages)
        text = processor.apply_chat_template(normalized[:-1], tools=row["tools"], tokenize=False,
                                            add_generation_prompt=True)
        prefix = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        start = 0
        for left, right in zip(prefix, example["input_ids"]):
            if left != right:
                break
            start += 1
        response_ids = example["input_ids"][start:]
        weights = example["loss_weights"][start:]
        assert response_ids and len(response_ids) <= 512 and all(w > 0 for w in weights)
        anchors.append({"task_id": task_id, "message_index": index,
                        "prompt_ids": example["input_ids"][:start],
                        "sampled_ids": response_ids, "weights": weights})
    write(run / "input/anchors.json", anchors)
    if prefix_config:
        assert anchors[:len(task_prefix)] == read(args.task_prefix_run / "input/anchors.json")
        assert protocol_prefix_matches(ref_config, prefix_config)
    packages = {n: importlib.metadata.version(n) for n in ["torch", "transformers", "peft", "accelerate"]}
    assert packages == cal_config["client_packages"]
    protocol = {k: ref_config["protocol"][k] for k in ["max_steps", "max_tokens", "temperature", "top_p",
                "context_window", "context_safety_margin", "context_compaction", "observation_token_budget",
                "observation_detail_token_budget", "observation_generic_token_budget", "observation_search_top_k"]}
    config = {"schema": "shop-opd-pilot-v1", "created_at": now(), "workspace": str(workspace),
              "model": str(args.model.resolve()), "teacher": str(args.teacher.resolve()),
              "initial_adapter": str(args.adapter.resolve()), "initial_adapter_sha256": initial_sha,
              "reference": str(reference), "calibration": str(cal), "training_task_ids": selected,
              "sampling_seed": args.sampling_seed, "selection_seed": args.selection_seed,
              "optimizer_steps": args.training_tasks // args.tasks_per_update,
              "tasks_per_update": args.tasks_per_update, "sampled_turns_per_task": 2,
              "budget_extension_reference": str(args.task_prefix_run.resolve()) if args.task_prefix_run else None,
              "budget_extension_prefix_tasks": len(task_prefix),
              "learning_rate": 5e-6, "sft_anchor_coefficient": 0.1, "max_grad_norm": 1.0,
              "dropout_during_training": 0.0, "optimizer": "AdamW", "weight_decay": 0.0,
              "rollout_temperature": 1.0, "rollout_top_p": 1.0, "rollout_top_k": 0,
              "one_update_per_fresh_policy_batch": True, "discount_factor": 0.0,
              "loss": "sampled immediate-token reverse-KL importance-sampling surrogate + 0.1 original SFT CE",
              "full_vocabulary_kl": False, "reward_filtering": False, "teacher_takeover": False,
              "control": "E1 + the identical 0.1 original-SFT anchor term and optimizer schedule, without OPD",
              "protocol": protocol, "packages": packages, "teacher_configuration": cal_config["teacher_configuration"],
              "official_metadata": cal_config["official_metadata"],
              "calibration_prompt_sha256": sha(source / "calibration_prompt.txt"),
              "dataset_sha256": DATA_SHA, "blind_final_sha256": blind["task_sha256"],
              "benchmark_sha256": sha(run / "input/benchmark.jsonl"),
              "training_tasks_sha256": sha(selected_file), "anchors_sha256": sha(run / "input/anchors.json"),
              "source_sha256": {str(p.relative_to(source)): sha(p) for p in source.rglob("*.py")},
              "source_reference_checks_passed": True, "student_teacher_token_id_mapping_identical": True,
              "training_task_manifest_sha256": sha(args.training_task_manifest) if args.training_task_manifest else None,
              "opd_method_source": "https://thinkingmachines.ai/blog/on-policy-distillation/",
              "final_evaluation_started": False}
    write(run / "run_config.json", config)
    write(run / "preparation-status.json", {"phase": "completed", "training_tasks": len(selected),
          "anchors": len(anchors), "max_anchor_context": max(len(x["prompt_ids"]) + len(x["sampled_ids"]) for x in anchors),
          "no_train_development_or_final_overlap": True, "packages_and_metadata_verified": True})
    print("PREPARATION_PASSED", len(selected), flush=True)


def protocol_prefix_matches(reference, prefix):
    return all(reference["protocol"][k] == v for k, v in prefix["protocol"].items())


def load_student(config, adapter, training=False, fast=False):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from peft import PeftModel
    if not fast:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as delta
        delta.causal_conv1d_fn = delta.causal_conv1d_update = None
        delta.chunk_gated_delta_rule = delta.fused_recurrent_gated_delta_rule = None
        delta.FusedRMSNormGated = None
        delta.is_fast_path_available = False
    torch.manual_seed(config["sampling_seed"])
    processor = AutoProcessor.from_pretrained(adapter, local_files_only=True)
    base = AutoModelForMultimodalLM.from_pretrained(config["model"], dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=training, local_files_only=True).to("cuda")
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.peft_config["default"].lora_dropout = 0.0
    else:
        model.eval()
    return model, processor


def make_client(transport, config, model_name="qwen35-sft", temperature=0.0):
    from shopping_grpo.evaluation.rollout import OpenAIChatClient
    p = config["protocol"]
    return OpenAIChatClient(model=model_name, base_url="local-transport", api_key="LOCAL",
        temperature=temperature, top_p=1.0, max_tokens=p["max_tokens"],
        context_window=p["context_window"], context_safety_margin=p["context_safety_margin"],
        context_compaction_enable=p["context_compaction"], observation_token_budget=p["observation_token_budget"],
        observation_detail_token_budget=p["observation_detail_token_budget"],
        observation_generic_token_budget=p["observation_generic_token_budget"],
        observation_search_top_k=p["observation_search_top_k"], token_counter=transport.count_chat_tokens,
        observation_token_counter=transport.count_text_tokens, transport=transport)


def rollout(args, config):
    import torch
    from transformers import GenerationConfig
    from shopping_grpo.evaluation.local_inference import TransformersTransport, parse_qwen_completion
    from shopping_grpo.evaluation.rollout import collect_for_task, _is_infrastructure_failure
    from shopping_grpo.evaluation.summary import summarize_trajectories
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    model, processor = load_student(config, args.adapter)
    assert sha(args.adapter / "adapter_model.safetensors") == args.policy_sha
    torch.manual_seed(config["sampling_seed"] + args.update)

    class RecordedTransport(TransformersTransport):
        def __init__(self, model, processor):
            super().__init__(model, processor)
            self.records = []

        def __call__(self, url, payload, headers, timeout):
            prompt = self.render(payload["messages"], payload["tools"])
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            inputs = {"input_ids": torch.tensor([prompt_ids], device="cuda"),
                      "attention_mask": torch.ones(1, len(prompt_ids), device="cuda", dtype=torch.long)}
            generation = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                max_new_tokens=int(payload["max_tokens"]), use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.eos_token_ids,
                return_dict_in_generate=True, output_scores=True)
            before = time.monotonic()
            with torch.inference_mode():
                generated = self.model.generate(**inputs, generation_config=generation)
            ids = generated.sequences[0, len(prompt_ids):].tolist()
            assert len(ids) == len(generated.scores)
            logp = [float(score[0].float().log_softmax(-1)[token].cpu())
                    for score, token in zip(generated.scores, ids)]
            raw = self.tokenizer.decode(ids, skip_special_tokens=False)
            message = parse_qwen_completion(raw)
            message["generation_metadata"].update(actor="opd_student", prompt_tokens=len(prompt_ids),
                completion_tokens=len(ids), elapsed_seconds=round(time.monotonic() - before, 4),
                max_tokens_reached=len(ids) >= int(payload["max_tokens"]), sampled_token_ids=ids,
                behavior_logprobs=logp, temperature=1.0, top_p=1.0, top_k=0,
                policy_adapter_sha256=args.policy_sha)
            self.records.append({"messages": deepcopy(payload["messages"]), "prompt_ids": prompt_ids,
                                 "sampled_ids": ids, "old_logp": logp, "raw_completion": raw,
                                 "policy_adapter_sha256": args.policy_sha})
            return {"choices": [{"message": message}], "usage": {
                    "prompt_tokens": len(prompt_ids), "completion_tokens": len(ids)}}

    transport = RecordedTransport(model, processor)
    client = make_client(transport, config, temperature=1.0)
    batch = config["tasks_per_update"]
    ids = config["training_task_ids"][args.update * batch:(args.update + 1) * batch]
    if args.smoke:
        ids = ids[:2]
    collected, selected = [], []
    for task_id in ids:
        transport.records = []
        trajectory = collect_for_task({"task_id": task_id}, client=client, base_url=args.shop_url,
                                      max_steps=config["protocol"]["max_steps"])
        append(output / "trajectories.jsonl", trajectory)
        collected.append(trajectory)
        if _is_infrastructure_failure(trajectory) or (trajectory.get("error") and trajectory["error"]["type"] != "ContextBudgetError"):
            raise RuntimeError(f"Unexpected rollout error in {task_id}: {trajectory.get('error')}")
        indices = random.Random(config["selection_seed"] + task_id + 1000 * args.update).sample(
            range(len(transport.records)), min(2, len(transport.records)))
        if not indices:
            raise ValueError(f"No sampled student turn in task {task_id}")
        for index in sorted(indices):
            record = transport.records[index]
            record.update(task_id=task_id, turn_index=index, update=args.update)
            append(output / "selected-turns.jsonl", record)
            selected.append(record)
        summary = summarize_trajectories(ids, collected)
        write(output / "summary.json", summary)
        write(output / "status.json", {"phase": "rolling_out", "completed_tasks": len(collected),
              "expected_tasks": len(ids), "selected_turns": len(selected), "strict_successes": summary["strict_successes"]})
        print("STUDENT_ROLLOUT", args.update, len(collected), task_id, trajectory["status"], flush=True)
    write(output / "status.json", {"phase": "completed", "completed_tasks": len(collected),
          "selected_turns": len(selected), "policy_sha256": args.policy_sha,
          "peak_torch_memory_gib": torch.cuda.max_memory_allocated() / 1024**3})


def score(args, config):
    from transformers import AutoProcessor
    from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
    source = args.run / "source"
    spec = importlib.util.spec_from_file_location("frozen_pilot", source / "run_pilot.py")
    pilot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot_module)
    processor = AutoProcessor.from_pretrained(config["teacher"], local_files_only=True)
    appendix = (source / "calibration_prompt.txt").read_text(encoding="utf-8")
    assert sha(source / "calibration_prompt.txt") == config["calibration_prompt_sha256"]

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

    # The same prompt injection class used in the calibration and dev200 runs.
    import ast
    old = ast.parse((Path(config["calibration"]) / "run-20261013-1/result/source/run_calibration.py").read_text())
    new = ast.parse(Path(__file__).read_text())
    extract = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "CalibratedTransport")
    assert ast.dump(extract(old), include_attributes=False) == ast.dump(extract(new), include_attributes=False)
    transport = CalibratedTransport(appendix)
    args.output.mkdir(parents=True, exist_ok=False)
    packets = rows(args.input / "selected-turns.jsonl")
    for index, packet in enumerate(packets):
        prompt_ids = transport.tokenizer(transport.render(packet["messages"], SHOP_TOOL_SCHEMAS),
                                        add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + packet["sampled_ids"]
        if len(full_ids) > config["protocol"]["context_window"]:
            raise ValueError("Teacher scoring context exceeds the frozen budget")
        response = transport.post({"model": "qwen38-teacher", "prompt": full_ids,
            "echo": True, "max_tokens": 0, "logprobs": 1, "prompt_logprobs": 1,
            "temperature": 0.0, "top_p": 1.0, "return_tokens_as_token_ids": True})
        teacher_logp = extract_teacher_scores(response, prompt_ids, packet["sampled_ids"])
        packet.update(teacher_logp=teacher_logp, teacher_prompt_tokens=len(prompt_ids),
                      endpoint_token_alignment=True, actual_sampled_ids_used=True)
        append(args.output / "scored-turns.jsonl", packet)
        write(args.output / "status.json", {"phase": "scoring", "completed_turns": index + 1,
              "expected_turns": len(packets), "active_task_id": packet["task_id"]})
        print("TEACHER_SCORE", args.update, index + 1, len(packets), packet["task_id"], flush=True)
    write(args.output / "status.json", {"phase": "completed", "scored_turns": len(packets),
          "scored_tokens": sum(len(p["sampled_ids"]) for p in packets),
          "all_token_id_alignments_verified": True})


def response_logp(model, packet):
    """Project only assistant response positions, never the full prompt logits."""
    import torch
    import torch.nn.functional as F
    ids = packet["prompt_ids"] + packet["sampled_ids"]
    length = len(packet["sampled_ids"])
    if not packet["prompt_ids"] or not length:
        raise ValueError("Prompt and response cannot be empty")
    inputs = torch.tensor([ids[:-1]], device="cuda", dtype=torch.long)
    output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False,
                   logits_to_keep=length, return_dict=True)
    if output.logits.shape[:2] != (1, length):
        raise ValueError("Causal response projection has the wrong token count")
    targets = torch.tensor(packet["sampled_ids"], device="cuda", dtype=torch.long)
    return -F.cross_entropy(output.logits[0].float(), targets, reduction="none")


def train(args, config):
    import torch
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    previous = sha(args.adapter / "adapter_model.safetensors")
    model, processor = load_student(config, args.adapter, training=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.0)
    if args.optimizer_state:
        # This path is created by our own previous update, not an external pickle.
        optimizer.load_state_dict(torch.load(args.optimizer_state, map_location="cpu", weights_only=True))
        assert {int(s["step"].item()) for s in optimizer.state.values()} == {args.update}
    elif not args.control and not args.smoke and args.update != 0:
        raise ValueError("A later OPD update requires its immediately preceding optimizer state")
    anchor_by_id = {p["task_id"]: p for p in read(args.run / "input/anchors.json")}
    if args.control:
        updates = range(config["optimizer_steps"])
    else:
        updates = [args.update]
    log = []
    for update in updates:
        batch = config["tasks_per_update"]
        ids = config["training_task_ids"][update * batch:(update + 1) * batch]
        if args.smoke:
            ids = ids[:2]
        anchors = [anchor_by_id[x] for x in ids]
        packets = [] if args.control else rows(args.input / "scored-turns.jsonl")
        if not args.control:
            assert {p["task_id"] for p in packets} == set(ids)
            assert all(p["policy_adapter_sha256"] == previous for p in packets)
            assert all(p["endpoint_token_alignment"] and p["actual_sampled_ids_used"] for p in packets)
            assert all(len(p["old_logp"]) == len(p["teacher_logp"]) == len(p["sampled_ids"]) for p in packets)
        optimizer.zero_grad(set_to_none=True)
        tokens = sum(len(p["sampled_ids"]) for p in packets)
        anchor_weight = sum(sum(p["weights"]) for p in anchors)
        statistics = {"update": update, "opd_tokens": tokens, "opd_turns": len(packets),
                      "anchor_tasks": len(anchors), "anchor_tokens": sum(len(p["sampled_ids"]) for p in anchors),
                      "anchor_weight_sum": anchor_weight, "surrogate_loss": 0.0,
                      "sampled_reverse_kl": 0.0, "weighted_sft_nll": 0.0,
                      "behavior_forward_abs_error_sum": 0.0, "behavior_forward_abs_error_max": 0.0}
        before = time.monotonic()
        for index, packet in enumerate(packets):
            current = response_logp(model, packet)
            old = torch.tensor(packet["old_logp"], device="cuda", dtype=torch.float32)
            teacher = torch.tensor(packet["teacher_logp"], device="cuda", dtype=torch.float32)
            discrepancy = (current.detach() - old).abs()
            statistics["behavior_forward_abs_error_sum"] += discrepancy.sum().item()
            statistics["behavior_forward_abs_error_max"] = max(statistics["behavior_forward_abs_error_max"], discrepancy.max().item())
            # BF16 cached generation and full causal forward can differ slightly.
            # Refuse a large discrepancy before taking any optimizer step.
            if discrepancy.mean().item() > 0.05 or discrepancy.max().item() > 0.5:
                raise ValueError("Behavior versus training log probabilities disagree; update not applied")
            losses, kl = importance_loss(current, old, teacher)
            loss = losses.sum() / tokens
            if not torch.isfinite(loss):
                raise ValueError("Non-finite OPD surrogate")
            statistics["surrogate_loss"] += loss.detach().item()
            statistics["sampled_reverse_kl"] += kl.sum().item() / tokens
            loss.backward()
            del loss, losses, kl, current, old, teacher
            write(output / "status.json", {"phase": "backpropagating_opd", "update": update,
                  "completed_turns": index + 1, "expected_turns": len(packets)})
            print("OPD_BACKWARD", update, index + 1, len(packets), flush=True)
        for index, anchor in enumerate(anchors):
            logp = response_logp(model, anchor)
            weights = torch.tensor(anchor["weights"], device="cuda", dtype=torch.float32)
            ce = -(logp * weights).sum() / anchor_weight
            statistics["weighted_sft_nll"] += ce.detach().item()
            (config["sft_anchor_coefficient"] * ce).backward()
            del ce, logp, weights
            write(output / "status.json", {"phase": "backpropagating_anchor", "update": update,
                  "completed_anchors": index + 1, "expected_anchors": len(anchors)})
            print("SFT_ANCHOR_BACKWARD", update, index + 1, len(anchors), flush=True)
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError("Missing or non-finite training gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
        statistics["gradient_norm_before_clip"] = norm.item()
        optimizer.step()
        statistics["elapsed_seconds"] = time.monotonic() - before
        statistics["behavior_forward_abs_error_mean"] = statistics["behavior_forward_abs_error_sum"] / tokens if tokens else None
        statistics["peak_torch_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        log.append(statistics)
        write(output / "training-log.json", log)
        print("OPTIMIZER_UPDATE", json.dumps(statistics), flush=True)
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    processor.save_pretrained(adapter)
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    optimizer_steps = sorted({int(s["step"].item()) for s in optimizer.state.values()})
    assert optimizer_steps == [config["optimizer_steps"] if args.control else args.update + 1]
    assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values()
               if isinstance(v, torch.Tensor))
    audit = checkpoint_audit(adapter, previous)
    write(output / "adapter-audit.json", audit)
    write(output / "status.json", {"phase": "completed", "optimizer_updates": len(log),
          "adapter_sha256": audit["sha256"], "previous_adapter_sha256": previous,
          "gradient_and_adapter_finite": True, "training_log": log,
          "optimizer_step_values": optimizer_steps, "optimizer_state_entries": len(optimizer.state),
          "optimizer_state_finite": True, "optimizer_sha256": sha(output / "optimizer.pt"),
          "peak_torch_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
          "smoke_only": args.smoke, "control": args.control})


def analyse(args, config):
    from shopping_grpo.evaluation.summary import summarize_trajectories
    ids = [r["task_id"] for r in rows(args.run / "input/benchmark.jsonl")]
    actors = {"E1": rows(args.run / "input/sft-trajectories.jsonl"),
              "SFT_control": rows(args.run / "evaluation/control/sft-trajectories.jsonl"),
              "OPD": rows(args.run / "evaluation/opd/sft-trajectories.jsonl")}
    summaries, sets = {}, {}
    for name, records in actors.items():
        assert [r["task_id"] for r in records] == ids and len(set(ids)) == len(records) == 200
        assert not any(r.get("release_error") for r in records)
        assert all((r.get("error") or {}).get("type") in [None, "ContextBudgetError"] for r in records)
        summary = summarize_trajectories(ids, records)
        sets[name] = set(summary["strict_success_task_ids"])
        summaries[name] = summary
    assert summaries["E1"]["strict_successes"] == 178
    paired = {}
    for left, right in [("E1", "SFT_control"), ("E1", "OPD"), ("SFT_control", "OPD")]:
        gained, lost = sorted(sets[right] - sets[left]), sorted(sets[left] - sets[right])
        n = len(gained) + len(lost)
        p = min(1, 2 * sum(math.comb(n, j) for j in range(min(len(gained), len(lost)) + 1)) / 2**n) if n else 1.0
        common = sets[left] & sets[right]
        paired[f"{left}_to_{right}"] = {"gained": gained, "lost": lost,
            "both_success": sorted(common), "both_fail": sorted(set(ids) - sets[left] - sets[right]),
            "percentage_point_difference": (len(sets[right]) - len(sets[left])) / 2,
            "exact_mcnemar_two_sided_p": p,
            "common_success_mean_steps": {name: sum(len(r["steps"]) for r in actors[name] if r["task_id"] in common) / len(common)
                                          if common else None for name in [left, right]}}
    write(args.run / "comparison.json", {"summaries": summaries, "paired": paired,
          "scope": f"{config['optimizer_steps']}-update sampled reverse-KL OPD; development set; one seed; not final generalization",
          "final_evaluation_started": False})
    print("COMPARISON", {name: s["strict_successes"] for name, s in summaries.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "rollout", "score", "train", "analyse"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--expected-parent-sha256", help="Explicit digest for a newly trained and evaluated SFT parent")
    parser.add_argument('--training-task-manifest', type=Path, help='Use these exact eligible IDs in the saved order')
    parser.add_argument("--training-tasks", type=int, default=64)
    parser.add_argument("--tasks-per-update", type=int, default=8)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=20261014)
    parser.add_argument("--task-prefix-run", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--update", type=int, default=0)
    parser.add_argument("--policy-sha")
    parser.add_argument("--optimizer-state", type=Path)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shop-url", default="http://127.0.0.1:5700")
    parser.add_argument("--teacher-url", default="http://127.0.0.1:5800/v1")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
        return
    config = read(args.run / "run_config.json")
    sys.path.insert(0, str(args.run / "source/src"))
    for relative, digest in config["source_sha256"].items():
        assert sha(args.run / "source" / relative) == digest, relative
    for name, digest in config["dataset_sha256"].items():
        filename = "train.jsonl" if name == "train" else "validation.jsonl"
        assert sha(Path(config["workspace"]) / "data/sft" / filename) == digest
    if args.phase == "rollout":
        rollout(args, config)
    elif args.phase == "score":
        score(args, config)
    elif args.phase == "train":
        train(args, config)
    else:
        analyse(args, config)


if __name__ == "__main__":
    main()
