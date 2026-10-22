#!/usr/bin/env python3
"""Join dispatch/completion audit to native admission and optimizer receipts."""
import argparse
from collections import Counter
import json
from fractions import Fraction
from pathlib import Path


def summarize(run):
    run = Path(run)
    events = [json.loads(s) for s in (run / "feedback_sampling.jsonl").read_text(encoding="utf-8").splitlines() if s]
    configuration = [r for r in events if r['stage'] == 'configuration']
    if len(configuration) != 1:
        raise ValueError('missing or duplicate sampling configuration')
    stages = {name: {} for name in ("dispatch", "finished", "cancel_requested", "cancelled", "consumed", "admitted", "skipped", "optimizer_used")}
    for event in events:
        stage = event["stage"]
        if stage == "configuration":
            continue
        choice = event["choice"]
        if event.get('failed'):
            raise ValueError('rollout worker failed; inspect the original error before reporting readiness')
        uid = choice["uid"]
        if stage not in stages or uid in stages[stage]:
            raise ValueError("unknown or duplicate sampler audit event")
        if stage != "dispatch" and stages["dispatch"].get(uid) != choice:
            raise ValueError("completion audit differs from dispatched choice")
        stages[stage][uid] = choice
    dispatch = sorted(stages['dispatch'].values(), key=lambda c: c['dispatch_index'])
    if [c['dispatch_index'] for c in dispatch] != list(range(1, len(dispatch) + 1)):
        raise ValueError('dispatch sequence differs')
    fraction = Fraction(str(configuration[0]['hard_fraction']))
    if sum(c['lane'] == 'hard' for c in dispatch) != int(len(dispatch) * fraction):
        raise ValueError('hard-lane dispatch quota differs')
    if not set(stages['consumed']) <= set(stages['finished']):
        raise ValueError('consumed rows lack worker completion evidence')
    if not set(stages['cancelled']) <= set(stages['cancel_requested']) or set(stages['cancel_requested']) & set(stages['consumed']):
        raise ValueError('cancelled rows lack shutdown intent or were reused')
    diagnostics = [json.loads(s) for s in (run / "training_diagnostics.jsonl").read_text(encoding="utf-8").splitlines() if s]
    admissions, optimizer_steps = [], []
    for event in diagnostics:
        if event["event"] not in ("flash_batch_admission", "optimizer_step"):
            continue
        rows = event["rollouts"]
        if len({r["task_id"] for r in rows}) != len(rows):
            raise ValueError("native batch contains duplicate tasks")
        if event["event"] == "flash_batch_admission":
            admissions.append(event)
            stage = "admitted" if event["decision"] == "admit" else "skipped"
        else:
            optimizer_steps.append(event["global_step"])
            stage = "optimizer_used"
        for row in rows:
            uid, choice = row["uid"], row["feedback_sampling"]
            if stages["consumed"].get(uid) != choice or uid in stages[stage]:
                raise ValueError("native admission/optimizer receipt is absent, changed or reused")
            stages[stage][uid] = choice
    if set(stages["admitted"]) & set(stages["skipped"]) or set(stages["admitted"]) | set(stages["skipped"]) != set(stages["consumed"]):
        raise ValueError("consumed trajectories do not match native admission receipts")
    if set(stages["admitted"]) != set(stages["optimizer_used"]):
        raise ValueError("admitted rows and committed optimizer rows differ; possible interrupted update")
    if optimizer_steps != list(range(1, len(optimizer_steps) + 1)):
        raise ValueError("optimizer update sequence differs")
    counts = {}
    for stage, records in stages.items():
        n = len(records)
        counts[stage] = {"total": n, "unique_tasks": len({v["task_id"] for v in records.values()}),
            "hard_lane": sum(c["lane"] == "hard" for c in records.values()),
            "hard_members": sum(c["hard_member"] for c in records.values())}
        counts[stage]["hard_lane_fraction"] = counts[stage]["hard_lane"] / n if n else None
        counts[stage]["hard_member_fraction"] = counts[stage]["hard_members"] / n if n else None
    return {"verified": True, "stages": counts, "candidate_batches": len(admissions),
            "optimizer_updates": len(optimizer_steps), "backlog_unconsumed": len(stages["dispatch"]) - len(stages["consumed"]),
            "note": "optimizer_used is whole-batch optimizer input; Flash sequence KL gate may give individual rows zero gradient"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
