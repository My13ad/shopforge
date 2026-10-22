"""Real pinned fit/CPU optimizer and queue barriers for validation-driven stopping."""
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from shopping_grpo.training.grpo.async_sampling import CompletionPool, Submitted
from shopping_grpo.training.grpo.checkpoint_selection import TrainingCheckpointSelection
from shopping_grpo.training.grpo.diagnostics import persist_unpadded_validation_batch
from test_checkpoint_selection import infos
from test_flash_batch_admission import candidate, native_harness
from test_grpo_entry_and_hook import args, entry, parent  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def test_pause_drains_pending_when_fifo_is_full_without_losing_or_reusing_items():
    submitted, first_two, entered, release = [], Event(), Event(), Event()

    def submit(item):
        future = Future()
        submitted.append((item, future))
        if item >= 2:
            future.set_result(item)
        if len(submitted) == 2:
            first_two.set()
        return Submitted(future, future.cancel)

    pool = CompletionPool(range(6), submit, inflight=2, capacity=1, timeout=3)
    assert first_two.wait(2)
    errors = []

    def validate():
        try:
            with pool.pause_dispatch():
                assert len(submitted) == 2 and all(f.done() for _, f in submitted)
                entered.set()
                assert release.wait(2)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=validate)
    thread.start()
    assert pool.paused.wait(2)
    for item, future in list(submitted):
        future.set_result(item)
    assert entered.wait(2), "validation must not wait for space in the full completed FIFO"
    assert len(submitted) == 2
    release.set()
    thread.join(2)
    assert not thread.is_alive() and not errors
    results = [output for _, output, _ in pool]
    pool.close()
    assert results == list(range(6)) and pool.delivered == pool.submitted == 6


@pytest.mark.parametrize("bad_validation", [False, True])
def test_real_native_fit_stops_cpu_optimizer_at_validation_boundary_and_retains_best(monkeypatch, tmp_path, bad_validation):
    run = native_harness(monkeypatch, tmp_path, [candidate() for _ in range(8)], target=6, execution="async")
    trainer = run.trainer
    c = trainer.config
    c.shopping_checkpoint_selection = {"enable": True, "interval": 1, "patience": 2, "minimum_updates": 0}
    c.data.val_files = [str(ROOT / "data/grpo/validation.parquet")]
    c.data.val_batch_size = 1
    c.data.validation_shuffle = False
    c.trainer.val_before_train = True
    c.actor_rollout_ref.rollout.update(enforce_eager=True, enable_prefix_caching=False,
        val_kwargs={"n": 1, "do_sample": False, "temperature": 0.0, "top_p": 1.0})
    c.actor_rollout_ref.rollout.agent.num_workers = 2
    trainer.async_rollout_manager.agent_loop_workers *= 2
    original_workers = trainer.async_rollout_manager.agent_loop_workers
    monkeypatch.setenv("SHOPPING_GRPO_ROOT", str(ROOT))
    monkeypatch.setenv("GRPO_OUTPUT_DIR", str(tmp_path / "output"))
    for key in ["SHOPPING_VALIDATION_RECEIPTS_DIR", "SHOPPING_DIAGNOSTIC_ROLLOUT_DIR", "SHOPPING_GIGPO_PROBE_DIR"]:
        monkeypatch.delenv(key, raising=False)
    import pyarrow.parquet as pq
    expected = [int(r["extra_info"]["task_id"]) for r in pq.read_table(c.data.val_files[0]).to_pylist()]

    def validate():
        assert len(trainer.async_rollout_manager.agent_loop_workers) == 1
        assert c.actor_rollout_ref.rollout.agent.num_workers == 1
        run.validations.append(trainer.global_steps)
        count = {0: 30, 1: 31, 2: 31, 3: 30}[trainer.global_steps]
        records = infos(count, trainer.global_steps)
        for record, task_id in zip(records, expected):
            record["task_id"] = task_id
        if bad_validation and trainer.global_steps == 1:
            records.pop()
        data = SimpleNamespace(non_tensor_batch={"shopping": records})
        class Batch:
            non_tensor_batch = data.non_tensor_batch
            def __len__(self): return len(records)
        persist_unpadded_validation_batch(Batch(), global_step=trainer.global_steps)
        return {"validation": 1.0}

    def pin(selector, step):
        artifact = selector.output / f"selected-{step}.safetensors"
        artifact.write_bytes(bytes([step]))
        return {"step": step, "adapter": str(artifact), "adapter_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}

    trainer._validate = validate
    monkeypatch.setattr(TrainingCheckpointSelection, "pin_best", pin)
    if bad_validation:
        with pytest.raises(ValueError, match="denominator"):
            trainer.fit()
        assert len(run.observed) == 1 and run.optimizer.state[run.parameter]["step"] == 1
        assert trainer._shopping_flash_pool.closed and trainer._validate == validate
        assert trainer.async_rollout_manager.agent_loop_workers is original_workers
        assert c.actor_rollout_ref.rollout.agent.num_workers == 2
        best = json.loads((tmp_path / "output/checkpoint_selection/best-checkpoint.json").read_text())
        assert best["step"] == 0
        return
    trainer.fit()
    assert len(run.observed) == 3 and run.optimizer.state[run.parameter]["step"] == 3
    assert run.scheduler.last_epoch == 3 and run.saves == [1, 2, 3]
    assert run.validations == [0, 1, 2, 3] and run.weights == [0, 1, 2, 3]
    assert trainer._shopping_flash_pool.closed and trainer._validate == validate
    assert trainer.async_rollout_manager.agent_loop_workers is original_workers
    assert c.actor_rollout_ref.rollout.agent.num_workers == 2
    state = json.loads((tmp_path / "output/checkpoint_selection/state.json").read_text())
    best = json.loads((tmp_path / "output/checkpoint_selection/best-checkpoint.json").read_text())
    assert state["should_stop"] and best["step"] == 1
    assert Path(best["adapter"]).read_bytes() == b"\x01"


def test_public_entry_selects_fixed_validation_and_protects_it(monkeypatch, parent, tmp_path):
    options = args(monkeypatch, parent, tmp_path, method="flashreinforce", profile="main")
    options.checkpoint_selection = True
    command, _ = entry.build_command(options)
    assert "shopping_checkpoint_selection.enable=true" in command
    assert "data.validation_shuffle=false" in command and "trainer.val_before_train=true" in command
    assert "actor_rollout_ref.rollout.enforce_eager=true" in command
    options.hydra_overrides = ["actor_rollout_ref.rollout.val_kwargs.temperature=0.7"]
    with pytest.raises(SystemExit, match="dedicated options"):
        entry.build_command(options)


def test_real_pinned_hydra_and_preflight_admit_only_frozen_validation50(monkeypatch, parent, tmp_path):
    import importlib.util
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    options = args(monkeypatch, parent, tmp_path, method="flashreinforce", profile="main")
    options.checkpoint_selection = True
    command, environment = entry.build_command(options)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    GlobalHydra.instance().clear()
    search = (ROOT / "outputs/preparation/v1/runtime-wheel/source/verl/trainer/config").as_posix()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        recipe = compose(config_name="grpo", overrides=command[5:] + [f'hydra.searchpath=["file://{search}"]'])
    spec = importlib.util.spec_from_file_location("checkpoint_preflight", ROOT / "scripts/check_grpo_runtime.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.validate_training_task_isolation(recipe)
    recipe.data.val_files = [str(ROOT / "outputs/preparation/flashreinforce-final200-v3/final200.parquet")]
    with pytest.raises(SystemExit, match="Final"):
        checker.validate_training_task_isolation(recipe)
