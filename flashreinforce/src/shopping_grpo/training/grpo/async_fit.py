"""Completion-order collection behind the pinned native training-loop interface."""
from dataclasses import dataclass
from functools import wraps
from time import monotonic
from uuid import uuid4

from shopping_grpo.training.grpo.async_sampling import CompletionPool, Submitted


@dataclass
class CompletedBatch:
    training: object
    output: object
    metrics: dict


def completion_batches(trainer, epoch):
    import numpy as np
    import ray
    from verl import DataProto

    options = trainer.config.shopping_flashreinforce
    manager = trainer.async_rollout_manager
    workers = manager.agent_loop_workers
    counter = 0
    feedback = None
    if trainer.config.get("shopping_feedback_sampling", {}).get("enable", False):
        from shopping_grpo.training.grpo.feedback_sampling import FeedbackSource
        feedback = FeedbackSource(trainer)
        trainer._shopping_feedback_source = feedback

    def inputs():
        for batch_dict in trainer.train_dataloader:
            training = DataProto.from_single_dict(batch_dict)
            training.meta_info["temperature"] = 1.0
            training.non_tensor_batch["uid"] = np.array([str(uuid4()) for _ in range(len(training))], dtype=object)
            generation = trainer._get_gen_batch(training)
            for i in range(len(training)):
                yield training.select_idxs([i]), generation.select_idxs([i])

    def submit(item):
        nonlocal counter
        training, generation = item[:2]
        generation.meta_info = dict(generation.meta_info, global_steps=trainer.checkpoint_manager.sampling_version)
        ref = workers[counter % len(workers)].generate_sequences.remote(generation)
        counter += 1
        future = ref.future()
        if feedback is not None:
            future.add_done_callback(lambda done: feedback.worker_done(item[2], done))
        return Submitted(future, lambda: ray.cancel(ref, force=False))

    batch_size = int(trainer.config.data.train_batch_size)
    pool = CompletionPool(feedback if feedback is not None else inputs(), submit, inflight=int(options.async_inflight),
                          capacity=batch_size * int(options.async_queue_batches),
                          timeout=float(options.async_timeout_seconds))
    trainer._shopping_flash_pool = pool
    try:
        while True:
            completed, outputs, worker_metrics = [], [], []
            wait_start = monotonic()
            for _ in range(batch_size):
                try:
                    item, output, _ = next(pool)
                except StopIteration:
                    if completed:
                        raise RuntimeError("async epoch ended with an incomplete candidate batch")
                    return
                completed.append(item[0])
                worker_metrics.append(output.meta_info.pop("metrics"))
                outputs.append(output)
                if feedback is not None:
                    choice = item[2]
                    if int(output.non_tensor_batch["shopping"][0]["task_id"]) != choice["task_id"]:
                        raise ValueError("feedback completion returned the wrong task")
                    output.non_tensor_batch["shopping"][0]["feedback_sampling"] = dict(choice)
                    feedback.record("consumed", choice)
            training, output = DataProto.concat(completed), DataProto.concat(outputs)
            output.meta_info["timing"] = manager._performance_metrics(worker_metrics, output)
            metrics = {**pool.metrics(), "flash_async/completion_wait_seconds": monotonic() - wait_start,
                       "flash_async/source_epoch": epoch}
            try:
                yield CompletedBatch(training, output, metrics)
            finally:
                if feedback is not None:
                    feedback.sampler.release([int(row["task_id"]) for row in output.non_tensor_batch["shopping"]])
    finally:
        pool.close()
        if feedback is not None:
            feedback.close()


def transform_fit(source):
    """The caller has already verified the full native V4 source and admission hook."""
    replacements = [
        ('        admission = WholeBatchAdmission.create(self.config, diagnostics_path)\n',
         '        admission = WholeBatchAdmission.create(self.config, diagnostics_path)\n'
         '        from shopping_grpo.training.grpo.async_fit import completion_batches\n'),
        ('            for batch_dict in self.train_dataloader:\n',
         '            for batch_dict in completion_batches(self, epoch):\n'),
        ('                metrics = {}\n', '                metrics = dict(batch_dict.metrics)\n'),
        ('                batch: DataProto = DataProto.from_single_dict(batch_dict)\n',
         '                batch = batch_dict.training\n'),
        ('                batch.non_tensor_batch["uid"] = np.array(\n'
         '                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object\n'
         '                )\n', ''),
        ('                gen_batch = self._get_gen_batch(batch)\n',
         '                gen_batch = batch.select_idxs(list(range(len(batch))))\n'
         '                gen_batch.meta_info = dict(batch.meta_info)\n'),
        ('                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)\n',
         '                        combined_gen_output = batch_dict.output\n'),
        ('                    batch = batch.union(gen_batch_output)\n',
         '                    batch = batch.union(gen_batch_output)\n'
         '                    batch.meta_info["flash_learner_version"] = current_step - 1\n'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError("async native fit insertion point changed")
        source = source.replace(before, after, 1)
    return source


def install_worker_setup(trainer_class):
    original = trainer_class.init_workers
    if getattr(original, "_shopping_flash_async", False):
        return

    @wraps(original)
    def init_workers(self, *args, **kwargs):
        if self.config.get("shopping_flashreinforce", {}).get("execution", "sync") == "async":
            from shopping_grpo.training.grpo.async_runtime import activate_runtime
            activate_runtime()
        return original(self, *args, **kwargs)

    init_workers._shopping_flash_async = True
    trainer_class.init_workers = init_workers
