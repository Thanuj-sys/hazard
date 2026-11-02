"""Evaluation tools; heavy training dependencies load only when requested."""
__all__ = ["MultiHazardEvaluator", "EvaluationRunner"]


def __getattr__(name):
    if name == "MultiHazardEvaluator":
        from .metrics import MultiHazardEvaluator
        return MultiHazardEvaluator
    if name == "EvaluationRunner":
        from .eval_runner import EvaluationRunner
        return EvaluationRunner
    raise AttributeError(name)
