from tasks.base import BaseTask
from tasks.causal_discovery import CausalDiscoveryTask
from tasks.shortest_path import ShortestPathTask

TASKS = {
    "causal_discovery": CausalDiscoveryTask,
    "shortest_path": ShortestPathTask,
}


def get_task(name: str) -> BaseTask:
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
    return TASKS[name]()
