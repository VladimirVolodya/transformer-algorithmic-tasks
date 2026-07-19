from src.tasks.addition import AdditionTask
from src.tasks.base import Task
from src.tasks.dyck import DyckTask
from src.tasks.index_task import IndexTask
from src.tasks.sort import SortTask
from src.tasks.vocab import TaskTokenizer

TASKS = {
    "addition": AdditionTask,
    "sort": SortTask,
    "dyck": DyckTask,
    "index": IndexTask,
}

__all__ = [
    "Task",
    "TaskTokenizer",
    "TASKS",
    "AdditionTask",
    "SortTask",
    "DyckTask",
    "IndexTask",
]
