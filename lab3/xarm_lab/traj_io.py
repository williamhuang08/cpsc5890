import json
from dataclasses import dataclass
from typing import List


@dataclass
class JointTrajectory:
    t: List[float]
    q: List[List[float]]


def save_traj(path: str, traj: JointTrajectory):
    with open(path, "w") as f:
        json.dump({"t": traj.t, "q": traj.q}, f, indent=2)


def load_traj(path: str) -> JointTrajectory:
    with open(path, "r") as f:
        data = json.load(f)
    return JointTrajectory(t=data["t"], q=data["q"])