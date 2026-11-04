from baseline.trainers.train_stgs import train_stgs
from baseline.trainers.train_reinforce import train_reinforce
from baseline.trainers.train_quantize import train_quantize
from baseline.trainers.train_qtable import train_qtable

TRAINERS = {
    "stgs": train_stgs,
    "reinforce": train_reinforce,
    "quantize": train_quantize,
    "qtable": train_qtable,
}

__all__ = ["TRAINERS", "train_stgs", "train_reinforce", "train_quantize", "train_qtable"]
