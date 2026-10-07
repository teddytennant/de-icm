"""Small equity table for a four-player field."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from de_icm import de_icm

stacks = np.array([10.0, 20.0, 30.0, 40.0])
prizes = np.array([0.5, 0.3, 0.2, 0.0])
Q, V = de_icm(stacks, prizes)
np.set_printoptions(precision=6, suppress=True)
print("stacks", stacks)
print("Q")
print(Q)
print("equity", V)
