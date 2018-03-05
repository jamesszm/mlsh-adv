"""
A classic four room problem described in intro to RL book chap 9.
Code adapted from frozen lake environment.
"""

import logging
import sys

import numpy as np
from gym import utils
from gym.envs.toy_text import discrete

logger = logging.getLogger(__name__)

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "9x9": [
        "XXGXXXXXX",
        "XOOOXOOOX",
        "XOOOOOOOX",
        "XOOOXOOOX",
        "XXOXXXOXX",
        "XOOOXOOOX",
        "XOOOOOOOX",
        "XOOOXOOOX",
        "XXXXXXXXX",
    ]
}


class Fourrooms(discrete.DiscreteEnv):
    """
    The agent must go through the doors to exit.
    An example of a 9X9 world would be:

    XXGXXXXXX
    XOOOXOOOX
    XOOOOOOOX
    XOOOXOOOX
    XXOXXXOXX
    XOOOXOOOX
    XOOOOOOOX
    XOOOXOOOX
    XXXXXXXXX

    X : Walls
    G : goal
    O : Normal floor, any normal ground can be a starting point.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, desc=None, map_name="9x9"):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'O').astype('float64').ravel()
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            orig_row = row
            orig_col = col
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # down
                row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                row = max(row - 1, 0)
            is_wall = desc[row][col] == b'X'
            if is_wall:
                return (orig_row, orig_col)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'G':
                        li.append((1.0, s, 0, True))
                    else:
                        # TODO(yejiayu): Add stochastic case.
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        done = bytes(newletter) in b'GH'
                        rew = float(newletter == b'G') * 100 - 1
                        li.append((1.0, newstate, rew, done))

        super(Fourrooms, self).__init__(nS, nA, P, isd)

    def reset(self, seed=None):
        np.random.seed(seed)
        return super(Fourrooms, self).reset()

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right",
                                             "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile
