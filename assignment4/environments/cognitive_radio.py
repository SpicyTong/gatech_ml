import sys

import numpy as np
from gym import utils
from gym.envs.toy_text import discrete
from six import StringIO

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "5x20": [
        "BBOOOAOOCCCCCCCCOOOY",
        "XBOOAAOOCCOOOOOOODDY",
        "OBBBAAOOCCOOAOADDDDY",
        "OOOBAAOOCCOAAAAADDDY",
        "OOOOBOOOOOOOOOOODDDY",
    ],
    "8x30": [
        "BBOOOAOOCCCCCCCCOOOCCOCCOCCOCY",
        "OBBOAAOOCCOOJJOOOAOOOAOOAOAOAY",
        "OOBBAAOOCCOOJJOOOAOOOAOOAOAOAY",
        "OOOBAAOOCCOOJJOOOAOOOAOOAOAOAY",
        "XOOCBBOOOOOOBJOOOBBBOOODDODOOY",
        "OOOCOBBBBOOOBJOOOBBOOOODODODDY",
        "OOOCOOBBBOOOOJBOBBOOOOODDODOOY",
        "OOOCOOOBOOOOOJBBOOOOOOODODODDY"
    ],
    "10x40": [
        "OOOOODOOODOOODOOODOOODOOODOOODOOODOOODOY",
        "OOOOODOOODOOODOOODOOODOOODOOODOOODOOODOY",
        "OOOOODOOODOOODOOODOOODOOODOOODOOODOOODOY",
        "OJJJJJOOOOOOOJJJJJJJJOOOOOOOOJJJJJOOOOOY",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAY",
        "OOAAAOOOAOOOAOOOAOOOAOOOAOOAOOOAOOOAOOOY",
        "OBBOBBOBBOOOOBBBBOOOOOOOOOOOOOBBBOBBOOOY",
        "XOOOOOOBBBBOOBBOOOOOOOCCCOOOOOCCCOOOCCCY",
        "OCCCOOOOOOCCCCCCCOOOCCOOOCCCOOOOOCCCOOOY",
        "OOOCCOOCCCOOOOOOOCCCOOOOOOOOOOOOOOOOOOOY"
]
}

class CognitiveRadio(discrete.DiscreteEnv):
    """
    This is an extremely simplified problem that illustrates a case in which cognitive radio could be used to allow
    an agent to communicate on a finite RF spectrum that has been divided into discrete channels (assuming no overlap and sufficient guard bands)

    The map is formatted such that each column corresponds to a time slot in the discrete world (pretend it's 802.11 or whatever) and
    each row corresponds to a discrete channel that the agent or any other transmitter in the scene could occupy.

    The agent (a transmit/receive pair) must learn a series of channels that it can use in order to provide gapless communication.

    The following restrictions are also placed on the agent:
    - The agent can tune to any channel that is open BUT:
        - Not needing to retune is a reward of (max num channels, so 5 in our example)
        - Each distance tuned from there removes 1 from the reward.
        # TODO maybe try to somehow favor gradual retuning? IE make this reward reduction exponential somehow?
    - The agent MUST transmit (gaps are provided such that this is always possible) or else gameover. (enforced via large negative reward)
    - The agent MUST move one and only one time slot forward in time (no time traveling)
        - There is no reward associated with this, it is simply a physical limitation.


    # MODES
    - There are two modes, a simple TDMA strucutre where we assume that we can always find a slot and that penalties for retuning are linear AND
        a more complex mode where the scene isnt so well behaves.


    The complex mode has the following rules:
    - We must transmit data for at least some percentage of the total time (say 75% of all time) or else we get a large negative reward at the end
    - If we tune to a channel that is adjacent to one that is filled, there is some probability that we won't succeed in transmitting on that slot
    - 

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="5x20", rewarding=True, step_reward=1, failure_reward=-100,
                 simple_tdma=False, collision_reward=-2, max_tune_dist=None):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, nrow) 
        self.step_reward = step_reward
        self.collision_reward = collision_reward
        if max_tune_dist is None:
            self.max_tune_dist = nrow
        else:
            self.max_tune_dist = max_tune_dist
        self.simple_tdma = simple_tdma
        self.failure_reward = failure_reward
        self.other_transmits = 'ABCDEFGHIJ'
        self.end_char = b'Y'
        self.start_char = b'X'
        self.jam_character = b'J'
        self.adjacent_collision_prob = .15
        self.out_of_band_tune_multiplier = 2
        self.tune_distance_reward_ratio = .5

        nA = nrow
        nS = nrow * ncol

        isd = np.array(desc == self.start_char).astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def tune(row, col, new_tune):
            """
            Incrementer. Always advances column.
            """
            row = new_tune
            return (row, col+1)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(self.nrow):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in self.end_char or letter in self.jam_character:
                        # Game over 
                        li.append((1.0, s, 0, True))
                    else:
                        # Tune and update reward
                        newrow, newcol = tune(row, col, a)
                        newstate = to_s(newrow, newcol)
                        new_letter = desc[newrow, newcol]
                        if self.simple_tdma:
                            rew, done = self.compute_reward_simple([newrow, newcol], [row, col], new_letter)
                        else:
                            rew, done = self.compute_reward_complex([newrow, newcol], [row, col], desc)
                        li.append((1.0, newstate, rew, done))

        super(CognitiveRadio, self).__init__(nS, nA, P, isd)

    
    def compute_reward_simple(self, new_pos, old_pos, new_letter):
        """
        Compute reward from old position to new position.

        Positions are in [row, col]
        """
        new_row = new_pos[0]
        old_row = old_pos[0]
        if new_letter in b'OY':
            rew = self.max_tune_dist - np.abs(new_row - old_row)
        else:
            rew = self.failure_reward

        done =  str(new_letter) in self.other_transmits or new_letter in self.end_char

        return rew, done

    def compute_reward_complex(self, new_pos, old_pos, spectrum):
        """
        Compute reward from old position to new position.

        Positions are in [row, col]
        """
        new_letter = spectrum[new_pos[0], new_pos[1]]
        new_row = new_pos[0]
        old_row = old_pos[0]
        total_reward = self.step_reward
        
        # If the tile is open or the end, compute an inverse linear reward with tune distance.
        if new_letter.astype(str) in "OY":
            # If the distance is greater than the tune dist (tuning outside of max dist)
            # Then we want to give a bigger negative reward that makes this expensive.
            potential_new_reward = (self.max_tune_dist - np.abs(new_row - old_row)) * self.tune_distance_reward_ratio
            if potential_new_reward < 0:
                potential_new_reward *= self.out_of_band_tune_multiplier
            # If the reward was open but has an occupied channel adjacent to it, compute prob. of collision.
            row_above = np.clip(new_pos[0] - 1, 0, self.nrow - 1)
            row_below = np.clip(new_pos[0] + 1, 0, self.nrow - 1)
            above = spectrum[row_above, new_pos[1]].astype(str)
            below = spectrum[row_below, new_pos[1]].astype(str)
            num_adjacent = 0
            if above in self.other_transmits:
                num_adjacent += 1
            if below in self.other_transmits:
                num_adjacent += 1
            if num_adjacent:
                p_collide = num_adjacent * self.adjacent_collision_prob
                # With some probability, perform a collision.
                collision = np.random.choice([True, False], 1,
                                                p=[p_collide, 1-p_collide])
                if collision:
                    potential_new_reward += self.collision_reward * num_adjacent

            total_reward += potential_new_reward

        elif new_letter == self.jam_character:
            total_reward += 0

        else:
            total_reward += self.collision_reward

        done = str(new_letter) == str(self.end_char) or new_letter == self.jam_character

        return total_reward, done

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile

    def colors(self):
        return {
            b'A': 'purple',
            b'B': 'skyblue',
            b'C': 'yellow',
            b'D': 'brown',
            b'E': 'orange',
            b'F': 'grey',
            b'J': 'red',
            b'O': 'white',
            b'X': 'green',
            b'Y': 'green',
        }

    def directions(self):
        return {
            -5: '5⬆',
            -4: '4⬆',
            -3: '3⬆',
            -2: '2⬆',
            -1: '1⬆',
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            10: '10'
        }

    def new_instance(self):
        return CognitiveRadio(desc=self.desc,step_reward=self.step_reward, simple_tdma=self.simple_tdma,
                                      failure_reward=self.failure_reward, collision_reward=self.collision_reward,
                                      max_tune_dist=self.max_tune_dist)
