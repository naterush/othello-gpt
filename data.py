# Adapted from https://github.com/likenneth/othello_world/blob/master/data/othello.py

import os
import numpy as np
import random
from tqdm import tqdm
import time
import multiprocessing
import pickle
import psutil
import seaborn as sns
import itertools
from copy import copy, deepcopy
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mingpt.dataset import CharDataset

rows = list("abcdefgh")
columns = [str(_) for _ in range(1, 9)]

mask = np.zeros(64).reshape(8, 8)
mask[3, 3] = 1
mask[3, 4] = 1
mask[4, 3] = 1
mask[4, 4] = 1
mask = mask.astype(bool)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Othello is a strategy board game for two players (Black and White), played on an 8 by 8 board. 
# The game traditionally begins with four discs placed in the middle of the board as shown below. Black moves first.
# W (27) B (28)
# B (35) W (36)

def permit(s):
    s = s.lower()
    if len(s) != 2:
        return -1
    if s[0] not in rows or s[1] not in columns:
        return -1
    return rows.index(s[0]) * 8 + columns.index(s[1])

def permit_reverse(integer):
    r, c = integer // 8, integer % 8
    return "".join([rows[r], columns[c]])

start_hands = [permit(_) for _ in ["d5", "d4", "e4", "e5"]]
eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

import os
import pickle
import itertools
import multiprocessing
from tqdm import tqdm

SYNTHETIC_PATH = 'othello_synthetic'
BATCH_SIZE = 10000  # Adjust this value as needed

def create_synthetic_dataset(total=1000000):
    """Generate a synthetic Othello dataset and split into batches."""
    os.makedirs(SYNTHETIC_PATH, exist_ok=True)  # Ensure the directory exists
    sequences = []
    batch_index = 969

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        for can in tqdm(p.imap(get_ood_game, range(total)), total=total):
            if can not in sequences:
                sequences.append(can)

            # If the batch size is reached, save to a pickle file and reset
            if len(sequences) >= BATCH_SIZE:
                _save_batch(sequences, batch_index)
                sequences.clear()  # Clear memory. This dramatically speeds processing
                batch_index += 1

        # Save remaining sequences if any
        if sequences:
            _save_batch(sequences, batch_index)

def _save_batch(sequences, batch_index):
    """Save a batch of sequences to a pickle file."""
    filepath = os.path.join(SYNTHETIC_PATH, f'batch_{batch_index}.pkl')
    with open(filepath, 'wb') as handle:
        pickle.dump(sequences, handle)

def get_synthetic_dataset(total=None):
    """Load all pickle files from the synthetic dataset folder."""
    sequences = []
    pickle_files = os.listdir(SYNTHETIC_PATH)
    num_pickles = len(pickle_files) if total is None else total // BATCH_SIZE

    for i in tqdm(range(num_pickles)):
        filepath = os.path.join(SYNTHETIC_PATH, f'batch_{i}.pkl')
        with open(filepath, 'rb') as handle:
            sequences.extend(pickle.load(handle))
        
    # Deduplicate after loading all batches
    sequences.sort()
    sequences = [k for k, _ in itertools.groupby(sequences)]
    return sequences

def get_ood_game(_):
    tbr = []
    ab = OthelloBoardState()
    possible_next_steps = ab.get_valid_moves(take_first=True)
    while possible_next_steps:
        next_step = random.choice(possible_next_steps)
        tbr.append(next_step)
        ab.update([next_step, ])
        possible_next_steps = ab.get_valid_moves(take_first=True)
    del ab
    return tbr

def lazy_random_range(n):
    available = list(range(n))
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        available[i], available[j] = available[j], available[i]
        yield available[i]
    yield available[0]
    
class OthelloBoardState():
    # 1 is black, -1 is white
    def __init__(self, board_size = 8):
        self.board_size = board_size * board_size
        board = np.zeros((8, 8))
        board[3, 4] = 1
        board[3, 3] = -1
        board[4, 3] = 1
        board[4, 4] = -1
        self.initial_state = board
        self.state = self.initial_state
        self.age = np.zeros((8, 8))
        self.next_hand_color = 1
        self.history = []

    def get_occupied(self, ):
        board = self.state
        tbr = board.flatten() != 0
        return tbr.tolist()
    def get_state(self, ):
        board = self.state + 1  # white 0, blank 1, black 2
        tbr = board.flatten()
        return tbr.tolist()
    def get_age(self, ):
        return self.age.flatten().tolist()
    def get_next_hand_color(self, ):
        return (self.next_hand_color + 1) // 2
    
    def update(self, moves, prt=False):
        # takes a new move or new moves and update state
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            if prt:
                self.__print__()

    def umpire(self, move):
        r, c = move // 8, move % 8
        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"
        occupied = np.sum(self.state != 0)
        color = self.next_hand_color
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) == 0:  # means one hand is forfeited
            # print(f"One {color} move forfeited")
            color *= -1
            self.next_hand_color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
        if len(tbf) == 0:
            valids = self.get_valid_moves()
            if len(valids) == 0:
                assert 0, "Both color cannot put piece, game should have ended!"
            else:
                assert 0, "Illegal move!"
                
        self.age += 1
        for ff in tbf:
            self.state[ff[0], ff[1]] *= -1
            self.age[ff[0], ff[1]] = 0
        self.state[r, c] = color
        self.age[r, c] = 0
        self.next_hand_color *= -1
        self.history.append(move)
        
    def __print__(self, ):
        print("-"*20)
        print([permit_reverse(_) for _ in self.history])
        a = "abcdefgh"
        for k, row in enumerate(self.state.tolist()):
            tbp = []
            for ele in row:
                if ele == -1:
                    tbp.append("O")
                elif ele == 0:
                    tbp.append(" ")
                else:
                    tbp.append("X")
            # tbp.append("\n")
            print(" ".join([a[k]] + tbp))
        tbp = [str(k) for k in range(1, 9)]
        print(" ".join([" "] + tbp))
        print("-"*20)
        
    def plot_hm(self, ax, heatmap, pdmove, logit=False):
        padding = np.array([0., 0.])
        trs = {-1: r'O', 0: " ", 1: r'X'}
        if len(heatmap) == 60:
            heatmap = [heatmap[:27], padding, heatmap[27:33], padding, heatmap[33:]]
            heatmap = np.concatenate(heatmap)
        assert len(heatmap) == 64
        heatmap = np.array(heatmap).reshape(8, 8)
        annot = [trs[_] for _ in self.state.flatten().tolist()]
        cloned = deepcopy(self)
        cloned.update([pdmove, ])

        next_color = 1 - cloned.get_next_hand_color()
        annot[pdmove] = ("\\underline{" + (trs[next_color * 2 -1]) + "}")[-13:]

        color = {-1:'white', 0:'grey', 1:'black'}
        ann_col = [color[_] for _ in self.state.flatten().tolist()]
        # ann_col[pdmove] = color[next_color * 2 -1]
        text_for_next_color = color[next_color * 2 -1].capitalize()

        del cloned
        if logit:
            max_logit = np.max(np.abs(heatmap))
            sns.heatmap(data=heatmap, cbar=False, xticklabels=list(range(1,9)), 
                        # cmap=LinearSegmentedColormap.from_list("custom_cmap",  ["#D3D3D3", "#3349F2"]),
                        cmap=sns.color_palette("vlag", as_cmap=True), 
                        yticklabels=list("ABCDEFGH"), ax=ax, fmt="", square=True, linewidths=.5, vmin=-max_logit, vmax=max_logit, center=0)
        else:
            sns.heatmap(data=heatmap, cbar=False, xticklabels=list(range(1,9)),
                        # cmap=LinearSegmentedColormap.from_list("custom_cmap",  ["#D3D3D3", "#B90E0A"]),
                        cmap=sns.color_palette("vlag", as_cmap=True), 
                        yticklabels=list("ABCDEFGH"), ax=ax, fmt="", square=True, linewidths=.5, vmin=-1, vmax=1, center=0)
        ax.set_title(f"Prediction: {text_for_next_color} at " + permit_reverse(pdmove).upper())
        ax.add_patch(Rectangle((pdmove%8, pdmove//8), 1, 1, fill=False, edgecolor='black', lw=2))

        patchList = []
        for loca, col in enumerate(ann_col):
            if col != 'grey':
                patchList.append(PatchCollection([mpatches.Circle((loca%8 + 0.5, loca//8 + 0.5) ,.25, facecolor=col)], match_original=True))
        for i in patchList:
            ax.add_collection(i)
        return ax
        
    def tentative_move(self, move):
        # tentatively put a piece, do nothing to state
        # returns 0 if this is not a move at all: occupied or both player have to forfeit
        # return 1 if regular move
        # return 2 if forfeit happens but the opponent can drop piece at this place
        r, c = move // 8, move % 8
        if not self.state[r, c] == 0:
            return 0
        occupied = np.sum(self.state != 0)
        color = self.next_hand_color
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) != 0:
            return 1
        else:  # means one hand is forfeited
            # print(f"One {color} move forfeited")
            color *= -1
            # self.next_hand_color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
            if len(tbf) == 0:
                return 0
            else:
                return 2
        
    def get_valid_moves(self, take_first=False):
        regular_moves = []
        forfeit_moves = []

        for move in lazy_random_range(64):
            x = self.tentative_move(move)
            if x == 1:
                regular_moves.append(move)
            elif x == 2:
                forfeit_moves.append(move)
            else:
                pass

            if take_first and x == 1:
                return [move, ]

        if len(regular_moves):
            return regular_moves
        elif len(forfeit_moves):
            return forfeit_moves
        else:
            return []
 
    def get_gt(self, moves, func, prt=False):
        # takes a new move or new moves and update state
        container = []
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            container.append(getattr(self, func)())  
            # to predict first y, we need already know the first x
            if prt:
                self.__print__()
        return container

def get_test_and_train_datasets(total=10000, train_ratio=0.8):
    sequences = get_synthetic_dataset(total=total)
    train_size = int(total * train_ratio)
    
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]

    train_dataset = CharDataset(train_sequences)
    test_dataset = CharDataset(test_sequences)

    return train_dataset, test_dataset

if __name__ == "__main__":
    pass