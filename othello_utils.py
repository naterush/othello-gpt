"""
Utility functions for mech int experiments.
"""
import numpy as np
import torch

from data import OthelloBoardState
from matplotlib.pyplot import imshow

from neel_plotly import imshow


STARTING_SQUARES = [27, 28, 35, 36]
ALPHA = "ABCDEFGH"
COLUMNS = [str(_) for _ in range(1, 9)]


def build_itos():

    """
    Build itos mapping.
    Handles 27, 28, 35, 36 squares (starting squares).
    """
    itos = {0: -100}
    for idx in range(1, 28):
        itos[idx] = idx - 1

    for idx in range(28, 34):
        itos[idx] = idx + 1

    for idx in range(34, 61):
        itos[idx] = idx + 3
    return itos


def build_stoi():
    """
    Build stoi mapping.
    Handles 27, 28, 35, 36 squares (starting squares).
    """
    _itos = build_itos()
    stoi = {y: x for x, y in _itos.items()}
    stoi[-1] = 0
    for sq in STARTING_SQUARES:
        assert sq not in stoi
    return stoi


ITOS = build_itos()
stoi = build_stoi()


def to_string(x):
    """
    Confusingly, maps x (board cell)to an int, but a board position
    label not a token label.
    (token labels have 0 == pass, and middle board cells don't exist)
    """
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_string(x.item())

    if isinstance(x, (list, np.ndarray, torch.Tensor)):
        print(x)
        return [to_string(i) for i in x]

    if isinstance(x, int):
        return itos[x]

    if isinstance(x, str):
        x = x.upper()
        return 8 * ALPHA.index(x[0]) + int(x[1])

    raise RuntimeError(f"Unknown type for x: {type(x)}.")


def to_int(x):
    """
    Convert x (board cell) to 'int' representation (model's vocabulary).
    Calls itself recursively.
    """
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_int(x.item())

    if isinstance(x, (list, np.ndarray, torch.Tensor)):
        return [to_int(i) for i in x]

    if isinstance(x, int):
        return stoi[x]

    if isinstance(x, str):
        x = x.upper()
        return to_int(to_string(x))

    raise RuntimeError(f"Unknown type for x: {type(x)}.")


def state_stack_to_one_hot_threeway(state_stack):
    one_hot = torch.zeros(
        1,  # even vs. odd vs. all (mode)
        state_stack.shape[0],
        state_stack.shape[1],
        8,  # rows
        8,  # cols
        3,  # the 2 options
        device=state_stack.device,
        dtype=torch.int,
    )
    one_hot[..., 0] = state_stack == 0
    one_hot[0, :, 0::2, ..., 1] = (state_stack == 1)[:, 0::2]
    one_hot[0, :, 1::2, ..., 1] = (state_stack == -1)[:, 1::2]

    one_hot[0, :, 0::2, ..., 2] = (state_stack == -1)[:, 0::2]
    one_hot[0, :, 1::2, ..., 2] = (state_stack == 1)[:, 1::2]

    return one_hot

def state_stack_to_one_hot_threeway_black_white(state_stack):
    """
    Channel 0: empty
    Channel 1: black
    Channel 2: white
    """
    one_hot = torch.zeros(
        1,  
        state_stack.shape[0],  
        state_stack.shape[1],  
        8,  
        8,  
        3,  
        device=state_stack.device,
        dtype=torch.int,
    )
    
    one_hot[..., 0] = (state_stack == 0)  # empty cells
    one_hot[..., 1] = (state_stack == 1)  # black pieces
    one_hot[..., 2] = (state_stack == -1)  # white pieces
    
    return one_hot


def seq_to_state_stack(str_moves):
    if isinstance(str_moves, torch.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move in str_moves:
        board.umpire(move)
        states.append(np.copy(board.state))
    states = np.stack(states, axis=0)
    return states


def make_plot_state(board):
    state = np.copy(board.state).flatten()
    valid_moves = board.get_valid_moves()
    next_move = board.get_next_hand_color()
    # print(next_move, valid_moves)
    for move in valid_moves:
        state[move] = next_move - 0.5
    return state


def add_counter(fig, position, color):
    is_black = color > 0
    row = position // 8
    col = position % 8
    fig.layout.shapes += (
        dict(
            type="circle",
            x0=col - 0.2,
            y0=row - 0.2,
            x1=col + 0.2,
            y1=row + 0.2,
            fillcolor="black" if is_black else "white",
            line_color="green",
            line_width=0.5,
        ),
    )
    return fig


def plot_board(moves, return_fig=False):
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()
    if isinstance(moves[0], str):
        moves = to_string(moves)
    board = OthelloBoardState()
    states = []
    states.append(make_plot_state(board))
    for move in moves:
        board.umpire(move)
        states.append(make_plot_state(board))
    states = np.stack(states, axis=0)
    fig = imshow(
        states.reshape(-1, 8, 8),
        color_continuous_scale="Geyser",
        aspect="equal",
        return_fig=True,
        animation_frame=0,
        y=["a", "b", "c", "d", "e", "f", "g", "h"],
        x=["0", "1", "2", "3", "4", "5", "6", "7"],
        animation_index=[
            f"{i+1} ({'W' if i%2==0 else 'B'}) [{to_board_label(moves[i]) if i>=0 else 'X'} -> {to_board_label(moves[i+1]) if i<len(moves)-1 else 'X'}]"
            for i in range(-1, len(moves))
        ],
        animation_name="Move",
    )
    fig = fig.update_layout(title_x=0.5)
    fig.update_traces(
        text=[[str(i + 8 * j) for i in range(8)] for j in range(8)],
        texttemplate="%{text}",
    )
    for c, frame in enumerate(fig.frames):
        for i in range(64):
            if states[c].flatten()[i] == 1:
                frame = add_counter(frame, i, True)
            elif states[c].flatten()[i] == -1:
                frame = add_counter(frame, i, False)
    fig.layout.shapes = fig.frames[0].layout.shapes
    if return_fig:
        return fig
    else:
        fig.show()


def moves_to_state(moves):
    # moves is a list of string entries (ints)
    state = np.zeros((8, 8), dtype=bool)
    for move in moves:
        state[move // 8, move % 8] = 1.0
    return state


def counter_shape(position, color, mode="normal"):
    is_black = color > 0
    row = position // 8
    col = position % 8
    shape = dict(
        type="circle",
        fillcolor="black" if is_black else "white",
    )
    if mode == "normal":
        shape.update(
            x0=col - 0.2,
            y0=row - 0.2,
            x1=col + 0.2,
            y1=row + 0.2,
            line_color="green",
            line_width=0.5,
        )
    elif mode == "flipped":
        shape.update(
            x0=col - 0.22,
            y0=row - 0.22,
            x1=col + 0.22,
            y1=row + 0.22,
            line_color="purple",
            line_width=3,
        )
    elif mode == "new":
        shape.update(
            line_color="red",
            line_width=4,
            x0=col - 0.25,
            y0=row - 0.25,
            x1=col + 0.25,
            y1=row + 0.25,
        )
    return shape


int_labels = (
    list(range(1, 28))
    + ["X", "X"]
    + list(range(28, 34))
    + ["X", "X"]
    + list(range(34, 61))
)

def to_label(x, from_int=True):
    # print("\t", x)
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_label(x.item(), from_int=from_int)
    elif (
        isinstance(x, list) or isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)
    ):
        return [to_label(i, from_int=from_int) for i in x]
    elif isinstance(x, int):
        if from_int:
            return to_board_label(to_string(x))
        else:
            return to_board_label(x)
    elif isinstance(x, str):
        return x


def plot_single_board(moves, model=None, return_fig=False, title=None):
    # moves is a list of string entries (ints)
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()
    if isinstance(moves[0], str):
        moves = to_string(moves)
    board = OthelloBoardState()
    if len(moves) > 1:
        board.update(moves[:-1])

    prev_state = np.copy(board.state)
    prev_player = board.next_hand_color
    prev_valid_moves = board.get_valid_moves()
    board.umpire(moves[-1])
    next_state = np.copy(board.state)
    next_player = board.next_hand_color
    next_valid_moves = board.get_valid_moves()

    empty = (prev_state == 0) & (next_state == 0)
    new = (prev_state == 0) & (next_state != 0)
    flipped = (prev_state != 0) & (next_state != prev_state) & (~new)
    prev_valid = moves_to_state(prev_valid_moves)
    next_valid = moves_to_state(next_valid_moves)

    state = np.copy(next_state)
    state[flipped] *= 0.9
    state[prev_valid] = 0.1 * prev_player
    state[next_valid] = 0.5 * next_player
    state[new] = 0.9 * prev_player
    if model is not None:
        logits = model(torch.tensor(to_int(moves)).cuda().unsqueeze(0)).cpu()
        log_probs = logits.log_softmax(-1)
        lps = torch.zeros(64) - 15.0
        stoi_indices = list(stoi.keys())
        lps[stoi_indices] = log_probs[0, -1, 1:]

    if title is None:
        title = f"{'Black' if prev_player!=1 else 'White'} To Play. Board State After {'Black' if prev_player==1 else 'White'} Plays {to_label(moves[-1], from_int=False)} "

    alpha = "ABCDEFGH"
    fig = imshow(
        state,
        color_continuous_scale="Geyser",
        title=title,
        y=[i for i in alpha],
        x=[str(i) for i in range(8)],
        aspect="equal",
        return_fig=True,
    )
    fig = fig.update_layout(title_x=0.5)
    fig.data[0]["hovertemplate"] = "<b>%{y}%{x}</b><br>%{customdata}<extra></extra>"

    shapes = []
    texts = []
    for i in range(64):
        texts.append("")
        if empty.flatten()[i]:
            texts[-1] = to_label(i, from_int=False)
        elif flipped.flatten()[i]:
            shapes.append(counter_shape(i, prev_player == 1, mode="flipped"))
        elif new.flatten()[i]:
            shapes.append(counter_shape(i, prev_player == 1, mode="new"))
        elif prev_state.flatten()[i] != 0:
            shapes.append(counter_shape(i, prev_state.flatten()[i] == 1, mode="normal"))
        else:
            raise ValueError(i)
    fig.layout.shapes = tuple(shapes)
    fig.data[0]["text"] = np.array(texts).reshape(8, 8)
    fig.data[0]["texttemplate"] = "%{text}"
    if model is not None:
        fig.data[0]["customdata"] = np.array(
            [f"LP:{lps[i].item():.4f}<br>I:{int_labels[i]}<br>S:{i}" for i in range(64)]
        ).reshape(8, 8)
    else:
        fig.data[0]["customdata"] = np.array(
            [f"I:{int_labels[i]}<br>S:{i}" for i in range(64)]
        ).reshape(8, 8)

    if return_fig:
        return fig
    else:
        fig.show()
    return


def build_state_stack(board_seqs_string):
    """
    Construct stack of board-states.
    """
    state_stack = []
    for idx, seq in enumerate(board_seqs_string):
        if idx % 1000 == 0:
            print(f"Processing {idx}th sequence.")
        _stack = seq_to_state_stack(seq)
        state_stack.append(_stack)
    return torch.tensor(np.stack(state_stack))


def to_board_label(idx):
    return f"{ALPHA[idx//8]}{COLUMNS[idx%8]}"


def run_with_cache_and_hooks(
    model,
    fwd_hooks,
    *model_args,
    **model_kwargs,
):
    """
    Runs the model and returns model output and a Cache object.
    Applies all hooks in fwd_hooks.
    """
    cache_dict = model.add_caching_hooks()
    for name, hook in fwd_hooks:
        if type(name) == str:
            model.mod_dict[name].add_hook(hook, dir="fwd")
        else:
            # Otherwise, name is a Boolean function on names
            for hook_name, hp in model.hook_dict.items():
                if name(hook_name):
                    hp.add_hook(hook, dir="fwd")

    model_out = model(*model_args, **model_kwargs)

    model.reset_hooks(False, including_permanent=False)
    return model_out, cache_dict