import numpy as np
import chess
import chess.pgn as pgn
import tensorflow as tf


def load_data():
    """
    Loads all data for the NN. Returns the data structure that contains
    the input and output vectors for all moves made by the player in each game.
    """

    pgn_casual = open("Vycio Casual Rapid.pgn")

    pgn_rated = open("Vycio Rated Rapid.pgn")

    rated_games = []

    casual_games = []

    for i in range(500):  # High number(has to be bigger than the amount of games in each pgn file)
        c_game = pgn.read_game(pgn_casual)
        r_game = pgn.read_game(pgn_rated)
        if c_game is not None:
            casual_games.append(c_game)
        if r_game is not None:
            rated_games.append(r_game)
        if c_game is None and r_game is None:
            break

    games = []
    games.extend(casual_games)
    games.extend(rated_games)

    data = []

    for game in games:  # Gets color, boards, moves and url
        game_data = {'color': 0, 'boards': [], 'moves': game.mainline_moves(), 'url': game.headers.get('Site')}

        if game.headers.get('White') == 'Vycio':
            game_data['color'] = 'White'
        else:
            game_data['color'] = 'Black'

        board = game.board()

        for move in game.mainline_moves():  # First board goes with first move, etc. Last board doesn't matter
            game_data['boards'].append(board.copy())
            board.push(move)

        data.append(game_data)

    for game in data:  # Adds inputs and outputs for each game
        game['inputs'] = get_inputs_from_game_boards(game['boards'], game['color'])

        game['outputs'] = get_outputs_from_game_boards_and_moves(game['boards'], game['moves'], game['color'])

    return data


def get_outputs_from_game_boards_and_moves(boards, moves, color):
    """
    Gets the coded output vector and the piece chosen(not used yet)
    for the NN given a certain board and the move made by the player.
    """
    # TODO use black OR white to make 2 different models (insufficient data maybe)
    outputs = []
    j = 0
    for board, move in zip(boards, moves):  # Overloaded for loop. Could use enumerate() for j.
        if j % 2 == 0:  # Send only even turns if color is white
            if color == 'White':
                outputs.append(output_vector_from_move(board, move))

        elif j % 2 == 1:  # Send only odd turns if color is black
            if color == 'Black':  # Mirror the board when black (because the model only plays as white - possibly wrong)
                # Mirror the start square of the move
                from_square = move.from_square
                from_square_set = chess.SquareSet.from_square(from_square)
                from_square_mirrored = list(from_square_set.mirror())[0]

                # Mirror the end square of the move
                to_square = move.to_square
                to_square_set = chess.SquareSet.from_square(to_square)
                to_square_mirrored = list(to_square_set.mirror())[0]

                mirrored_move = chess.Move(from_square_mirrored, to_square_mirrored)
                mirrored_board = board.mirror()

                outputs.append(output_vector_from_move(mirrored_board, mirrored_move))
        j += 1
    return outputs


def output_vector_from_move(board, move):
    """
    Codes the output vector(1x4096) given a move.
    4096 represents the 64 squares to move from, multiplied by the 64 squares to move to.
    Square class from py-chess also works as an int 0-63.
    """

    from_square = move.from_square

    to_square = move.to_square

    output = from_square * 64 + to_square

    # Piece chosen could be used for a piece selector model (not used yet)
    piece_chosen = board.piece_type_at(move.from_square)

    return {'piece': piece_chosen, 'output_move': output}


def get_inputs_from_game_boards(boards, color):
    """
    Gets all inputs for the NN from a list of boards and
    the color the player was in each of those boards.
    """

    inputs = []

    for j, board in enumerate(boards):

        if j % 2 == 0:  # Send even turns if color is white
            if color == 'White':
                inputs.append(input_array_from_board(board, color))

        elif j % 2 == 1:  # Send odd turns if color is black
            if color == 'Black':
                inputs.append(input_array_from_board(board, color))

    return inputs


def input_array_from_board(board, color):
    """
    Returns a 8x8x6 array from a board. Each of the 6 channels
    responds to a single piece type, coding it as a 1 for
    an ally piece and a -1 for an enemy piece.
    TODO make 12 channels, 6 for ally and 6 for enemy
    """

    array = np.zeros((8, 8, 6))  # 6 8x8 planes

    if color == 'White':
        ally = 1  # 1 is white in py-chess
        enemy = 0  # 0 is black in py-chess
    else:
        ally = 0
        enemy = 1

    for piece in range(1, 7):  # 1 to 6 for pawn, knight, bishop, rook, queen and king in that order

        if color == 'White':
            ally_square_set = board.pieces(piece, ally)  # Stores the squares in which the player has a {piece}
            enemy_square_set = board.pieces(piece, enemy)
        else:
            ally_square_set = board.pieces(piece, ally).mirror()  # Mirror when playing black
            enemy_square_set = board.pieces(piece, enemy).mirror()

        # Fills the array with 1 and -1
        for ally_square, enemy_square in zip(ally_square_set, enemy_square_set):
            row, col = index_8x8_from_integer(ally_square)
            array[row, col, piece - 1] = 1
            row, col = index_8x8_from_integer(enemy_square)
            array[row, col, piece - 1] = -1

    return array


def index_8x8_from_integer(num):
    """
    Gets 8x8 row and column numbers from a 0-63 input
    """
    col = num % 8
    row = int((num - col)/8)
    return row, col


def get_labels(data):
    """
    Returns all labels from the data list.
    """

    labels = []
    for game in data:
        moves = game['outputs']
        for move in moves:
            labels.append(move['output_move'])

    labels = tf.keras.utils.to_categorical(labels, num_classes=4096)

    return labels


def get_inputs(data):
    """
    Returns all inputs from the data list.
    """

    inputs = []

    for game in data:
        boards = game['inputs']
        for board in boards:
            inputs.append(board)

    return inputs
