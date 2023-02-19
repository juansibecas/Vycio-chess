import numpy as np
import chess
import chess.pgn as pgn


def load_data():
    """
    Loads all data for the NN. Returns the input and output
    vectors of all moves made by the player in each game.
    """

    pgn_casual = open("Vycio Casual Rapid.pgn")

    pgn_rated = open("Vycio Rated Rapid.pgn")

    rated_games = []

    casual_games = []

    for i in range(500):  # high number, idk how many games are in there
        c_game = pgn.read_game(pgn_casual)
        r_game = pgn.read_game(pgn_rated)
        if c_game is not None:
            casual_games.append(c_game)
        if r_game is not None:
            rated_games.append(r_game)

    games = []
    games.extend(casual_games)
    games.extend(rated_games)

    data = []

    for game in games:
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

    for game in data:
        game['inputs'] = get_inputs_from_game_boards(game['boards'], game['color'])

        game['outputs'] = get_outputs_from_game_boards_and_moves(game['boards'], game['moves'], game['color'])

    return data


def get_outputs_from_game_boards_and_moves(boards, moves, color):
    """
    Gets the coded output vector and the piece chosen(not used yet)
    for the NN given a certain board and the move made by the player.
    """

    outputs = []
    j = 0
    for board, move in zip(boards, moves):
        if j % 2 == 0:  # Send even turns if color is white
            if color == 'White':
                outputs.append(vector_from_move(board, move))

        elif j % 2 == 1:  # Send odd turns if color is black
            if color == 'Black':  # Mirror the board when black
                from_square = move.from_square
                from_square_set = chess.SquareSet.from_square(from_square)
                from_square_mirrored = list(from_square_set.mirror())[0]

                to_square = move.to_square
                to_square_set = chess.SquareSet.from_square(to_square)
                to_square_mirrored = list(to_square_set.mirror())[0]

                mirrored_move = chess.Move(from_square_mirrored, to_square_mirrored)
                mirrored_board = board.mirror()

                outputs.append(vector_from_move(mirrored_board, mirrored_move))
        j += 1
    return outputs


def vector_from_move(board, move):
    """
    Codes the output vector(1x4096) given a move.
    4096 represents the 64 squares to move from by the 64 squares to move to.
    """

    from_square = move.from_square

    to_square = move.to_square

    output = from_square * 64 + to_square

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
                inputs.append(array_from_board(board, color))

        elif j % 2 == 1:  # Send odd turns if color is black
            if color == 'Black':
                inputs.append(array_from_board(board, color))

    return inputs


def array_from_board(board, color):
    """
    Collects a 8x8x6 array from a board. Each of the 6 channels
    responds to a single piece type, coding it as a 1 for
    an ally piece and a -1 for an enemy piece.
    """

    array = np.zeros((8, 8, 6))  # 6 8x8 planes
    ally_squares = []
    enemy_squares = []

    if color == 'White':
        ally = 1  # 1 is white in py-chess
        enemy = 0  # 0 is black in py-chess
    else:
        ally = 0
        enemy = 1

    for piece in range(1, 7):  # 1 to 6 for pawn, knight, bishop, rook, queen and king

        if color == 'White':
            ally_square_set = board.pieces(piece, ally)
            enemy_square_set = board.pieces(piece, enemy)
        else:
            ally_square_set = board.pieces(piece, ally).mirror()
            enemy_square_set = board.pieces(piece, enemy).mirror()

        for square in ally_square_set:
            ally_squares.append(square)

        for square in enemy_square_set:
            enemy_squares.append(square)

        for square in ally_squares:
            row, col = index_8x8_from_integer(square)
            array[row, col, piece-1] = 1

        for square in enemy_squares:
            row, col = index_8x8_from_integer(square)
            array[row, col, piece-1] = -1

        ally_squares.clear()
        enemy_squares.clear()
    return array


def index_8x8_from_integer(num):
    """
    Gets 8x8 row and column numbers from a 0-63 input
    """
    col = num % 8
    row = int((num - col)/8)
    return row, col
