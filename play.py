import chess
import numpy as np
from load import array_from_board


def play(model):  # plays as white only for now TODO play as black too
    """
    Used to play against a previously trained model.
    """

    board = chess.Board()
    array = np.zeros((1, 8, 8, 6))
    while True:
        array[0][:][:][:] = array_from_board(board, "White")
        output = model.predict(array)

        while True:
            prediction = output.argmax()
            to_square = int(prediction % 64)  # Decodes the move
            from_square = int((prediction - to_square)/64)
            ai_uci = chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]
            ai_move = chess.Move.from_uci(ai_uci)
            if ai_move in board.legal_moves:  # Checks if move is legal
                break
            else:
                output[0][prediction] = 0  # Masks 0 for that illegal move and goes to the next one

        print("AI moves ", ai_uci)
        board.push(ai_move)

        player_uci = input("Write your move. Ex: a1h8")
        if player_uci == "exit":
            print("game ended by user")
            break
        player_move = chess.Move.from_uci(player_uci)
        board.push(player_move)
