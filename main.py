import cv2
import numpy as np
import pyautogui
import time
import random  # <-- CRITICAL FIX: Import the 'random' module
from PIL import ImageGrab
from typing import Tuple, List, Set, Optional


# ======================================================================
#                        CONFIGURATION
# ======================================================================
class GoConfig:
    """Centralized configuration for the Go Bot."""
    # BOARD_REGION = (31, 353, 836, 1157)
    BOARD_REGION = (30, 272, 631, 878)
    BOARD_TOP_LEFT_SCREEN = (73, 316)
    # BOARD_TOP_LEFT_SCREEN = (89, 409)
    BOARD_WIDTH = 603
    BOARD_HEIGHT = 603
    BOARD_SIZE = 19
    MY_PLAYER_ID = 1
    OPPONENT_ID = 2
    LOOP_DELAY = 2.0
    MAX_CONSECUTIVE_ERRORS = 5

    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        assert cls.BOARD_SIZE in [9, 13, 19], "Board size must be 9, 13, or 19"
        assert cls.MY_PLAYER_ID in [1, 2], "Player ID must be 1 or 2"
        assert cls.OPPONENT_ID == 3 - cls.MY_PLAYER_ID, "Opponent ID is incorrect"


# ======================================================================
#                      VISION & UTILITY FUNCTIONS
# ======================================================================
class BoardCapture:
    """Handles screen capture and coordinate calculations."""

    @staticmethod
    def capture_board() -> np.ndarray:
        """Captures the board region from the screen."""
        try:
            img = ImageGrab.grab(bbox=GoConfig.BOARD_REGION)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except RuntimeError as e:  # Catch a more specific error
            raise RuntimeError(f"Failed to capture screen: {e}")

    @staticmethod
    def get_cell_center_screen(row: int, col: int) -> Tuple[int, int]:
        """Calculates the absolute screen pixel coordinate for a cell's center."""
        cell_w = GoConfig.BOARD_WIDTH / (GoConfig.BOARD_SIZE - 1)
        cell_h = GoConfig.BOARD_HEIGHT / (GoConfig.BOARD_SIZE - 1)
        x = GoConfig.BOARD_TOP_LEFT_SCREEN[0] + col * cell_w
        y = GoConfig.BOARD_TOP_LEFT_SCREEN[1] + row * cell_h
        return int(x), int(y)

    @staticmethod
    def extract_cell(board_img: np.ndarray, row: int, col: int, size: int = 25) -> Optional[np.ndarray]:
        """Extracts a square image of a single intersection from the board image."""
        start_x = GoConfig.BOARD_TOP_LEFT_SCREEN[0] - GoConfig.BOARD_REGION[0]
        start_y = GoConfig.BOARD_TOP_LEFT_SCREEN[1] - GoConfig.BOARD_REGION[1]
        spacing_x = GoConfig.BOARD_WIDTH / (GoConfig.BOARD_SIZE - 1)
        spacing_y = GoConfig.BOARD_HEIGHT / (GoConfig.BOARD_SIZE - 1)
        center_x = start_x + col * spacing_x
        center_y = start_y + row * spacing_y

        half = size // 2
        left = int(center_x - half)
        top = int(center_y - half)
        right = left + size
        bottom = top + size

        if not (0 <= left and 0 <= top and right <= board_img.shape[1] and bottom <= board_img.shape[0]):
            return None

        return board_img[top:bottom, left:right]


# ======================================================================
#                       VISION ENGINE (CORRECTED)
# ======================================================================
class VisionEngine:
    """
    Determines the state of each intersection with robust marker detection.
    """

    @staticmethod
    def identify_stone(cell: np.ndarray) -> Tuple[int, bool]:
        """
        Identifies stone and last-move marker with improved logic.
        Checks for a board-colored "hole" in the center of a stone.
        """
        if cell is None:
            return 0, False

        # --- Color Analysis Thresholds (Grayscale: 0=Black, 255=White) ---
        BLACK_STONE_THRESHOLD = 90
        WHITE_STONE_THRESHOLD = 180

        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        average_color = np.mean(cell_gray)

        # 1. Determine if a stone is present based on the average color
        player_id = 0
        if average_color < BLACK_STONE_THRESHOLD:
            player_id = 1  # Black stone
        elif average_color > WHITE_STONE_THRESHOLD:
            player_id = 2  # White stone

        # 2. If a stone is present, check for the marker
        is_current = False
        if player_id != 0:
            h, w = cell_gray.shape

            # Define a small 5x5 pixel region in the very center of the intersection image
            center_y_slice = slice(h // 2 - 2, h // 2 + 3)
            center_x_slice = slice(w // 2 - 2, w // 2 + 3)
            center_patch = cell_gray[center_y_slice, center_x_slice]

            if center_patch.size > 0:
                # Get the average color of just the center patch
                center_color = np.mean(center_patch)

                # A marked stone has a center that is the color of the board.
                # Check if the center's color falls between our stone thresholds.
                if BLACK_STONE_THRESHOLD < center_color < WHITE_STONE_THRESHOLD:
                    is_current = True

        return player_id, is_current


# ======================================================================
#                       GAME STATE & LOGIC
# ======================================================================
class GoGameState:
    """Represents the state of the Go board and provides game logic."""

    def __init__(self, board_size: int = 19, played_moves: Optional[Set[Tuple[int, int]]] = None):
        self.size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.played_moves = played_moves if played_moves is not None else set()

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Gets valid neighbor coordinates for a given cell."""
        neighbors = []
        if row > 0: neighbors.append((row - 1, col))
        if row < self.size - 1: neighbors.append((row + 1, col))
        if col > 0: neighbors.append((row, col - 1))
        if col < self.size - 1: neighbors.append((row, col + 1))
        return neighbors

    def find_group(self, row: int, col: int) -> Tuple[Set[Tuple[int, int]], int]:
        """Finds the connected group of stones and their liberties."""
        stone_color = self.board[row, col]
        if stone_color == 0:
            return set(), 0

        group, liberties = set(), set()
        q, visited = [(row, col)], {(row, col)}

        while q:
            r, c = q.pop(0)
            group.add((r, c))
            for nr, nc in self.get_neighbors(r, c):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    neighbor_stone = self.board[nr, nc]
                    if neighbor_stone == 0:
                        liberties.add((nr, nc))
                    elif neighbor_stone == stone_color:
                        q.append((nr, nc))
        return group, len(liberties)

    def is_legal_move(self, row: int, col: int, player: int) -> bool:
        """Checks if a move is legal (not occupied, not suicide)."""
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.board[row, col] != 0 or (row, col) in self.played_moves:
            return False

        self.board[row, col] = player
        captured_any = any(self.find_group(nr, nc)[1] == 0 for nr, nc in self.get_neighbors(row, col) if
                           self.board[nr, nc] == 3 - player)
        _, liberties = self.find_group(row, col)
        self.board[row, col] = 0

        return liberties > 0 or captured_any


# ======================================================================
#                       AI ENGINE (CORRECTED)
# ======================================================================
class AiEngine:
    """
    Finds the best move using a corrected alpha-beta search algorithm.
    """

    def __init__(self):
        self.nodes_searched = 0

    def evaluate_board(self, game_state: GoGameState, player: int) -> float:
        """
        Evaluates the board from the perspective of the given player.
        """
        my_score, opponent_score = 0, 0
        opponent = 3 - player
        my_liberties, opponent_liberties = 0, 0
        my_stones, opponent_stones = 0, 0
        my_influence, opponent_influence = 0, 0
        center_zone = range(4, 15)
        visited_groups = set()

        for r in range(game_state.size):
            for c in range(game_state.size):
                stone_color = game_state.board[r, c]
                if stone_color != 0 and (r, c) not in visited_groups:
                    group, liberties = game_state.find_group(r, c)
                    visited_groups.update(group)

                    if stone_color == player:
                        my_stones += len(group)
                        my_liberties += liberties
                        if liberties == 1: my_score -= 25 * len(group)
                        my_influence += sum(1 for gr, gc in group if gr in center_zone and gc in center_zone)
                    else:
                        opponent_stones += len(group)
                        opponent_liberties += liberties
                        if liberties == 1: opponent_score -= 25 * len(group)
                        opponent_influence += sum(1 for gr, gc in group if gr in center_zone and gc in center_zone)

        return (
                (my_liberties - opponent_liberties) * 2.0 +
                (my_stones - opponent_stones) * 10.0 +
                (my_influence - opponent_influence) * 5.0 +
                my_score - opponent_score
        )

    def find_best_move(self, game_state: GoGameState, player: int, depth: int = 2) -> Optional[Tuple[int, int]]:
        """
        Uses the minimax algorithm with alpha-beta pruning to find the best move.
        """
        self.nodes_searched = 0
        start_time = time.time()

        best_move, best_score = self._alpha_beta_search(game_state, depth, -float('inf'), float('inf'), True, player)

        print(f" AI Search: {self.nodes_searched} nodes in {time.time() - start_time:.2f}s. Score: {best_score:.2f}")
        return best_move

    def _alpha_beta_search(self, game_state: GoGameState, depth: int, alpha: float, beta: float,
                           maximizing_player: bool, current_player: int) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Core alpha-beta search function with corrected simulation logic.
        """
        if depth == 0:
            self.nodes_searched += 1
            return None, self.evaluate_board(game_state, GoConfig.MY_PLAYER_ID)

        legal_moves = [
            (r, c) for r in range(game_state.size) for c in range(game_state.size)
            if game_state.is_legal_move(r, c, current_player)
        ]

        if not legal_moves:
            self.nodes_searched += 1
            return None, self.evaluate_board(game_state, GoConfig.MY_PLAYER_ID)

        best_move = random.choice(legal_moves)

        if maximizing_player:
            max_eval = -float('inf')
            for move in legal_moves:
                r, c = move

                # --- START OF FIX ---
                # Create a true deep copy of the state for simulation
                simulated_state = GoGameState(game_state.size, game_state.played_moves.copy())
                simulated_state.board = np.copy(game_state.board)

                # Simulate the move on the copy
                simulated_state.board[r, c] = current_player
                simulated_state.played_moves.add((r, c))  # CRITICAL: Update played moves in the copy

                _, eval_score = self._alpha_beta_search(simulated_state, depth - 1, alpha, beta, False,
                                                        3 - current_player)
                # --- END OF FIX ---

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return best_move, max_eval
        else:  # Minimizing player
            min_eval = float('inf')
            for move in legal_moves:
                r, c = move

                # --- START OF FIX ---
                # Create a true deep copy of the state for simulation
                simulated_state = GoGameState(game_state.size, game_state.played_moves.copy())
                simulated_state.board = np.copy(game_state.board)

                # Simulate the move on the copy
                simulated_state.board[r, c] = current_player
                simulated_state.played_moves.add((r, c))  # CRITICAL: Update played moves in the copy

                _, eval_score = self._alpha_beta_search(simulated_state, depth - 1, alpha, beta, True,
                                                        3 - current_player)
                # --- END OF FIX ---

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha: break
            return best_move, min_eval

    def find_best_move(self, game_state: GoGameState, player: int, depth: int = 2) -> Optional[Tuple[int, int]]:
        """
        Uses the minimax algorithm with alpha-beta pruning to find the best move.
        """
        self.nodes_searched = 0
        start_time = time.time()

        best_move, best_score = self._alpha_beta_search(
            game_state, depth, -float('inf'), float('inf'), True, player
        )

        print(
            f" AI Search Complete: Searched {self.nodes_searched} nodes in {time.time() - start_time:.2f}s. Best score: {best_score:.2f}")
        return best_move

    def _alpha_beta_search(self, game_state: GoGameState, depth: int, alpha: float, beta: float,
                           maximizing_player: bool, current_player: int) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Core alpha-beta search function.
        """
        if depth == 0:
            self.nodes_searched += 1
            return None, self.evaluate_board(game_state, GoConfig.MY_PLAYER_ID)

        legal_moves = []
        for r in range(game_state.size):
            for c in range(game_state.size):
                if game_state.is_legal_move(r, c, current_player):
                    legal_moves.append((r, c))

        if not legal_moves:
            self.nodes_searched += 1
            return None, self.evaluate_board(game_state, GoConfig.MY_PLAYER_ID)

        best_move = random.choice(legal_moves)  # Start with a random valid move

        if maximizing_player:
            max_eval = -float('inf')
            for move in legal_moves:
                r, c = move

                # Create a copy of the board to simulate the move
                temp_board_state = np.copy(game_state.board)
                game_state.board[r, c] = current_player

                _, eval_score = self._alpha_beta_search(game_state, depth - 1, alpha, beta, False, 3 - current_player)

                # Undo the move
                game_state.board = temp_board_state

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Prune
            return best_move, max_eval
        else:  # Minimizing player
            min_eval = float('inf')
            for move in legal_moves:
                r, c = move

                # Create a copy of the board to simulate the move
                temp_board_state = np.copy(game_state.board)
                game_state.board[r, c] = current_player

                _, eval_score = self._alpha_beta_search(game_state, depth - 1, alpha, beta, True, 3 - current_player)

                # Undo the move
                game_state.board = temp_board_state

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Prune
            return best_move, min_eval


# ======================================================================
#                       MAIN BOT CONTROLLER (CORRECTED)
# ======================================================================
class GoBot:
    """The main bot class with a robust, stateful control loop."""

    def __init__(self):
        GoConfig.validate()
        self.ai = AiEngine()
        self.consecutive_errors = 0
        self.played_moves = set()
        print(" Go Bot initialized!")
        print(f"Configuration: Playing as {'Black' if GoConfig.MY_PLAYER_ID == 1 else 'White'}")

    def analyze_board(self) -> GoGameState:
        """Analyzes the screen to build the current game state."""
        board_img = BoardCapture.capture_board()
        game_state = GoGameState(GoConfig.BOARD_SIZE, self.played_moves.copy())  # Use a copy of played moves
        last_move_player, black_count, white_count = None, 0, 0

        for r in range(GoConfig.BOARD_SIZE):
            for c in range(GoConfig.BOARD_SIZE):
                cell_img = BoardCapture.extract_cell(board_img, r, c)
                player, is_current = VisionEngine.identify_stone(cell_img)
                game_state.board[r, c] = player
                if player == 1:
                    black_count += 1
                elif player == 2:
                    white_count += 1
                if is_current: last_move_player = player

        if last_move_player is not None:
            game_state.current_player = 3 - last_move_player
        else:
            game_state.current_player = 1 if (black_count + white_count) % 2 == 0 else 2

        return game_state

    def run_single_turn(self) -> bool:
        """
        Executes a single turn for the bot.
        Returns True if a move was made, False otherwise.
        """
        game_state = self.analyze_board()
        black = np.count_nonzero(game_state.board == 1)
        white = np.count_nonzero(game_state.board == 2)
        print(f"\nBoard: B:{black}, W:{white}. Turn: {'B' if game_state.current_player == 1 else 'W'}")

        if game_state.current_player != GoConfig.MY_PLAYER_ID:
            return False  # Not our turn, so no move was made.

        print(" Our turn!")
        best_move = self.ai.find_best_move(game_state, GoConfig.MY_PLAYER_ID)

        if best_move is None:
            print(" No legal moves found. Passing.")
            return False

        row, col = best_move
        letter = chr(65 + col if col < 8 else 66 + col)
        number = GoConfig.BOARD_SIZE - row
        go_coord = f"{letter}{number}"
        click_x, click_y = BoardCapture.get_cell_center_screen(row, col)

        print(f"⚡ Move: {go_coord} ({row},{col}). Clicking ({click_x},{click_y}).")
        pyautogui.click(x=click_x, y=click_y, duration=0.1)
        self.played_moves.add((row, col))
        print(f" {go_coord} added to memory.")
        self.consecutive_errors = 0
        return True  # A move was successfully made

    def start(self):
        """Starts the main continuous game loop with intelligent waiting."""
        print("=" * 50 + "\n Starting continuous game loop...\n Press Ctrl+C to stop.\n" + "=" * 50)
        try:
            while True:
                # This is the main play loop
                move_was_made = self.run_single_turn()

                # --- THE DEFINITIVE FIX ---
                if move_was_made:
                    # If we just moved, we MUST wait for the opponent to respond.
                    # We do this by waiting for the number of opponent stones to change.
                    print(" Move made. Now waiting for opponent's response...")

                    # Give the GUI a moment to draw our stone before we check the stone count.
                    time.sleep(1.0)
                    state_after_my_move = self.analyze_board()
                    opponent_stone_count = np.count_nonzero(state_after_my_move.board == GoConfig.OPPONENT_ID)

                    wait_start_time = time.time()
                    while time.time() - wait_start_time < 60:  # Max wait 60 seconds
                        time.sleep(GoConfig.LOOP_DELAY)

                        current_state = self.analyze_board()
                        new_opponent_count = np.count_nonzero(current_state.board == GoConfig.OPPONENT_ID)

                        if new_opponent_count > opponent_stone_count:
                            print(" Opponent move detected. Proceeding to our turn.")
                            break  # Success! Exit the inner wait loop.
                        else:
                            print(f"... still waiting for opponent. (Current count: {new_opponent_count})")
                    else:  # This 'else' belongs to the 'while' loop, for timeout
                        print(" Timed out waiting for opponent. Re-analyzing.")
                else:
                    # If it wasn't our turn, just wait normally before checking again.
                    print(" Waiting for opponent...")
                    time.sleep(GoConfig.LOOP_DELAY)
                # --- END OF THE DEFINITIVE FIX ---

        except KeyboardInterrupt:
            print("\n Bot stopped by user.")
        except Exception as fatal_e:
            print(f"\n Fatal error forced bot to stop: {fatal_e}")
            import traceback
            traceback.print_exc()

    def run_single_turn(self) -> bool:
        """
        Executes a single turn for the bot.
        Returns True if a move was made, False otherwise.
        """
        game_state = self.analyze_board()
        black = np.count_nonzero(game_state.board == 1)
        white = np.count_nonzero(game_state.board == 2)
        print(f"\nBoard: B:{black}, W:{white}. Turn: {'B' if game_state.current_player == 1 else 'W'}")

        if game_state.current_player != GoConfig.MY_PLAYER_ID:
            return False  # It's not our turn, so no move was made

        print(" Our turn!")
        best_move = self.ai.find_best_move(game_state, GoConfig.MY_PLAYER_ID)

        if best_move is None:
            print(" No legal moves found. Passing.")
            return False  # No move was made

        row, col = best_move
        letter = chr(65 + col if col < 8 else 66 + col)
        number = GoConfig.BOARD_SIZE - row
        go_coord = f"{letter}{number}"
        click_x, click_y = BoardCapture.get_cell_center_screen(row, col)

        print(f"⚡ Move: {go_coord} ({row},{col}). Clicking ({click_x},{click_y}).")
        pyautogui.click(x=click_x, y=click_y, duration=0.1)
        self.played_moves.add((row, col))
        print(f" {go_coord} added to memory.")
        self.consecutive_errors = 0
        return True  # A move was successfully made

    def start(self):
        """Starts the main continuous game loop with intelligent waiting."""
        print("=" * 50 + "\n Starting continuous game loop...\n Press Ctrl+C to stop.\n" + "=" * 50)
        try:
            while True:
                # This is the main play loop
                move_was_made = self.run_single_turn()

                # --- START OF THE CRITICAL FIX ---
                if move_was_made:
                    # If we just moved, we MUST wait for the opponent to respond.
                    # We do this by waiting for the number of opponent stones to increase.
                    print(" Move made. Now waiting for opponent's response...")

                    # Get the board state right after our move
                    state_after_my_move = self.analyze_board()
                    opponent_stone_count = np.count_nonzero(state_after_my_move.board == GoConfig.OPPONENT_ID)

                    wait_start_time = time.time()
                    while time.time() - wait_start_time < 60:  # Max wait 60 seconds
                        time.sleep(GoConfig.LOOP_DELAY)  # Check every few seconds

                        current_state = self.analyze_board()
                        new_opponent_count = np.count_nonzero(current_state.board == GoConfig.OPPONENT_ID)

                        if new_opponent_count > opponent_stone_count:
                            print(" Opponent move detected. Proceeding to our turn.")
                            break  # Break the inner wait loop and start our turn analysis
                        else:
                            print("... still waiting for opponent.")
                    else:  # This 'else' belongs to the 'while' loop
                        print(" Timed out waiting for opponent. Re-analyzing.")
                else:
                    # If it wasn't our turn, just wait normally before checking again.
                    print(" Waiting for opponent...")
                    time.sleep(GoConfig.LOOP_DELAY)
                # --- END OF THE CRITICAL FIX ---

        except KeyboardInterrupt:
            print("\n Bot stopped by user.")
        except Exception as fatal_e:
            print(f"\n Fatal error forced bot to stop: {fatal_e}")
            import traceback
            traceback.print_exc()

    def run_single_turn(self):
        """Executes a single turn for the bot."""
        game_state = self.analyze_board()
        black = np.count_nonzero(game_state.board == 1)
        white = np.count_nonzero(game_state.board == 2)
        print(f"\nBoard: B:{black}, W:{white}. Turn: {'B' if game_state.current_player == 1 else 'W'}")

        if game_state.current_player != GoConfig.MY_PLAYER_ID:
            print(" Waiting for opponent...")
            return

        print(" Our turn!")
        best_move = self.ai.find_best_move(game_state, GoConfig.MY_PLAYER_ID)

        if best_move is None:
            print(" No legal moves found. Passing.")
            return

        row, col = best_move
        letter = chr(65 + col if col < 8 else 66 + col)
        number = GoConfig.BOARD_SIZE - row
        go_coord = f"{letter}{number}"
        click_x, click_y = BoardCapture.get_cell_center_screen(row, col)

        print(f"⚡ Move: {go_coord} ({row},{col}). Clicking ({click_x},{click_y}).")
        pyautogui.click(x=click_x, y=click_y, duration=0.1)
        self.played_moves.add((row, col))
        print(f" {go_coord} added to memory.")
        self.consecutive_errors = 0

    def start(self):
        """Starts the main continuous game loop."""
        print("=" * 50 + "\n Starting continuous game loop...\n Press Ctrl+C to stop.\n" + "=" * 50)
        try:
            while True:
                try:
                    self.run_single_turn()
                except Exception as e:  # Catching exceptions per-turn allows recovery
                    self.consecutive_errors += 1
                    print(f" Error in turn #{self.consecutive_errors}: {e}")
                    if self.consecutive_errors >= GoConfig.MAX_CONSECUTIVE_ERRORS:
                        print(" Too many consecutive errors. Stopping.")
                        raise  # Re-raise the exception to stop the bot
                time.sleep(GoConfig.LOOP_DELAY)
        except KeyboardInterrupt:
            print("\n Bot stopped by user.")
        except Exception as fatal_e:  # Catches the re-raised exception
            print(f"\n Fatal error forced bot to stop: {fatal_e}")
            import traceback
            traceback.print_exc()


# ======================================================================
#                              ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    print("Welcome to the Go Bot!")
    try:
        bot = GoBot()
        bot.start()
    except Exception as init_e:
        print(f"Failed to initialize bot: {init_e}")