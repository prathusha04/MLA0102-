"""
AI Search Algorithms — All 5 Game-Based Learning Activities (CO1)
==================================================================
1. AI Maze Escape        -> Breadth-First Search (BFS)
2. N-Queens (12x12)      -> Backtracking / Constraint Satisfaction
3. Water Jug Puzzle      -> BFS (State-Space Search)
4. Connect Four AI       -> Minimax with Alpha-Beta Pruning
5. 8-Puzzle              -> A* Search with Manhattan Distance Heuristic

Run this file and choose which game to run from the menu,
or choose "6" to run all five one after another.
"""

from collections import deque
from heapq import heappush, heappop
import math


# ======================================================================
# 1. AI MAZE ESCAPE — BFS Shortest Path
# ======================================================================
def run_maze_escape():
    print("\n" + "=" * 60)
    print("1. AI MAZE ESCAPE — BFS Shortest Path")
    print("=" * 60)

    maze = [
        [0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
    ]
    start = (0, 0)
    goal = (4, 4)

    def bfs_maze(maze, start, goal):
        rows, cols = len(maze), len(maze[0])
        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == goal:
                return path

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and maze[nr][nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return None

    path = bfs_maze(maze, start, goal)
    print("Shortest path found using BFS:")
    for step in path:
        print(step)
    print("Number of steps taken:", len(path) - 1)


# ======================================================================
# 2. N-QUEENS CHALLENGE (12x12) — Backtracking Search
# ======================================================================
def run_n_queens():
    print("\n" + "=" * 60)
    print("2. N-QUEENS CHALLENGE (12x12) — Backtracking Search")
    print("=" * 60)

    def solve_n_queens(n):
        board = [-1] * n

        def is_safe(row, col):
            for r in range(row):
                c = board[r]
                if c == col or abs(c - col) == abs(r - row):
                    return False
            return True

        def backtrack(row):
            if row == n:
                return board[:]
            for col in range(n):
                if is_safe(row, col):
                    board[row] = col
                    result = backtrack(row + 1)
                    if result:
                        return result
                    board[row] = -1
            return None

        return backtrack(0)

    N = 12
    solution = solve_n_queens(N)

    print(f"{N}-Queens solution (row -> column), 0-indexed:")
    for row, col in enumerate(solution):
        print(f"Row {row}: Column {col}")

    print("\nBoard visualization:")
    for row in range(N):
        line = ""
        for col in range(N):
            line += "Q " if solution[row] == col else ". "
        print(line)

    print("\nConflicts:", 0)


# ======================================================================
# 3. WATER JUG PUZZLE — BFS State-Space Search
# ======================================================================
def run_water_jug():
    print("\n" + "=" * 60)
    print("3. WATER JUG PUZZLE — BFS State-Space Search")
    print("=" * 60)

    def solve_water_jug(cap_a, cap_b, target):
        start = (0, 0)
        visited = {start}
        queue = deque([(start, [])])

        while queue:
            (a, b), path = queue.popleft()

            if a == target or b == target:
                return path + [(a, b)]

            moves = [
                (cap_a, b),
                (a, cap_b),
                (0, b),
                (a, 0),
                (a - min(a, cap_b - b), b + min(a, cap_b - b)),
                (a + min(b, cap_a - a), b - min(b, cap_a - a)),
            ]

            for state in moves:
                if state not in visited:
                    visited.add(state)
                    queue.append((state, path + [(a, b)]))
        return None

    solution = solve_water_jug(11, 9, 8)
    print("Number of moves:", len(solution) - 1)
    for i, state in enumerate(solution):
        print(f"Step {i}: Jug A = {state[0]}L, Jug B = {state[1]}L")


# ======================================================================
# 4. CONNECT FOUR AI — Minimax with Alpha-Beta Pruning
# ======================================================================
def run_connect_four():
    print("\n" + "=" * 60)
    print("4. CONNECT FOUR AI — Minimax with Alpha-Beta Pruning")
    print("=" * 60)

    ROWS, COLS = 6, 7
    EMPTY, PLAYER, AI = 0, 1, 2

    def create_board():
        return [[EMPTY] * COLS for _ in range(ROWS)]

    def print_board(board):
        for row in board:
            print(" ".join({EMPTY: '.', PLAYER: 'X', AI: 'O'}[v] for v in row))
        print()

    def valid_moves(board):
        return [c for c in range(COLS) if board[0][c] == EMPTY]

    def drop_piece(board, col, piece):
        for r in range(ROWS - 1, -1, -1):
            if board[r][col] == EMPTY:
                board[r][col] = piece
                return r
        return -1

    def check_win(board, piece):
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(board[r][c + i] == piece for i in range(4)):
                    return True
        for c in range(COLS):
            for r in range(ROWS - 3):
                if all(board[r + i][c] == piece for i in range(4)):
                    return True
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(board[r + i][c + i] == piece for i in range(4)):
                    return True
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if all(board[r - i][c + i] == piece for i in range(4)):
                    return True
        return False

    def score_position(board, piece):
        center_col = [board[r][COLS // 2] for r in range(ROWS)]
        return center_col.count(piece) * 3

    def minimax(board, depth, alpha, beta, maximizing):
        valid = valid_moves(board)
        if check_win(board, AI):
            return (None, 1000000)
        if check_win(board, PLAYER):
            return (None, -1000000)
        if depth == 0 or not valid:
            return (None, score_position(board, AI))

        if maximizing:
            value = -math.inf
            best_col = valid[0]
            for col in valid:
                temp = [row[:] for row in board]
                drop_piece(temp, col, AI)
                _, new_score = minimax(temp, depth - 1, alpha, beta, False)
                if new_score > value:
                    value, best_col = new_score, col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best_col, value
        else:
            value = math.inf
            best_col = valid[0]
            for col in valid:
                temp = [row[:] for row in board]
                drop_piece(temp, col, PLAYER)
                _, new_score = minimax(temp, depth - 1, alpha, beta, True)
                if new_score < value:
                    value, best_col = new_score, col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return best_col, value

    board = create_board()
    scripted_player_moves = [3, 2, 4, 1]
    move_num = 0

    print("Initial board:")
    print_board(board)

    while move_num < len(scripted_player_moves):
        p_col = scripted_player_moves[move_num]
        drop_piece(board, p_col, PLAYER)
        print(f"Player drops in column {p_col}")
        print_board(board)
        if check_win(board, PLAYER):
            print("Player wins!")
            break

        ai_col, _ = minimax(board, 4, -math.inf, math.inf, True)
        drop_piece(board, ai_col, AI)
        print(f"AI (Minimax, depth=4) drops in column {ai_col}")
        print_board(board)
        if check_win(board, AI):
            print("AI wins!")
            break

        move_num += 1

    print("Simulation complete.")


# ======================================================================
# 5. 8-PUZZLE — A* Search with Manhattan Distance Heuristic
# ======================================================================
def run_eight_puzzle():
    print("\n" + "=" * 60)
    print("5. 8-PUZZLE — A* Search with Manhattan Distance Heuristic")
    print("=" * 60)

    GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def manhattan(state):
        dist = 0
        for i, tile in enumerate(state):
            if tile == 0:
                continue
            goal_i = GOAL.index(tile)
            r1, c1 = divmod(i, 3)
            r2, c2 = divmod(goal_i, 3)
            dist += abs(r1 - r2) + abs(c1 - c2)
        return dist

    def neighbors(state):
        idx = state.index(0)
        r, c = divmod(idx, 3)
        moves = []
        for dr, dc, name in [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                nidx = nr * 3 + nc
                new_state = list(state)
                new_state[idx], new_state[nidx] = new_state[nidx], new_state[idx]
                moves.append((tuple(new_state), name))
        return moves

    def a_star(start):
        open_set = [(manhattan(start), 0, start, [])]
        visited = {start: 0}

        while open_set:
            f, g, state, path = heappop(open_set)
            if state == GOAL:
                return path + [state]

            for nstate, move in neighbors(state):
                ng = g + 1
                if nstate not in visited or ng < visited[nstate]:
                    visited[nstate] = ng
                    nf = ng + manhattan(nstate)
                    heappush(open_set, (nf, ng, nstate, path + [state]))
        return None

    start_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)
    print("Initial scrambled state:")
    for i in range(0, 9, 3):
        print(start_state[i:i + 3])

    solution = a_star(start_state)
    print("\nSolved using A* search with Manhattan distance heuristic.")
    print("Number of moves to solve:", len(solution) - 1)
    print("\nStep-by-step states:")
    for i, s in enumerate(solution):
        print(f"Step {i}:")
        for r in range(0, 9, 3):
            print(s[r:r + 3])
        print()


# ======================================================================
# MENU
# ======================================================================
def main():
    games = {
        "1": run_maze_escape,
        "2": run_n_queens,
        "3": run_water_jug,
        "4": run_connect_four,
        "5": run_eight_puzzle,
    }

    print("Select a game to run:")
    print("1. AI Maze Escape (BFS)")
    print("2. N-Queens 12x12 (Backtracking)")
    print("3. Water Jug Puzzle (BFS)")
    print("4. Connect Four AI (Minimax)")
    print("5. 8-Puzzle (A*)")
    print("6. Run ALL five")

    choice = input("Enter choice (1-6): ").strip()

    if choice == "6":
        for func in games.values():
            func()
    elif choice in games:
        games[choice]()
    else:
        print("Invalid choice. Running all five by default.\n")
        for func in games.values():
            func()


if __name__ == "__main__":
    main()
