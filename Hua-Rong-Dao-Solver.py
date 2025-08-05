from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
import sys

#====================================================================================

char_goal = '1'
char_single = '2'

class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_goal: bool, is_single: bool, coord_x: int, coord_y: int, orientation: str):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v')
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single, \
            self.coord_x, self.coord_y, self.orientation)

class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, pieces: list[Piece]):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()


    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.

        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def display(self):
        """
        Print out the current board.

        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces.
    State has a Board and some extra information that is relevant to the search:
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board: Board, f: int, depth: int, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.
        
    # Modified / Added - Modifying the comparison operator
    def __lt__(self, nxt):
        
        if self.f == nxt.f:
            return self.id < nxt.id
        else:
            return self.f < nxt.f
        
        

def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^': # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<': # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if g_found == False:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)

    return board


# ===========================================================================================



def is_Goal(state: State) -> bool:
    """
    Returns TRUE if the state is a goal state, FLASE otherwise.

    :param state: State
    :type state: State
    :return: True if the state is a goal state, false otherwise
    :rtype: boolean
    """

    # looping through all the pieces of the board to find the goal
    # piece and checking if it is in the goal state or not
    
    for x in state.board.pieces:
        if x.is_goal:
            if x.coord_y == 3 and x.coord_x == 1:
                return True
    
    return False


def heuristic_Val(state: State) -> int:
    """
    Takes a state and returns the state's heuristic value (or the h value).
    """
    
    x_cord = -1     # x-coordinate of the goal piece
    y_cord = -1     # y-coordinate of the goal piece
    x_heuristic = 0
    y_heuristic = 0
    
    for x in state.board.pieces:
        if x.is_goal:
            x_cord = x.coord_x
            y_cord = x.coord_y
            break
        
    if y_cord == 0:
        y_heuristic = 3
    elif y_cord == 1:
        y_heuristic = 2
    elif y_cord == 2:
        y_heuristic = 1
    elif y_cord == 3:
        y_heuristic = 0
    elif y_cord == 4:   # should never be in this case
        y_heuristic = 1
        
        
    if x_cord == 0:
        x_heuristic = 1
    elif x_cord == 1:
        x_heuristic = 0
    elif x_cord == 2:
        x_heuristic = 1
    elif x_cord == 3:   # should never be in this case
        x_heuristic = 2
        
    return x_heuristic + y_heuristic


def get_solution(state: State): # -> List[State]
    """
    Given a goal state, backtrack through the parent state references until the initial state.
    Return a sequence of states from the initial state to the goal state.
    """
    solution = []
    
    curr = state
    
    while curr.parent != None:
        solution.append(curr)
        curr = curr.parent
        
    solution.append(curr) # adding the initial state to the solution
    
    solution.reverse() # correcting the ordering of the solution
    
    return solution
        

def generate_dfs(state: State) -> State:
    """
    Runs dfs and gives us the goal state.
    Takes the initial state as input.
    Returns the goal state as output.
    - Make sure that the parent of the initial state recieved is None.
    - Make sure that all the initial values of the initial state are as it should be.
    """
    
    frontier = []
    explored = set()
    
    # Add initial state to frontier
    frontier.append(state)
    
    while len(frontier) > 0:
        
        curr_state = frontier.pop()
        
        if hash(str(curr_state.board.grid)) in explored:
            continue
        
        explored.add(hash(str(curr_state.board.grid)))
        
        if is_Goal(curr_state):
            return curr_state
        
        # add curr's successors to frontier
        for piece in curr_state.board.pieces:
            i_up, state_up = is_up(piece, curr_state)
            i_down, state_down = is_down(piece, curr_state)
            i_left, state_left = is_left(piece, curr_state)
            i_right, state_right = is_right(piece, curr_state)
            
            if i_up == 0:
                frontier.append(state_up)
            if i_down == 0:
                frontier.append(state_down)
            if i_left == 0:
                frontier.append(state_left)
            if i_right == 0:
                frontier.append(state_right)
        
    # code should never reach here    
    return state
    
    

        
def write_to_file(solution: list[State], filename: str):
    """
    Takes in a list of states and the required solution file name.
    Writes the solution to the solution file.
    """
    
    f = open(filename, "w")
    
    for state in solution:
        f.write(str(state.board.grid[0][0]) + str(state.board.grid[0][1]) + str(state.board.grid[0][2]) + str(state.board.grid[0][3]) + "\n")
        f.write(str(state.board.grid[1][0]) + str(state.board.grid[1][1]) + str(state.board.grid[1][2]) + str(state.board.grid[1][3]) + "\n")
        f.write(str(state.board.grid[2][0]) + str(state.board.grid[2][1]) + str(state.board.grid[2][2]) + str(state.board.grid[2][3]) + "\n")
        f.write(str(state.board.grid[3][0]) + str(state.board.grid[3][1]) + str(state.board.grid[3][2]) + str(state.board.grid[3][3]) + "\n")
        f.write(str(state.board.grid[4][0]) + str(state.board.grid[4][1]) + str(state.board.grid[4][2]) + str(state.board.grid[4][3]) + "\n")
        f.write("\n")
        
    f.close()
        
        
        
        
        
def generate_astar(state: State) -> State:
    """
    Runs A* and gives us the goal state.
    Takes the initial state as input.
    Returns the goal state as output.
    - Make sure that the parent of the initial state recieved is None.
    - Make sure that all the initial values of the initial state are as it should be.
    """
    
    frontier = []
    explored = set()
    
    # Add initial state to frontier
    frontier.append(state)
    
    while len(frontier) > 0:
        
        curr_state = heappop(frontier)
        
        if hash(str(curr_state.board.grid)) in explored:
            continue
        
        explored.add(hash(str(curr_state.board.grid)))
        
        if is_Goal(curr_state):
            return curr_state
        
        # add curr's successors to frontier
        for piece in curr_state.board.pieces:
            i_up, state_up = is_up(piece, curr_state)
            i_down, state_down = is_down(piece, curr_state)
            i_left, state_left = is_left(piece, curr_state)
            i_right, state_right = is_right(piece, curr_state)
            
            if i_up == 0:
                heappush(frontier, state_up)
            if i_down == 0:
                heappush(frontier, state_down)
            if i_left == 0:
                heappush(frontier, state_left)
            if i_right == 0:
                heappush(frontier, state_right)
        
    # code should never reach here    
    return state
        


def is_up(org_piece: Piece, org_state: State): # -> List[int, State]
    """
    Takes a piece and a State.
    Returns [-1, org_state] if the piece can't be moved up
    Returns [0, new_state] if the piece can be moved up
    """
    
    # Checking base (impossible) case
    if org_piece.coord_y == 0:
        return [-1, org_state]
    
    # making a copy
    piece = deepcopy(org_piece)
    state = org_state
    piece_lst = deepcopy(org_state.board.pieces)
    
    # removing the piece to be (possibly) modified from the list
    # piece_lst.remove(piece)
    for p in piece_lst:
        if p.is_goal == piece.is_goal and p.is_single == piece.is_single and p.coord_x == piece.coord_x and p.coord_y == piece.coord_y and p.orientation == piece.orientation:
            piece_lst.remove(p)
            break
    
    
    if piece.is_goal:
        if state.board.grid[piece.coord_y - 1][piece.coord_x] == '.' and state.board.grid[piece.coord_y - 1][piece.coord_x + 1] == '.':
            
            # updating the coordinates of the piece
            piece.coord_y = piece.coord_y - 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
    elif piece.is_single:
        if state.board.grid[piece.coord_y - 1][piece.coord_x] == '.':
            
            # updating the coordinates of the piece
            piece.coord_y = piece.coord_y - 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]  
    elif piece.orientation == 'h':
        if state.board.grid[piece.coord_y - 1][piece.coord_x] == '.' and state.board.grid[piece.coord_y - 1][piece.coord_x + 1] == '.':
            
            # updating the coordinates of the piece
            piece.coord_y = piece.coord_y - 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
    elif piece.orientation == 'v':
        if state.board.grid[piece.coord_y - 1][piece.coord_x] == '.':
            
            # updating the coordinates of the piece
            piece.coord_y = piece.coord_y - 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
            
    # If we have not returned till now, then the up-move is not possible
    
    return [-1, org_state]
    
    
    
    
    
    
def is_down(org_piece: Piece, org_state: State): # -> List[int, State]
    """
    Takes a piece and a State.
    Returns [-1, org_state] if the piece can't be moved down
    Returns [0, new_state] if the piece can be moved down
    """
    
    # Checking base (impossible) case
    if org_piece.is_goal or org_piece.orientation == 'v':
        if org_piece.coord_y == 3:
            return [-1, org_state]
    elif org_piece.is_single or org_piece.orientation == 'h':
        if org_piece.coord_y == 4:
            return [-1, org_state]
         
    
    # making a copy
    piece = deepcopy(org_piece)
    state = org_state
    piece_lst = deepcopy(state.board.pieces)
    
    # removing the piece to be (possibly) modified from the list
    # piece_lst.remove(piece)
    for p in piece_lst:
        if p.is_goal == piece.is_goal and p.is_single == piece.is_single and p.coord_x == piece.coord_x and p.coord_y == piece.coord_y and p.orientation == piece.orientation:
            piece_lst.remove(p)
            break
    
    
    if piece.is_goal:
        if state.board.grid[piece.coord_y + 2][piece.coord_x] == '.' and state.board.grid[piece.coord_y + 2][piece.coord_x + 1] == '.':
            
            # updating the coordinates of the piece
            piece.coord_y = piece.coord_y + 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
    elif piece.is_single:
        if state.board.grid[piece.coord_y + 1][piece.coord_x] == '.':
            
            # updating the coordinates of the piece
            piece.coord_y = piece.coord_y + 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
    elif piece.orientation == 'h':
        if state.board.grid[piece.coord_y + 1][piece.coord_x] == '.' and state.board.grid[piece.coord_y + 1][piece.coord_x + 1] == '.':
            
            # updating the coordinates of the piece
            piece.coord_y = piece.coord_y + 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
    elif piece.orientation == 'v':
        if state.board.grid[piece.coord_y + 2][piece.coord_x] == '.':
            
            # updating the coordinates of the piece
            piece.coord_y = piece.coord_y + 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
            
    # If we have not returned till now, then the up-move is not possible
    
    return [-1, org_state]
    
    
    
    
def is_left(org_piece: Piece, org_state: State): # -> List[int, State]
    """
    Takes a piece and a State.
    Returns [-1, org_state] if the piece can't be moved left
    Returns [0, new_state] if the piece can be moved left
    """
    
    # Checking base (impossible) case
    if org_piece.coord_x == 0:
        return [-1, org_state]

         
    
    # making a copy
    piece = deepcopy(org_piece)
    state = org_state
    piece_lst = deepcopy(state.board.pieces)
    
    # removing the piece to be (possibly) modified from the list
    # piece_lst.remove(piece)
    for p in piece_lst:
        if p.is_goal == piece.is_goal and p.is_single == piece.is_single and p.coord_x == piece.coord_x and p.coord_y == piece.coord_y and p.orientation == piece.orientation:
            piece_lst.remove(p)
            break
    
    
    if piece.is_goal:
        if state.board.grid[piece.coord_y][piece.coord_x - 1] == '.' and state.board.grid[piece.coord_y + 1][piece.coord_x - 1] == '.':
            
            # updating the coordinates of the piece
            piece.coord_x = piece.coord_x - 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
    elif piece.is_single:
        if state.board.grid[piece.coord_y][piece.coord_x - 1] == '.':
            
            # updating the coordinates of the piece
            piece.coord_x = piece.coord_x - 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s] 
    elif piece.orientation == 'h':
        if state.board.grid[piece.coord_y][piece.coord_x - 1] == '.':
            
            # updating the coordinates of the piece
            piece.coord_x = piece.coord_x - 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
    elif piece.orientation == 'v':
        if state.board.grid[piece.coord_y][piece.coord_x - 1] == '.' and state.board.grid[piece.coord_y + 1][piece.coord_x - 1] == '.':
            
            # updating the coordinates of the piece
            piece.coord_x = piece.coord_x - 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
            
    # If we have not returned till now, then the up-move is not possible
    
    return [-1, org_state]
    
    
    
    
def is_right(org_piece: Piece, org_state: State): # -> List[int, State]
    """
    Takes a piece and a State.
    Returns [-1, org_state] if the piece can't be moved right
    Returns [0, new_state] if the piece can be moved right
    """
    
    # Checking base (impossible) case
    if org_piece.is_single or org_piece.orientation == 'v':
        if org_piece.coord_x == 3:
            return [-1, org_state]
    if org_piece.is_goal or org_piece.orientation == 'h':
        if org_piece.coord_x == 2:
            return [-1, org_state]

         
    # making a copy
    piece = deepcopy(org_piece)
    state = org_state
    piece_lst = deepcopy(state.board.pieces)
    
    # removing the piece to be (possibly) modified from the list
    # piece_lst.remove(piece)
    for p in piece_lst:
        if p.is_goal == piece.is_goal and p.is_single == piece.is_single and p.coord_x == piece.coord_x and p.coord_y == piece.coord_y and p.orientation == piece.orientation:
            piece_lst.remove(p)
            break
    
    
    if piece.is_goal:
        if state.board.grid[piece.coord_y][piece.coord_x + 2] == '.' and state.board.grid[piece.coord_y + 1][piece.coord_x + 2] == '.':
            
            # updating the coordinates of the piece
            piece.coord_x = piece.coord_x + 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
    elif piece.is_single:
        if state.board.grid[piece.coord_y][piece.coord_x + 1] == '.':
            
            # updating the coordinates of the piece
            piece.coord_x = piece.coord_x + 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
    elif piece.orientation == 'h':
        if state.board.grid[piece.coord_y][piece.coord_x + 2] == '.':
            
            # updating the coordinates of the piece
            piece.coord_x = piece.coord_x + 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
    elif piece.orientation == 'v':
        if state.board.grid[piece.coord_y][piece.coord_x + 1] == '.' and state.board.grid[piece.coord_y + 1][piece.coord_x + 1] == '.':
            
            # updating the coordinates of the piece
            piece.coord_x = piece.coord_x + 1
            # Adding the updated piece to the piece_lst
            piece_lst.append(piece)
            
            # updating the state -
            
            b = Board(piece_lst)
            s = State(b, 0, org_state.depth + 1, org_state)

            s.f = s.depth + heuristic_Val(s)

            return [0, s]
            
    # If we have not returned till now, then the up-move is not possible
    
    return [-1, org_state]
    
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board = read_from_file(args.inputfile)
    
    # My commands
    
    output_file = args.outputfile
    algorithm = args.algo
    
    if algorithm == 'dfs':
        
        # Over here the f-value of 0 does not matter because this state would always
        # be automatically removed and added to the frontier
        init_state = State(board, 0, 0, None) 
        
        goal_state = generate_dfs(init_state)
        
        sol_lst = get_solution(goal_state)
        
        write_to_file(sol_lst, output_file)
        
    if algorithm == 'astar':
        
        # Over here the f-value of 0 does not matter because this state would always
        # be automatically removed and added to the frontier
        init_state = State(board, 0, 0, None) 
        
        goal_state = generate_astar(init_state)
        
        sol_lst = get_solution(goal_state)
        
        write_to_file(sol_lst, output_file)
    
    