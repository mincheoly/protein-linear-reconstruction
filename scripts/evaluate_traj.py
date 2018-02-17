# Filename: evaulate_traj.py
# Description: Contains functions to evaluate the reconstructed trajectories
# Authors: Christian Choe, Min Cheol Kim

import numpy as np

# This function returns the average RMSD error between each atom between each frame
def get_RMSD_error(md_obj):
	num_frames, num_atoms, num_axis = md_obj.xyz.shape
	recon = np.reshape(md_obj.xyz, (num_frames, num_atoms*num_axis))

	# Construct difference operator
	temp1 = np.identity(num_frames-1)
	temp1 = np.concatenate((temp1, np.zeros((num_frames-1, 1))), axis=1)
	temp2 = -1*np.identity(num_frames-1)
	temp2 = np.concatenate((np.zeros((num_frames-1, 1)), temp2), axis=1)
	difference_operator = temp1+temp2

	# Get difference matrix (each row is the difference between frames)
	difference_matrix = difference_operator.dot(recon)
	return np.sqrt((np.linalg.norm(difference_matrix)**2)/( (num_frames-1)*(num_atoms*num_axis)))

# Function: Creates the match matrix for use to calculate the score of each match
# Inputs:
#   num: number of frames to consider
# Returns: 
#   match: match matrix
def make_match_matrix(num):
    match = np.zeros((num, num))
    for i in range(0,num):
        for j in range(0,num):
            if i == j:
                match[i,j] = 1
            else:
                match[i,j] = 0
    return match

# Function: Create the matrix M, Ix, and Iy
# Inputs:
#   A: sequence A 
#   B: sequence B
#   gap_penalty: array containing the gap penalty for A and B
#   mrow: letters in alphabet for sequence A
#   mcol: letters in alphabet for sequence B
#   match: match matrix
#   M: Alignment matrix M (also contains the pointers)
#   Ix: Alignment matrix Ix (also contains the pointers)
#   Iy: Alignment matrix Iy (also contains the pointers)
#   x: length of A + 1
#   y: length of B + 1
#   align_type: boolean; global == True; local == False
# Returns: 
#   matrix: contains M, Ix, and Iy. M = [M, Ix, and Iy].
#           each matrix is two layered to also store the pointers
def make_M_Ix_Iy(A, B, gap_penalty, mrow, mcol, match, x, y, align_type):
    
    matrix = []
    for i in range(3): #to store matrix M, Ix, and Iy
        matrix.append(np.zeros((2, x, y)))

    for i in range(1,x):
        for j in range(1, y):
            match_i = A[i-1]
            match_j = B[j-1]
            score = round(match[match_i, match_j],2)
            fill_M(matrix[0], matrix[1], matrix[2], i, j, score, align_type)
            fill_Ix(matrix[0], matrix[1], i, j, gap_penalty[2], gap_penalty[3], align_type) #dy and ey
            fill_Iy(matrix[0], matrix[2], i, j, gap_penalty[0], gap_penalty[1], align_type) #dx and ey
    return matrix
    
# Function: Fills in matrix M
# Inputs:
#   M: Alignment matrix M (also contains the pointers)
#   Ix: Alignment matrix Ix (also contains the pointers)
#   Iy: Alignment matrix Iy (also contains the pointers)
#   i: specifies current row being examined
#   j: specifies current column being examined
#   align_type: boolean; global == True; local == False
# Returns: 
#   nothing
def fill_M(M, Ix, Iy, i, j, score, align_type):
    a = round(M[0][i - 1][j - 1] + score, 3) #score for M
    b = round(Ix[0][i - 1][j - 1] + score, 3) #score for Ix
    c = round(Iy[0][i - 1][j - 1] + score, 3) #score for Iy
    
    values = np.array([a, b, c])
    M[0][i][j] = np.amax(values)

    if (M[0][i][j] < 0) and (align_type == 0):
        M[0][i][j] = 0
    if a >= b and a >= c: #M is the best
        M[1][i][j] = 1
    if b >= a and b >= c: #Ix is the best
        M[1][i][j] = M[1][i][j]*10 + 2
    if c >= a and c >= b: #Iy is the best
        M[1][i][j] = M[1][i][j]*10 + 3

# Function: Fills in matrix Ix
# Inputs:
#   M: Alignment matrix M (also contains the pointers)
#   Ix: Alignment matrix Ix (also contains the pointers)
#   i: specifies current row being examined
#   j: specifies current column being examined
#   dy: opening gap penalty for B
#   ey: extending gap penalty for B
#   align_type: boolean; global == True; local == False
# Returns: 
#   nothing
def fill_Ix(M, Ix, i, j, dy, ey, align_type):
    a = round(M[0][i - 1][j] - dy, 3)
    b = round(Ix[0][i - 1][j] - ey, 3)
    
    values = np.array([a, b])
    Ix[0][i][j] = np.amax(values)
    
    if (Ix[0][i][j] < 0) and (align_type == 0):
        Ix[0][i][j] = 0
    if a >= b:
        Ix[1][i][j] = 1
    if b >= a:
        Ix[1][i][j] = Ix[1][i][j]*10 + 2

# Function: Fills in matrix Iy
# Inputs:
#   M: Alignment matrix M (also contains the pointers)
#   Iy: Alignment matrix Ix (also contains the pointers)
#   i: specifies current row being examined
#   j: specifies current column being examined
#   dx: opening gap penalty for A
#   ex: extending gap penalty for A
#   align_type: boolean; global == True; local == False
# Returns: 
#   nothing
def fill_Iy(M, Iy, i, j, dx, ex, align_type):
    a = round(M[0][i][j - 1] - dx, 3)
    b = round(Iy[0][i][j - 1] - ex, 3)

    values = np.array([a, b])
    Iy[0][i][j] = np.amax(values)

    if (Iy[0][i][j] < 0) and (align_type == 0):
        Iy[0][i][j] = 0
    if a >= b:
        Iy[1][i][j] = 1
    if b >= a:
        Iy[1][i][j] = Iy[1][i][j]*10 + 3

# Function: Finds the starting point for sequence alignment and conducts it
# Inputs:
#   A: sequence A 
#   B: sequence B
#   matrix: contains alignment matrix M, Ix, and Iy
#   x: length of A + 1
#   y: length of B + 1
#   align_type: boolean; global == True; local == False
# Returns: 
#   result: contains all the alignment results and alignment score
def get_score(A, B, matrix, x, y, align_type):
    M = matrix[0]
    if align_type:
        end = np.concatenate([M[0][x,:y], np.flipud(M[0][:,y])])
        seed = np.argwhere(end == np.amax(end)).flatten().tolist()
    else:
        end = M[0][:][:]
        seed = np.argwhere(end == np.amax(end)).flatten().tolist()
    
    return np.amax(end)

# Function: Run sequence alignment
# First it gathers the data from the file and then runs sequence alignment
# Inputs:
#   filename: name for the file that contains information for sequencing
#   output_filename: prints all the results into a file of the same name
# Returns: 
#   nothing
def seq_align(seq_A, seq_B, align_type, open_gap_penalty, extend_gap_penality):
    #gather data from file
    align_type = align_type == 0 #global vs local
    gap_penalty = [open_gap_penalty, extend_gap_penality, open_gap_penalty, extend_gap_penality]
    mrow = np.amax(np.hstack([seq_A ,seq_B])) + 1
    mcol = mrow

    #generate match matrix
    match = make_match_matrix(mrow)
    
    #calculate the M, Ix, and Iy matrix
    x = len(seq_A) + 1
    y = len(seq_B) + 1
    matrix = make_M_Ix_Iy(seq_A, seq_B, gap_penalty, mrow, mcol, match, x, y, align_type)

    #do global or local alignment
    result = get_score(seq_A, seq_B, matrix, x - 1, y - 1, align_type)
    return result

def calculate_frame_alignment_score(seq_A, seq_B, align_type, open_gap_penalty, extend_gap_penalty):
    score_fwd = seq_align(seq_A, seq_B, align_type, open_gap_penalty, extend_gap_penalty)
    seq_B = np.fliplr([np.array(seq_B)])[0]
    score_rev = seq_align(seq_A, seq_B, align_type, open_gap_penalty, extend_gap_penalty)
    return np.amax([score_fwd, score_rev])