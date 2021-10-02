import numpy as np
import random
import matplotlib.pyplot as plt

""" discount factor, epsilon as from exploration """
gamma = 0.9
epsilon = 0.1


""" Some References: The problem statment says that we don't have any reward function and same time give rewards"""
""" Reference searched to see what would be best way to attempt it """
"https://stackoverflow.com/questions/39247265/stochastic-state-transitions-in-mdp-how-does-q-learning-estimate-that"
""" Assumes that state transition and rewards are know  """
"https://stats.stackexchange.com/questions/149231/why-and-when-does-one-have-to-learn-the-reward-function-from-samples-in-reinfo"







# wherever there is reward is not given we will consider -1 
""" for matrices reward and q-matrix columns are in order (U,D,L,R,N) """
""" first column for UP, then DOWN, LEFT and RIGHT respectively, final column is reward when no action is taken!"""
""" Here each ROW is a state and in total we have 38 states including terminal"""
reward = np.array([
    [0,-100,0,-1,-1],
    [0,0,-100,-1,-1],
    [0,-1,-100,-100,-1], #s3 -1,-1 instead -100 in both places
    [0,-1,-1,-1,-1],
    [0,0,-1,-1,-1],
    [0,-1,-1,-1,-1],
    [0,-1,-1,0,-1],
    [-100,-100,0,0,-1],
    [-1,-1,0,-1,-1], #s9
    [-1,-1,-1,0,-1],
    [-1,-1,0,-1,-1],
    [-1,-1,-1,0,-1],
    [-100,-100,0,-1,-1],
    [0,0,-100,-1,-1],
    [-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1], #s16
    [0,0,-1,-1,-1],
    [-1,-1,-1,-1,-1],
    [-1,-1,-1,0,-1],
    [-100,-100,0,0,-1],
    [-1,-1,0,-1,-1],
    [-1,-1,-1,0,-1],
    [-1,-1,0,-1,-1],
    [-1,-100,-1,0,-1],
    [-100,-100,0,-1,-1],
    [0,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1],
    [0,-1,-1,-1,-1],
    [-1,-1,-1,-100,-1], #s30
    [-100,100,-1,0,-1],
    [-100,0,0,-1,-1],
    [-1,0,-100,-1,-1],
    [-1,0,-1,-1,-1],
    [-1,0,-1,-1,-1],
    [-1,0,-1,-1,-1],
    [-1,0,-1,100,-1],
    [-1,0,-100,0,100]
])



""" Q-Value are initially started with zero and we update them with bellman equation """
Q_matrix = np.zeros((38,5))




# this is to inform the agent that from specific state where does it go.
""" Transition matrix is written by looking at graph given """
transition_matrix = np.array([
    [-1,7,-1,1,1],
    [-1,-1,0,2,2],
    [-1,8,1,3,3],
    [-1,9,2,4,4],
    [-1,-1,3,5,5],
    [-1,10,4,6,6],
    [-1,11,5,-1,7],
    [0,12,-1,-1,8],
    [2,14,-1,9,9],
    [3,15,8,-1,10],
    [5,17,-1,11,11],
    [6,18,10,-1,12],
    [7,19,-1,13,13],
    [-1,-1,12,14,14],
    [8,20,13,15,15],
    [9,21,14,16,16],
    [-1,-1,15,17,17],
    [10,22,16,18,18],
    [11,23,17,-1,19],
    [12,24,-1,-1,20],
    [14,26,-1,21,21],
    [15,27,20,-1,22],
    [17,29,-1,23,23],
    [18,30,22,-1,24],
    [19,31,-1,25,25],
    [-1,32,24,26,26],
    [20,33,25,27,27],
    [21,34,26,28,28],
    [-1,35,27,29,29],
    [22,36,28,30,30],
    [23,37,29,-1,31],
    [24,-1,-1,32,32],
    [25,-1,31,33,33],
    [26,-1,32,34,34],
    [27,-1,33,35,35],
    [28,-1,34,36,36],
    [29,-1,35,37,37],
    [30,-1,36,-1,37]
])




# reach row belongs to one state
""" For ease we have put legal action in separate matrix """
""" UP=0, DOWN=1, LEFT=2, RIGHT=3, NO-MOVE = 4 """
""" As these are the possible actions """
valid_actions = np.array([
    [1,3,4],
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],
    [2,3,4],
    [1,2,3,4],
    [1,2,4],
    [0,1,4],
    [0,1,3,4],
    [0,1,2,4],
    [0,1,3,4],
    [0,1,2,4],
    [0,1,3,4],
    [2,3,4],
    [0,1,2,3,4],
    [0,1,2,3,4],
    [2,3,4],
    [0,1,2,3,4],
    [0,1,2,4],
    [0,1,4],
    [0,1,3,4],
    [0,1,2,4],
    [0,1,3,4],
    [0,1,2,4],
    [0,1,3,4],
    [1,2,3,4],
    [0,1,2,3,4],
    [0,1,2,3,4],
    [1,2,3,4],
    [0,1,2,3,4],
    [0,1,2,4],
    [0,3,4],
    [0,2,3,4],
    [0,2,3,4],
    [0,2,3,4],
    [0,2,3,4],
    [0,2,3,4],
    [0,2,4]
    ])


#encoded up as 0, down as 1, left as 2, right as 3, no action as 4
 #up, down, left right, 
"""  This is N-matrix given and from where we took the learning parameter """
"""
N_matrix= np.array([
    [1,1,1,1,1],
    [127,164,145,4886,1],
    [375,6467,762,246,1],
    [119,2596,400,75,1],
    [189,134,377,4050,1],
    [136,5213,196,153,1],
    [85,2682,229,73,1],
    [1,1,1,1,1],
    [364,8734,6073,387,1],
    [173,5293,369,154,1],
    [244,9337,336,245,1],
    [228,8063,427,189,1],
    [1,1,1,1,1],
    [78,84,88,3066,1],
    [516,896,570,13483,1],
    [453,14768,486,429,1],
    [353,343,402,12661,1],
    [688,25908,702,685,1],
    [384,404,14336,385,1],
    [1,1,1,1,1],
    [343,343,1694,10328,1],
    [692,24141,707,680,1],
    [700,26368,773,688,1],
    [5897,185,1834,108,1],
    [1,1,1,1,1],
    [78,68,58,2330,1],
    [357,196,215,6666,1],
    [835,842,848,30436,1],
    [946,33274,915,1482,1],
    [808,30408,867,782,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [95,76,74,2790,1],
    [219,186,251,6463,1],
    [495,446,467,16600,1],
    [1189,1334,1273,45174,1],
    [1802,1739,1735,64171,1],
    [1,1,1,1,1]
])
"""

N_matrix = np.zeros((38,5))



""" utility function to take best Action from learned Q-values  """
def best_action_value(q_table, state):
    best_action = 0
    max_value = 0
    for action in range(4):
        if q_table[state, action] > max_value:
            best_action = action
            max_value = q_table[state,action]
    return  best_action

# remove function later on
""" best coarse of action """
def best_direction(q_table, state):
    ''' use of private variables'''
    _action = 0
    _value = 0
    for _action in range(3):
        if q_table[state, _action] > _value:
            _action = _action
            _value = q_table[state,_action]
    return  _action





""" Running 50,000 trials """
for i in range(5000): #episodes
    # start_state = random.randint(0,37)
    start_state = 0
    current_state =  start_state
    
    """ Checking whether we reached at goal state """
    while current_state !=37:
        
        """ Exploration vs Exploitation """
        action = 0 
        if  random.random() > epsilon:
            action = random.choice(valid_actions[current_state])
        else: 
            action = best_action_value(Q_matrix,current_state)
    
        
        next_state = transition_matrix[current_state][action]
    
        max_of_Q = np.max(Q_matrix[next_state,:])
  

        N_matrix[current_state][action]+=1 
        # alpha = 1/(N_matrix[current_state][action])
        alpha = 0.8
   
        """ Updating the Equation as given in assignment Report """
        # q_state =   Q_matrix[current_state][action] +  ( alpha *(reward[current_state][action] + (gamma*(max(future_rewards))-Q_matrix[current_state][action] )))
        #q_state =   Q_matrix[current_state][action] +  ( alpha *(reward[current_state][action] + (gamma*((max_of_Q)-Q_matrix[current_state][action]))))
        q_state =   Q_matrix[current_state][action] +  ( alpha * (reward[current_state][action] + (gamma*((max_of_Q)-Q_matrix[current_state][action]))))
        Q_matrix[current_state][action] = round(q_state,2)
        current_state = next_state




""" For displaying Q-Value in a formatted way. """
Result = {}

for i in range(len(Q_matrix)):
    # print("----------------")
    print("====STATE: "+str(i)+"===")
    print("   ","%.2f"%Q_matrix[i][0])
    print("","%.2f"%Q_matrix[i][2], end="  ")
    print(" ","%.2f"%Q_matrix[i][3])
    print("   ","%.2f"%Q_matrix[i][1])
    # print("----------------")
            


""" Only for display purposes. """
Result = {}
for i in range(len(Q_matrix)):
    max_number = 0
    state = 0
    direction = 0
    for j in range(len(Q_matrix[i])):
        #print("%.2f"%Q_matrix[i][j], end=',')
        if Q_matrix[i][j] > max_number:
            max_number = Q_matrix[i][j]
            state = i 
            direction = j
            if direction == 0:
                Result[i] = "^^^^"
            if direction == 1:
                Result[i] = "˅˅˅˅"
            if direction == 2:
                Result[i] = "<<<<"
            if direction == 3:
                Result[i] = ">>>>"
            



""" Showing the trajectory """
print('\n\n\n\n')
for i in range(len(Result)):
    if i ==7 or i==12 or i==19 or i==24 or i==31:
        print("\n")      
    if i==8 or i==10 or i==20 or i==22:
        print("####", end=' ')

    print(Result[i], end=' ')


print("\n")
print("STATE, UP, DOWN, LEFT, RIGHT")
for i in range(len(Q_matrix)):
    print(str(i)+"  :"+"  ^^"+str(Q_matrix[i][0])+"  ˅˅"+str(Q_matrix[i][1])+"  <<"+str(Q_matrix[i][2])+"  >>"+str(Q_matrix[i][3]))
# print(Q_matrix)

print("printing random stuff for viewing")
print("\n")
print("STATE, UP, DOWN, LEFT, RIGHT")
print("here are the N(s,a) values")
print("---------------------------")
for i in range(len(N_matrix)):
    print(str(i)+"  :"+"  ^^"+str(N_matrix[i][0])+"  ˅˅"+str(N_matrix[i][1])+"  <<"+str(N_matrix[i][2])+"  >>"+str(N_matrix[i][3]))