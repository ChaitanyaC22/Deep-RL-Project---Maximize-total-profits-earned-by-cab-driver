# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 0 ..... m-1
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        ## Retain (0,0) state and all states of the form (i,j) where i!=j
        self.action_space= [(p,q) for p in range(m) for q in range(m) if p!=q]
        self.action_space.insert(0, (0,0))         ## Insert (0,0) at index 0 for no ride action   
        
        ## All possible combinations of (m,t,d) in state_space
        self.state_space = [(loc,time,day) for loc in range(m) for time in range(t) for day in range(d)]   
        
        ## Random state initialization
        self.state_init = self.state_space[np.random.choice(len(self.state_space))] 

        # Start the first round
        self.reset()
        


    ## Encoding state for NN input
    ## NOTE: Considering Architecture 2 given in the problem statement (where Input: STATE ONLY)
    ## --->Used in Agent_Architecture2_(Input_State_only).ipynb

    def state_encod_arch2(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint:
        The vector is of size m + t + d."""
        curr_loc, curr_time, curr_day= state

        ## Initialize arrays
        loc_arr = np.zeros(m, dtype=int)   # For location
        time_arr= np.zeros(t, dtype=int)   # For time
        day_arr= np.zeros(d, dtype= int)   # For day

        ## Encoding respective arrays
        loc_arr[curr_loc] = 1
        time_arr[curr_time] = 1
        day_arr[curr_day] = 1

        ## Horizontal stacking to get vector of size m+t+d
        state_encod= np.hstack((loc_arr, time_arr, day_arr))
        state_encod= state_encod.tolist()
       
        return state_encod

    ## Encoding (state-action) for NN input
    ## Use this function if you are using architecture-1 
    ## def state_encod_arch2(self, state, action):
    ##     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into
    ## a vector format. Hint: The vector is of size m + t + d + m + m."""


    
    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
        if requests >15:
            requests =15
        
        ## (0,0) implies no action. The driver is free to refuse customer request at any point in time.
        ## Hence, add the index of  action (0,0)->[0] to account for no ride action.
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0]  
       
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index, actions   


    def update_time_day(self, curr_time, curr_day, ride_duration):
        """
        Takes in the current time, current day and duration taken for driver's journey and returns
        updated time and updated day post that journey.
        """
        ride_duration = int(ride_duration)

        if (curr_time + ride_duration) < 24:
            updated_time = curr_time + ride_duration  
            updated_day= curr_day                        # Meaning, day is unchanged
        else:
            # duration spreads over to subsequent days
            # convert the time to 0-23 range
            updated_time = (curr_time + ride_duration) % 24 
            
            # Get the number of days
            num_days = (curr_time + ride_duration) // 24
            
            # Convert the day to 0-6 range
            updated_day = (curr_day + num_days ) % 7

        return updated_time, updated_day
    
    def get_next_state_and_time_func(self, state, action, Time_matrix):
        """Takes state, action and Time_matrix as input and returns next state, wait_time, transit_time, ride_time."""
        next_state = []
        
        # Initialize various times
        total_time   = 0
        transit_time = 0         # To go from current location to pickup location
        wait_time = 0    # in case driver chooses to refuse all requests. for action: (0,0) 
        ride_time    = 0         # From Pick-up to drop
        
        # Derive the current location, time, day and request locations
        curr_loc, curr_time, curr_day = state
        pickup_loc, drop_loc= action
        
        """
         3 Possible Scenarios: 
           i) Refuse all requests. Engage in Idle Time (wait: 1hr (i.e. 1 time unit))
           ii) Driver is already at the pickup spot
           iii) Driver is not at the pickup spot
        """    
        if ((pickup_loc== 0) and (drop_loc == 0)):
            wait_time = 1    # Refuse all requests, so wait time is 1 unit, next location is current location
            next_loc = curr_loc
        elif (curr_loc == pickup_loc):
            # Means driver is already at the pickup spot. Thus, the wait and transit are both 0
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            
            # Next location is the drop location
            next_loc = drop_loc
        else:
            # Driver is not at the pickup spot. He/she needs to commute to the pickup spot from the curr_loc
            # Time take to reach pickup spot (from current location to pickup spot)
            transit_time      = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            new_time, new_day = self.update_time_day(curr_time, curr_day, transit_time)
            
            # The cab driver is now at the pickup spot
            # Time taken to drop the passenger
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            next_loc  = drop_loc

        # Calculate total time as sum of all durations
        total_time = (wait_time + transit_time + ride_time)
        next_time, next_day = self.update_time_day(curr_time, curr_day, total_time)
        
        # Finding next_state using the next_loc and the next time states.
        next_state = [next_loc, next_time, next_day]
        
        return next_state, wait_time, transit_time, ride_time
        
    def next_state_func(self, state, action, Time_matrix):
        """Takes state, action and Time_matrix as input and returns next state"""
        next_state= self.get_next_state_and_time_func(state, action, Time_matrix)[0]   ## get_next_state_and_time_func() defined above       
        return next_state
        
    
    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time_matrix and returns the reward"""
        ## get_next_state_and_time_func() defined above 
        wait_time, transit_time, ride_time = self.get_next_state_and_time_func(state, action, Time_matrix)[1:]
        
        
        # transit and wait time yield no revenue and consumes battery; leading to battery charging costs. So, these are idle times.
        idle_time = wait_time + transit_time
        customer_ride_time = ride_time
        
        reward = (R * customer_ride_time) - (C * (customer_ride_time + idle_time))
        #Returns (-C) in case of no action i.e. customer_ride_time is 0, leading to battery charging costs. Hence, penalizing the model
        
        return reward
    
    def step(self, state, action, Time_matrix):
        """
        Take a trip as a cab driver. Takes state, action and Time_matrix as input and returns next_state, reward and total time spent
        """
        # Get the next state and the various time durations
        next_state, wait_time, transit_time, ride_time = self.get_next_state_and_time_func(state, action, Time_matrix)

        # Calculate the reward and total_time of the step
        reward = self.reward_func(state, action, Time_matrix)
        total_time = wait_time + transit_time + ride_time
        
        return next_state, reward, total_time

    def reset(self):
        return self.action_space, self.state_space, self.state_init
