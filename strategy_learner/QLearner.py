"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import numpy as np
import random as rand


def author():
    return 'cli620'


class QLearner(object):

    def __init__(self, num_states=100, num_actions = 4, alpha = 0.2, gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar= rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.experience = []
        self.Q = np.random.uniform(0.0, 1.0, size=(self.num_states, self.num_actions))
        # self.R = np.zeros([self.num_states, self.num_actions], dtype='float')

    def author(self):
        return 'cli620'

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if rand.random() <= self.rar:
            # This is random if our random value is less than the probability rar
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[self.s,:]) # this is the expected action (most likely)

        # action = rand.randint(0, self.num_actions-1)
        if self.verbose:
            print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """ 			  		 			 	 	 		 		 	  		   	  			  	
        @summary: Update the Q table and return an action 			  		 			 	 	 		 		 	  		   	  			  	
        @param s_prime: The new state 			  		 			 	 	 		 		 	  		   	  			  	
        @param r: The ne state 			  		 			 	 	 		 		 	  		   	  			  	
        @returns: The selected action 			  		 			 	 	 		 		 	  		   	  			  	
        """

        # Update Q
        # learning instance --> <s, a, s', r>
        self.experience.append([self.s, self.a, s_prime, r])
        self.Q[self.s, self.a] =(1.0-self.alpha)*self.Q[self.s, self.a] + self.alpha*(r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime,:])])

        # Return an integer --> next action.
        # choose random action with probability rar --> update rar according to decay rate radr at each step.
        # This is if it is a random action.
        # High rar = good chance it will be random.
        if rand.random() <= self.rar:
            # This is random if our random value is less than the probability rar
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime,:]) # this is the expected action (most likely)

        if self.dyna != 0:
            nothing = 1
            # Pseudo code
            # random index of value between 0 and self.dyna
            # while looping through all the indices
            # update T (the T used via random index)
            # grab the return of this T
            # Next T --> this T / sum of T
            # save off the R correlated with the T
            # grab random s, a (done via random indices)
            # update Q with the new s,a,s',r (T)

            for i in range(self.dyna):
                # randomize s and a
                thisindex = rand.randint(0,len(self.experience)-1)
                # grab the T and R matrix.
                [s_exp, a_exp, s_prime_exp, r_exp]=self.experience[thisindex]

                self.Q[s_exp, a_exp] = (1.0-self.alpha)*self.Q[s_exp, a_exp] + self.alpha * (r_exp + self.gamma * self.Q[s_prime_exp, np.argmax(self.Q[s_prime_exp,:])])


        self.rar= self.rar * self.radr # the actions will become less and less random.
        self.s = s_prime
        self.a = action
        if self.verbose:
            print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
