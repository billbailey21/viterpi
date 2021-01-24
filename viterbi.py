import numpy as np

class Observation:
    def __init__(self, value):
        self.value = value

class State:
    def __init__(self, state_name):
        self.state_name = state_name

    def emissionProbability(self, observation):
        return 0

    def transitionProbability(self, from_observation, to_state, to_observation):
        return 0

    def initialProbability(self, observation):
        return 0

    def __str__(self):
        return self.state_name

    def __repr__(self):
        return f"[State] {self.state_name}"


class ObservedStateProbability:
    def __init__(self, state, observation, logarithmic):
        self.state = state
        self.observation = observation
        self.logarithmic = logarithmic

        self.viterbi_probability = 0
        self.backpointer = None
    
    def setToComputedProbability(self, previous_states):
        # Calculate probability of observation first--if 0, can skip
        emit_prob = self.state.emissionProbability(self.observation)

        if emit_prob==0:
            if self.logarithmic:
                self.viterbi_probability = np.log2(0)
            else:
                self.viterbi_probability = 0
        else:
            # Calculate most probable previous state
            probable_state = None
            if self.logarithmic:
                highest_probability = np.NINF
            else:
                highest_probability = 0

            for previous_state in previous_states:
                previous_prob = previous_state.viterbi_probability
                trans_prob = previous_state.computeTransitionProbability(self)
                
                if self.logarithmic:
                    total_trans_prob = trans_prob + previous_prob
                else:
                    total_trans_prob = trans_prob * previous_prob
                
                if total_trans_prob > highest_probability:
                    probable_state = previous_state
                    highest_probability = total_trans_prob
            
            self.backpointer = probable_state
            
            if self.logarithmic:
                self.viterbi_probability = np.log2(emit_prob) + highest_probability
            else:
                self.viterbi_probability = emit_prob * highest_probability
    

    def computeTransitionProbability(self, next_probable_state):
        probability = self.state.transitionProbability(self.observation, next_probable_state.state, next_probable_state.observation)

        if self.logarithmic:
            probability = np.log2(probability)
        
        return probability

    def setToInitialProbability(self):
        init_prob = self.state.initialProbability(self.observation)
        emit_prob = self.state.emissionProbability(self.observation)

        if self.logarithmic:
            self.viterbi_probability = np.log2(init_prob) + np.log2(emit_prob)
        else:
            self.viterbi_probability = init_prob * emit_prob


class ViterbiAlgorithm:
    
    def __init__(self, states, observations, logarithmic=True):
        self.logarithmic = logarithmic

        self.states = states
        self.observations = observations
        self.trellis = [[ObservedStateProbability(s, o, self.logarithmic) for s in states] for o in observations]

    def solve(self):
        first = True

        for state_list in self.trellis:
            for probable_state in state_list:
                if first:
                    probable_state.setToInitialProbability()
                else:
                    probable_state.setToComputedProbability(previous_states)
            first = False
            previous_states = state_list  # Carry forward to the next iteration

        probable_state_path = [None for n in self.observations]

        # Find maximum final state, and traverse using backpointers
        max_prob = np.NINF if self.logarithmic else 0
        max_state = None
        for probable_state in self.trellis[-1]:
            if probable_state.viterbi_probability > max_prob:
                max_state = probable_state
                max_prob = max_state.viterbi_probability
        
        # Traverse the maximum state
        backpointer = max_state.backpointer
        probable_state_path[-1] = max_state
        i = len(self.trellis)-2  # Last index is filled already

        while backpointer is not None and i >= 0:
            probable_state_path[i] = backpointer
            backpointer = backpointer.backpointer
            i -= 1

        return probable_state_path

    
if __name__=="__main__":
    # Super simple example to test it out on: https://www.cis.upenn.edu/~cis262/notes/Example-Viterbi-DNA.pdf
    class HState(State):
        def __init__(self):
            super().__init__('H')
            self.emissionDict = {'A': 0.2, 'C': 0.3, 'G': 0.3, 'T': 0.2}

        def emissionProbability(self, observation):
            return self.emissionDict[observation]

        def transitionProbability(self, from_observation, to_state, to_observation):
            return 0.5  # Both are the same for "H"

        def initialProbability(self, observation):
            return 0.5


    class LState(State):
        def __init__(self):
            super().__init__('L')
            self.emissionDict = {'A': 0.3, 'C': 0.2, 'G': 0.2, 'T': 0.3}

        def emissionProbability(self, observation):
            return self.emissionDict[observation]

        def transitionProbability(self, from_observation, to_state, to_observation):
            if type(to_state) is LState:
                return 0.6
            else:
                return 0.4

        def initialProbability(self, observation):
            return 0.5
    h_state = HState()
    l_state = LState()
    observation = ['G','G','C','A','C','T','G','A','A']

    solver = ViterbiAlgorithm([h_state, l_state], observation, False)
    sequence = solver.solve()

    for probable_state in sequence:
        if probable_state is None:
            print("None")
        else:
            print(probable_state.state.state_name)
