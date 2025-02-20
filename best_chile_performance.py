import time
import math
import random

####################################################
# Mock definitions for testing
####################################################

# We'll define states and nodes for both methods.
# The "previous code" and the "modified code" methods differ,
# so we will implement both and test their performance.

# We'll assume we have constants C and heuristic_weight defined for the modified code.
C = 1.414  # for example, sqrt(2)
heuristic_weight = 0.5

class State:
    def __init__(self, delay, area):
        self.delay = delay
        self.area = area

class Node:
    def __init__(self, state, children=None, visit_times=None, quality_value=None, best_reward=None):
        self._state = state
        self._children = children if children is not None else []
        self._visit_times = visit_times if visit_times is not None else 1
        self._quality_value = quality_value if quality_value is not None else random.random()
        self._best_reward = best_reward if best_reward is not None else random.uniform(0,1)

    def get_state(self):
        return self._state

    def get_children(self):
        return self._children

    def get_visit_times(self):
        return self._visit_times

    def get_quality_value(self):
        return self._quality_value

    def get_best_reward(self):
        return self._best_reward


####################################################
# Previous best_child implementation (for reference)
####################################################
def best_child_previous(node, is_exploration):
    best_score = -math.inf
    best_sub_node = None
    for sub_node in node.get_children():
        if is_exploration:
            C_val = 1 / math.sqrt(2.0)
        else:
            C_val = 0.0

        if node.get_visit_times() >= 1e-2 and sub_node.get_visit_times() >= 1e-2:
            left = sub_node.get_best_reward() * 0.99 + \
                   sub_node.get_quality_value() / sub_node.get_visit_times() * 0.01
            right = math.log(node.get_visit_times()) / (sub_node.get_visit_times() + 1e-5)
            right = C_val * 10.0 * math.sqrt(right)
            score = left + right
        else:
            score = 1e9

        if score > best_score:
            best_sub_node = sub_node
            best_score = score

    return best_sub_node

####################################################
# compute_heuristic for the modified code
####################################################
def compute_heuristic(node):
    state = node.get_state()
    heuristic_value = -state.delay - state.area
    return heuristic_value

####################################################
# Modified best_child implementation with heuristic
####################################################
def best_child_modified(node, is_exploration):
    best_score = -float('inf')
    best_sub_node = None
    for sub_node in node.get_children():
        if sub_node.get_visit_times() == 0:
            score = float('inf')
        else:
            exploitation = sub_node.get_quality_value() / sub_node.get_visit_times()
            exploration = math.sqrt(2 * math.log(node.get_visit_times()) / sub_node.get_visit_times())
            heuristic = compute_heuristic(sub_node)
            score = exploitation + C * exploration + heuristic_weight * heuristic
        if score > best_score:
            best_sub_node = sub_node
            best_score = score
    return best_sub_node

####################################################
# Testing code
####################################################
if __name__ == "__main__":
    # We'll create a scenario to test performance:
    # 1) Build a node with a large number of children.
    # 2) For each child, assign random values for delay, area, visit_times, quality_value, and best_reward.
    # 3) Run both best_child methods multiple times to measure execution speed.
    
    num_children = 10000
    children = []
    for _ in range(num_children):
        delay = random.uniform(0, 10)
        area = random.uniform(0, 10)
        child_state = State(delay, area)

        visit_times = random.randint(1, 100)
        quality_value = random.uniform(0, 1)
        best_reward = random.uniform(0, 1)

        child_node = Node(
            state=child_state,
            children=[],
            visit_times=visit_times,
            quality_value=quality_value,
            best_reward=best_reward
        )
        children.append(child_node)

    root_state = State(5, 5)
    root_node = Node(
        state=root_state,
        children=children,
        visit_times=100,
        quality_value=10.0,
        best_reward=0.8
    )

    # Test previous best_child
    start_time = time.time()
    for _ in range(10):  # run multiple times for a better measure
        selected_child_prev = best_child_previous(root_node, is_exploration=True)
    end_time = time.time()
    prev_time = end_time - start_time

    # Test modified best_child
    start_time = time.time()
    for _ in range(10):
        selected_child_mod = best_child_modified(root_node, is_exploration=True)
    end_time = time.time()
    mod_time = end_time - start_time

    print("Previous best_child execution time:", prev_time, "seconds")
    print("Modified best_child execution time:", mod_time, "seconds")
