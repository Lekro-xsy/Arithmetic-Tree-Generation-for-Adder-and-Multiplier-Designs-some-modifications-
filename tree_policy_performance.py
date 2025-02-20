import time
import random


class Args:
    max_iter = 1000  # for testing

args = Args()
global_iter = 0

class State:
    def __init__(self, terminal=False):
        self.terminal = terminal

    def is_terminal(self):
        return self.terminal

class Node:
    def __init__(self, state, children=None, expanded=False):
        self._state = state
        self._children = children if children else []
        self._expanded = expanded

    def get_state(self):
        return self._state

    def get_children(self):
        return self._children

    def is_all_expand(self):
        return self._expanded

def best_child(node, is_exploration):
    if node.get_children():
        return random.choice(node.get_children())
    return node

def expand(node):
    new_child = Node(state=State(terminal=(random.random() < 0.01)), children=[], expanded=False)
    node.get_children().append(new_child)
    node._expanded = True
    return new_child


####################################################
# Previous tree_policy implementation
####################################################
def tree_policy_previous(node):
    global global_iter
    if global_iter >= args.max_iter:
        return None
    eps = 0.8
    while not node.get_state().is_terminal():
        if global_iter >= args.max_iter:
            return None
        # Increment global_iter to simulate iteration count
        global_iter += 1
        if node.is_all_expand() or (random.random() > eps and len(node.get_children()) >= 1):
            # print("IS ALL EXPAND")
            node = best_child(node, True)
        else:
            node = expand(node)
            break
    return node

####################################################
# Modified tree_policy with Epsilon Decay
####################################################
def tree_policy_modified(node):
    global global_iter
    if global_iter >= args.max_iter:
        return None
    initial_eps = 0.8
    min_eps = 0.1
    decay_rate = 0.99  # Decay rate for epsilon
    eps = max(min_eps, initial_eps * (decay_rate ** (global_iter // 100)))
    while not node.get_state().is_terminal():
        if global_iter >= args.max_iter:
            return None
        # Increment global_iter to simulate iteration count
        global_iter += 1
        if node.is_all_expand() or (random.random() > eps and len(node.get_children()) >= 1):
            # print("IS ALL EXPAND")
            node = best_child(node, True)
        else:
            node = expand(node)
            break
    return node


####################################################
# Testing code
####################################################
if __name__ == "__main__":

    root_state = State(terminal=False)
    root_node = Node(root_state, children=[], expanded=False)
    global_iter = 0
    start_time = time.time()
    for _ in range(1000):
        test_root = Node(root_state, children=[], expanded=False)
        result_node = tree_policy_previous(test_root)
    end_time = time.time()
    previous_time = end_time - start_time
    global_iter = 0
    start_time = time.time()
    for _ in range(1000):
        test_root = Node(root_state, children=[], expanded=False)
        result_node = tree_policy_modified(test_root)
    end_time = time.time()
    modified_time = end_time - start_time

    print("Previous tree_policy execution time:", previous_time, "seconds")
    print("Modified tree_policy execution time:", modified_time, "seconds")
