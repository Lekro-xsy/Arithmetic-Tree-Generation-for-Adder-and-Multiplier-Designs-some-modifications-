import numpy as np
import time

class State:
    def __init__(self, level, size, cell_map, level_map, min_map, step_num, action, reward):
        self.level = level
        self.size = size
        self.cell_map = cell_map
        self.level_map = level_map
        self.min_map = min_map
        self.step_num = step_num
        self.action = action
        self.reward = reward


class MyEnv:
    def __init__(self):
        self.input_bit = 2
        self.available_choice_list = list(range(8))  
        self.available_choice = len(self.available_choice_list)
        
        self.level = 0
        self.size = 20  
        self.step_num = 0
        
        self.cell_map = np.ones((4, 4))
        self.level_map = np.zeros((4, 4))
        self.min_map = np.ones((4, 4))
        
        self.level_bound = 2
        self.cumulative_choices = []

    def legalize(self, cell_map, min_map, start_bit):
        return cell_map, min_map, [start_bit]

    def update_level_map(self, cell_map, level_map, start_bit, activate_x_list):
        return level_map

    # ------------------------------
    # Original Code
    # ------------------------------
    def get_next_state_with_random_choice_original(self, set_action=None, remove_action=None):
        while self.available_choice > 0:
            sample_prob = np.ones((self.available_choice))
            choice_idx = np.random.choice(self.available_choice, size=1,
                                          replace=False, p=sample_prob / sample_prob.sum())[0]
            random_choice = self.available_choice_list[choice_idx]
            retry = 0
            while remove_action is not None and random_choice in remove_action and retry < 20:
                choice_idx = np.random.choice(self.available_choice, size=1,
                                              replace=False, p=sample_prob / sample_prob.sum())[0]
                random_choice = self.available_choice_list[choice_idx]
                retry += 1

            action_type = random_choice // (self.input_bit ** 2)  # self.input_bit=2 => self.input_bit**2=4
            x = (random_choice % (self.input_bit ** 2)) // self.input_bit
            y = (random_choice % (self.input_bit ** 2)) % self.input_bit

            if self.min_map[x, y] != 1 or self.cell_map[x, y] != 1:
                self.available_choice_list.remove(random_choice)
                self.available_choice -= 1
                continue

            if action_type != 1:
                self.available_choice_list.remove(random_choice)
                self.available_choice -= 1
                continue

            next_cell_map = np.copy(self.cell_map)
            next_level_map = np.copy(self.level_map)
            next_min_map = np.copy(self.min_map)
            next_cell_map[x, y] = 0
            next_min_map[x, y] = 0

            next_cell_map, next_min_map, activate_x_list = self.legalize(next_cell_map, next_min_map, start_bit=x)
            next_level_map = self.update_level_map(next_cell_map, next_level_map, start_bit=x, activate_x_list=activate_x_list)
            next_level = next_level_map.max()
            next_size = next_cell_map.sum() - self.input_bit
            next_step_num = self.step_num + 1

            if (next_level <= self.level and next_size <= self.size) or \
               (next_level < self.level and next_size <= self.size) or \
               (next_level <= self.level_bound and next_size <= self.size):
                reward = -1 + self.input_bit * (next_level - self.level)
                action = random_choice
                next_state = State(next_level, next_size, next_cell_map,
                                   next_level_map, next_min_map,
                                   next_step_num, action, reward)
                self.cumulative_choices.append(action)
                return next_state
            else:
                self.available_choice_list.remove(random_choice)
                self.available_choice -= 1

        return None

    # ------------------------------
    # Modified Code
    # ------------------------------
    def get_next_state_with_random_choice_modified(self, set_action=None, remove_action=None):
        heuristic_scores = []
        for choice in self.available_choice_list:
            action_type = choice // (self.input_bit ** 2)
            x = (choice % (self.input_bit ** 2)) // self.input_bit
            y = (choice % (self.input_bit ** 2)) % self.input_bit
            impact_score = self.level_map[x, y]
            heuristic_scores.append((choice, impact_score))

        heuristic_scores.sort(key=lambda x: x[1])

        for choice, _ in heuristic_scores:
            if remove_action is not None and choice in remove_action:
                continue

            action_type = choice // (self.input_bit ** 2)
            x = (choice % (self.input_bit ** 2)) // self.input_bit
            y = (choice % (self.input_bit ** 2)) % self.input_bit

            if self.min_map[x, y] != 1 or self.cell_map[x, y] != 1:
                continue

            if action_type != 1:
                continue

            next_cell_map = np.copy(self.cell_map)
            next_level_map = np.copy(self.level_map)
            next_min_map = np.copy(self.min_map)

            next_cell_map[x, y] = 0
            next_min_map[x, y] = 0

            next_cell_map, next_min_map, activate_x_list = self.legalize(next_cell_map, next_min_map, start_bit=x)
            next_level_map = self.update_level_map(next_cell_map, next_level_map, start_bit=x, activate_x_list=activate_x_list)
            next_level = next_level_map.max()
            next_size = next_cell_map.sum() - self.input_bit
            next_step_num = self.step_num + 1

            if (next_level <= self.level and next_size <= self.size) or \
               (next_level < self.level and next_size <= self.size) or \
               (next_level <= self.level_bound and next_size <= self.size):
                reward = -1 + self.input_bit * (next_level - self.level)
                action = choice
                next_state = State(next_level, next_size, next_cell_map,
                                   next_level_map, next_min_map,
                                   next_step_num, action, reward)
                self.cumulative_choices.append(action)
                return next_state

        return None


def run_performance_test(env_class, num_trials=1000, remove_action=None):
    original_success = 0
    original_rewards = []
    start_time = time.time()
    for _ in range(num_trials):
        env = env_class()
        next_state = env.get_next_state_with_random_choice_original(remove_action=remove_action)
        if next_state is not None:
            original_success += 1
            original_rewards.append(next_state.reward)
    original_time = time.time() - start_time
    modified_success = 0
    modified_rewards = []
    start_time = time.time()
    for _ in range(num_trials):
        env = env_class()
        next_state = env.get_next_state_with_random_choice_modified(remove_action=remove_action)
        if next_state is not None:
            modified_success += 1
            modified_rewards.append(next_state.reward)
    modified_time = time.time() - start_time

    original_success_rate = original_success / num_trials
    modified_success_rate = modified_success / num_trials
    original_avg_reward = np.mean(original_rewards) if original_rewards else float('nan')
    modified_avg_reward = np.mean(modified_rewards) if modified_rewards else float('nan')

    return {
        "original_time": original_time,
        "modified_time": modified_time,
        "original_success_rate": original_success_rate,
        "modified_success_rate": modified_success_rate,
        "original_avg_reward": original_avg_reward,
        "modified_avg_reward": modified_avg_reward
    }


if __name__ == "__main__":
    results = run_performance_test(MyEnv, num_trials=1000, remove_action=None)

    print("Performance Comparison:")
    print(f"Original Code Total Time: {results['original_time']:.4f}s")
    print(f"Modified Code Total Time: {results['modified_time']:.4f}s")
    print(f"Original Success Rate: {results['original_success_rate']*100:.2f}%")
    print(f"Modified Success Rate: {results['modified_success_rate']*100:.2f}%")
    print(f"Original Average Reward: {results['original_avg_reward']:.4f}")
    print(f"Modified Average Reward: {results['modified_avg_reward']:.4f}")
