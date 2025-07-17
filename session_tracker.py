import numpy as np

class SessionTracker():
    def __init__(self, n_timesteps, print_interval=None, window_length=64):
        self.current_timestep = 0
        self.n_timesteps = n_timesteps
        self.print_interval = print_interval
        self.window_length = window_length

        self.all_returns = []
        self.all_lengths = []

    def session_complete(self):
        return self.current_timestep >= self.n_timesteps

    def increment_timestep(self, n=1):
        previous_timestep = self.current_timestep

        self.current_timestep += n

        if self.print_interval and\
            previous_timestep // self.print_interval != self.current_timestep // self.print_interval:
            print(f"Current timestep: {self.current_timestep}, "
                  f"recent mean: {np.mean(self.all_returns[-self.window_length:]):.2f}, "
                  f"recent length: {np.mean(self.all_lengths[-self.window_length:]):.2f}, "
                  f"(window: {self.window_length})")

        return self.session_complete()
    
    def log_finished_episode(self, episode_return, episode_length):
        self.all_returns.append(episode_return)
        self.all_lengths.append(episode_length)

    def log_finished_episodes(self, episode_returns, episode_lengths):
        self.all_returns.extend(episode_returns)
        self.all_lengths.extend(episode_lengths)