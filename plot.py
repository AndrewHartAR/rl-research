import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# params should be a dict
def params_to_title(params):
    # Create params_title with line breaks after every 5 parameters
    param_items = []

    for key, value in params.items():
        param_items.append(f"{key}: {value}")
    
    # Add line breaks after every 5 parameters
    params_title = ""
    for i, item in enumerate(param_items):
        if i > 0 and i % 5 == 0:
            params_title += "\n"
        params_title += item
        if i < len(param_items) - 1:
            params_title += ", "

    return params_title

def scores_timesteps_to_linspace(scores, timesteps, total_timesteps, increments):
    x_sampled = np.arange(0, total_timesteps, increments)

    # # Create interpolation function
    interp_func = interp1d(
        timesteps, scores,
        kind='linear', bounds_error=False,
        fill_value=(scores[0], scores[-1]))

    # # Get the interpolated values
    y_sampled = interp_func(x_sampled)

    return x_sampled, y_sampled

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

# each session should be a tuple of (scores, timesteps)
def plot_sessions_with_timesteps(sessions, title=None, ylabel='Score', ylim=None, average_window=None, n_timesteps=None):
    alpha = (1 / len(sessions)) ** 0.5

    all_scores_and_timesteps = []

    linspaces = []

    plt.figure(figsize=(10, 6))
    
    for i, session in enumerate(sessions):
        scores, timesteps = session

        for j in range(len(scores)):
            all_scores_and_timesteps.append((scores[j], timesteps[j]))

        label = 'Scores' if i == 0 else None
        plt.plot(timesteps, scores, label=label, color='#1f77b4', alpha=alpha)

        # # add plot for average over the surrounding `average_window` episodes
        if average_window is not None:
            averages = moving_average(scores, window=average_window)
            truncated_timesteps = moving_average(timesteps, window=average_window)
            label = 'Average' if i == 0 else None
            plt.plot(truncated_timesteps, averages, label=label, color='#d62728', alpha=alpha, linewidth=2)
            plt.legend()

            ls = scores_timesteps_to_linspace(averages, truncated_timesteps, n_timesteps, increments=64)
            linspaces.append(ls)
            
    all_scores_and_timesteps = sorted(all_scores_and_timesteps, key=lambda x: x[1])

    # now plot the overall average across all runs
    if average_window is not None:
        # each linspace is a tuple of (x_sampled, y_sampled)
        # each x_sampled is the same, so we can just take the first one
        x_sampled = linspaces[0][0]
        y_sampled = np.mean([ls[1] for ls in linspaces], axis=0)

        plt.plot(x_sampled, y_sampled, label='Overall Average', color='#d62728', linewidth=2)
        plt.legend()

    plt.xlabel('Timestep')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if n_timesteps is not None:
        plt.xlim(0, n_timesteps)

    plt.ylabel(ylabel)
    
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.show()

def plot_session_with_timesteps(scores, timesteps, title=None, ylabel='Score', ylim=None, average_window=None, n_timesteps=None):
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, scores, label='Scores', color='#1f77b4')

    # # add plot for average over the surrounding `average_window` episodes
    if average_window is not None:
        averages = moving_average(scores, window=average_window)
        truncated_timesteps = moving_average(timesteps, window=average_window)
        print(averages.shape, truncated_timesteps.shape)
        plt.plot(truncated_timesteps, averages, label='Average', color='#d62728', linewidth=2)
        plt.legend()
            
    plt.xlabel('Timestep')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if n_timesteps is not None:
        plt.xlim(0, n_timesteps)

    plt.ylabel(ylabel)
    
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.show()

def ylim_for_env(env_name):
    """
    Returns the y-axis limits for the given environment.
    :param env_name: (str) name of the environment
    :return: (list) [min, max] y-axis limits
    """
    if env_name == 'CartPole-v1':
        return [0, 600]
    elif env_name == 'MountainCar-v0':
        return [-200, 0]
    elif env_name == 'Acrobot-v1':
        return [-500, 0]
    elif env_name == 'LunarLander-v3':
        return [-600, 600]
    else:
        return [0, 1000]  # default limits for other environments

def plot_session(scores, episode_lengths, algo_name, env_name, n_timesteps, method_title=None, params=None, window_length=64):
    mean_score = np.mean(scores[-window_length:]) if len(scores) >= window_length else np.mean(scores)
    mean_length = np.mean(episode_lengths[-window_length:]) if len(episode_lengths) >= window_length else np.mean(episode_lengths)

    timesteps = np.cumsum(episode_lengths)

    params_str = f'\n{params_to_title(params)},' if params else ''

    plot_session_with_timesteps(
        scores, timesteps,
        title=f'{algo_name}, {env_name}{f", {method_title}" if method_title else ""},'
        f'{params_str}'
        f'\nepisodes: {len(scores):.2f}, (mean scores: {mean_score:.2f}, '
        f'mean length: {mean_length:.2f}) -{window_length} eps',
        ylim=ylim_for_env(env_name),
        average_window=window_length,
        n_timesteps=n_timesteps)
    
def plot_sessions(all_results, algo_name, env_name, n_timesteps, method_title=None, params=None, window_length=64):
    all_scores = [scores for scores, _ in all_results]
    all_episode_lengths = [episode_lengths for _, episode_lengths in all_results]

    all_timesteps = [np.cumsum(episode_lengths) for episode_lengths in all_episode_lengths]
    sessions = [(all_scores[i], all_timesteps[i]) for i in range(len(all_scores))]

    flattened_scores = [score for scores in all_scores for score in scores]

    last_scores_mean = np.mean([np.mean(scores[-window_length:]) for scores in all_scores])
    last_episode_lengths_mean = np.mean([np.mean(episode_lengths[-window_length:])
        for episode_lengths in all_episode_lengths])
    
    params_str = f'\n{params_to_title(params)},' if params else ''

    plot_sessions_with_timesteps(
        sessions,
        title=f'{algo_name}, {env_name}{f", {method_title}" if method_title else ""}, '
        f'sessions: {len(all_results)},'
        f'{params_str}'
        f'\nepisodes: {len(flattened_scores):.2f}, (mean scores: {last_scores_mean:.2f}, '
        f'mean length: {last_episode_lengths_mean:.2f}) -{window_length} eps',
        ylim=ylim_for_env(env_name),
        average_window=window_length,
        n_timesteps=n_timesteps)