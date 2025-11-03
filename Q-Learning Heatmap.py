#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:50:42 2025

@author: hiro
"""
import os
os.chdir('/Users/hiro/Documents/RL Project with Giordano')
os.listdir()

import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import Counter

np.random.seed(14)

import pickle

def save_to_pickle(data, filename):
    """
    Save data to a pickle file.

    Parameters:
        data (dict): Dictionary containing the data to save.
        filename (str): Name of the file to save the data in.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data successfully saved to {filename}")

def load_from_pickle(filename):
    """
    Load data from a pickle file.

    Parameters:
        filename (str): Name of the file to load the data from.

    Returns:
        dict: The data loaded from the pickle file.
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Data successfully loaded from {filename}")
    return data

def q_learning_2x2_game_with_pushforward(payoffs, T, alpha, beta0, k, messages, message_probabilities):
    
    
    message_probabilities = np.array(message_probabilities) / np.sum(message_probabilities)
    if not np.isclose(sum(message_probabilities), 1) == True:
        raise ValueError("Message probabilities must sum to 1.")
    num_players = 2
    num_actions = 2
    msg_1 = [sublist[0] for sublist in messages]
    msg_2 = [sublist[1] for sublist in messages]
    msgs = [list(set(sublist)) for sublist in [msg_1, msg_2]]
    msg_1, msg_2 = msgs
    
    
    def choose_message(messages, probabilities):
        index = np.random.choice(len(messages), p=probabilities)
        return messages[index]
    
    choose_message(messages, message_probabilities)
    
    num_messages = len(messages)
    
    Q = [
        {message: np.zeros(num_actions) for message in msgs[_]}
        for _ in range(num_players)
    ]
    #[{'m1': array([0., 0.]), 'm2': array([0., 0.])}, Q-values foraction 0 and action 1 for message 1
    #{'m1': array([0., 0.]), 'm2': array([0., 0.])}]
    
    strategies = [{message: [] for message in msgs[i]} for i in range(num_players)]
    action_counts = [
        {message: list(np.zeros(num_actions)) for message in msgs[i]}
        for i in range(num_players)
    ]
    
    action_history = [[] for _ in range(num_players)]
    
    # Compute the average correlated strategy (a distribution over joint actions)
    action_profiles = [(0, 0), (0, 1), (1, 0), (1, 1)]  # All possible joint actions (a1, a2)
    #correlated_strategy = {profile: 0 for profile in action_profiles}
    
    #2
    for t in range(1, T + 1):
        # Sample a message
        
        beta = beta0 + k*t
        message = choose_message(messages, probabilities=message_probabilities)
        
        mixed_strategies = []
        
        # Compute mixed strategies for the selected message
        for i in range(num_players):
            mixed_strategy = np.exp(beta * Q[i][message[i]] - np.max(beta * Q[i][message[i]]))
            mixed_strategy /= np.sum(mixed_strategy)
            mixed_strategies.append(mixed_strategy)
            
        for i in range(num_players):
            strategies[i][message[i]].append(mixed_strategies[i]) 
        
        
        actions = [np.random.choice(num_actions, p=mixed_strategies[i]) for i in range(num_players)] 
        
        # Record actions
        for i in range(num_players): 
            action_history[i].append(actions[i]) 
        
        # Update action counts
        for i in range(num_players):
            action_counts[i][message[i]][actions[i]] += 1
        
        # Compute payoffs for the current actions
        #rewards
        #rewards = [payoffs[i][actions[i]][actions[1-i]] for i in range(num_players)]
        
        # Update Q-values for the selected message
        for i in range(num_players):
            for a in range(num_actions):
                Q[i][message[i]][a] = (1-alpha) * Q[i][message[i]][a] + payoffs[i][a][actions[1-i]]

            #Q[i][message[i]][actions[i]] = (1-alpha) * Q[i][message[i]][actions[i]] + rewards[i]
    
        
    #3 
    
    avg_mixed_strategies = []
    
    left_mean_0_m1 = sum(arr[0] for arr in strategies[0]['m1'])/ len(strategies[0]['m1']) if len(strategies[0]['m1']) > 0 else 0
    left_mean_0_m2 = sum(arr[0] for arr in strategies[0]['m2']) / len(strategies[0]['m2']) if len(strategies[0]['m2']) > 0 else 0
    
    right_mean_0_m1 = sum(arr[1] for arr in strategies[0]['m1']) / len(strategies[0]['m1']) if len(strategies[0]['m1']) > 0 else 0
    right_mean_0_m2 = sum(arr[1] for arr in strategies[0]['m2']) / len(strategies[0]['m2']) if len(strategies[0]['m2']) > 0 else 0
         
    left_mean_1_m1 = sum(arr[0] for arr in strategies[1]['m1']) / len(strategies[1]['m1']) if len(strategies[1]['m1']) > 0 else 0
    left_mean_1_m2 = sum(arr[0] for arr in strategies[1]['m2']) / len(strategies[1]['m2']) if len(strategies[1]['m2']) > 0 else 0
    
    right_mean_1_m1 = sum(arr[1] for arr in strategies[1]['m1']) / len(strategies[1]['m1']) if len(strategies[1]['m1']) > 0 else 0
    right_mean_1_m2 = sum(arr[1] for arr in strategies[1]['m2']) / len(strategies[1]['m2']) if len(strategies[1]['m2']) > 0 else 0
    
    
    avg_mixed_strategies = [{'m1': [left_mean_0_m1, right_mean_0_m1], 'm2': [left_mean_0_m2, right_mean_0_m2]}, {'m1': [left_mean_1_m1, right_mean_1_m1], 'm2': [left_mean_1_m2, right_mean_1_m2]}]
    
    print(avg_mixed_strategies)
    
    
    expected_payoff_player0 = 0.0
    expected_payoff_player1 = 0.0
    
    
    for i, (m0, m1) in enumerate(messages):
        prob_m = message_probabilities[i]
        if prob_m == 0:
            continue  # No contribution if probability is zero
        
        # Retrieve each player's mixed strategy for these messages
        p0 = avg_mixed_strategies[0][m0]  # e.g. [p(0), p(1)] for player 0
        p1 = avg_mixed_strategies[1][m1]  # e.g. [p(0), p(1)] for player 1
    
        # Compute conditional expected payoffs
        # sum_{a0 in {0,1}} sum_{a1 in {0,1}} payoffs[i][a0][a1] * p0[a0] * p1[a1]
        E0 = 0.0
        E1 = 0.0
        for a0 in [0, 1]:
            for a1 in [0, 1]:
                joint_prob = p0[a0] * p1[a1]
                E0 += payoffs[0][a0][a1] * joint_prob
                E1 += payoffs[1][a0][a1] * joint_prob
        
        # Weight by the probability of this message profile
        expected_payoff_player0 += prob_m * E0
        expected_payoff_player1 += prob_m * E1
        
    avg_social_welfare = expected_payoff_player0 + expected_payoff_player1

    
    last_iterate_strategies = [{m: strategies[i][m][-1] if len(strategies[i][m]) > 0 else [0.5, 0.5] for m in msgs[i]}  for i in range(num_players)]
    

    
    expected_payoff_player0 = 0.0
    expected_payoff_player1 = 0.0
    
    
    for i, (m0, m1) in enumerate(messages):
        prob_m = message_probabilities[i]
        if prob_m == 0:
            continue  # No contribution if probability is zero
        
        # Retrieve each player's mixed strategy for these messages
        p0 = last_iterate_strategies[0][m0]  # e.g. [p(0), p(1)] for player 0
        p1 = last_iterate_strategies[1][m1]  # e.g. [p(0), p(1)] for player 1
    
        # Compute conditional expected payoffs
        # sum_{a0 in {0,1}} sum_{a1 in {0,1}} payoffs[i][a0][a1] * p0[a0] * p1[a1]
        E0 = 0.0
        E1 = 0.0
        for a0 in [0, 1]:
            for a1 in [0, 1]:
                joint_prob = p0[a0] * p1[a1]
                E0 += payoffs[0][a0][a1] * joint_prob
                E1 += payoffs[1][a0][a1] * joint_prob
        
        # Weight by the probability of this message profile
        expected_payoff_player0 += prob_m * E0
        expected_payoff_player1 += prob_m * E1
        
    last_social_welfare = expected_payoff_player0 + expected_payoff_player1

    
    return {
        "last_mixed_strategy": last_iterate_strategies,
        "avg_mixed_strategy": avg_mixed_strategies,
        "avg_social_welfare": avg_social_welfare,
        "last_social_welfare": last_social_welfare,
        #"pushforward_average_strategy": total_average_payoff,
        #"pushforward_last_strategy": total_average_payoff_last,
        "action_history": action_history,
        "strategies": strategies
        
    }


def q_plot(payoffs, T, alpha, beta0, k, messages, message_probabilities, z):
    mixed_strategy_history = [{'m1': [], 'm2': []},{'m1': [], 'm2': []}]
    for j in range(10):
        results = q_learning_2x2_game_with_pushforward(payoffs, T, alpha, beta0, k, messages, message_probabilities)
        for i in range(num_players):
            for m in ms:
                mixed_strategy_history[i][m].append(results['strategies'][i][m])
                
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    
    for i in range(num_players):
        for j, m in enumerate(ms):
            ax = axes[i, j]

            # Plot each of the 10 simulation time-series:
            for sim_idx, sim_data in enumerate(mixed_strategy_history[i][m]):
                # sim_data is assumed shape (T, 2). 
                # We'll plot the probability of playing action 1 vs. time.
                sim_data = np.array(sim_data)  # Now sim_data is T×2
                prob_action_0 = sim_data[:, 0]
                ax.plot(prob_action_0,
                        color=colors[sim_idx],
                        alpha=0.4,
                        linewidth=2,
                        label=f"Simulation {sim_idx+1}" if sim_idx == 0 else None)

            # Make it look nice
            ax.set_title(f"Player {i}, Message '{m}'", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Probability of Action 0", fontsize=12)
            ax.set_ylim(0, 1)  # Probabilities should be between 0 and 1
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # If you want a single legend outside, you can do this:
    # fig.legend(*axes[0,0].get_legend_handles_labels(), loc="upper center",
    #            bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=10)

    plt.tight_layout()
    fig.savefig(f"qlearning_results{z}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def q_plot_single(results):
    mixed_strategy_history = [{'m1': [], 'm2': []}, {'m1': [], 'm2': []}]
    
    # Extract the strategies directly from the results variable
    num_players = 2  # Assuming you have 2 players
    ms = ['m1', 'm2']  # Messages

    for i in range(num_players):
        for m in ms:
            mixed_strategy_history[i][m] = results['strategies'][i][m]

    # Plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    
    for i in range(num_players):
        for j, m in enumerate(ms):
            ax = axes[i, j]

            # Extract time-series data for this player and message
            sim_data = np.array(mixed_strategy_history[i][m])  # Now sim_data is T×2
            if len(sim_data) > 0:  # Check to avoid errors with empty data
                prob_action_0 = sim_data[:, 0]  # Probability of playing action 0
                ax.plot(prob_action_0,
                        color='blue',
                        alpha=0.8,
                        linewidth=2,
                        label=f"Player {i}, Message '{m}'")

            # Make it look nice
            ax.set_title(f"Player {i}, Message '{m}'", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Probability of Action 0", fontsize=12)
            ax.set_ylim(0, 1)  # Probabilities should be between 0 and 1
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig("qlearning_single_results.pdf", format="pdf", bbox_inches="tight")
    plt.show()





def count_combinations(results_list, threshold=0.995):
    """
    Count the occurrences of specific combinations of last_mixed_strategy for m1 and m2.
    
    Parameters:
        results_list (list): A list of results dictionaries from the simulations.
        threshold (float): The threshold to consider a probability as close to 1 or 0.
        
    Returns:
        dict: A dictionary with combinations as keys and counts as values.
    """
    def classify(probabilities, threshold):
        """Classify a probability vector as [1, 0] or [0, 1] if close enough."""
        if probabilities[0] >= threshold:
            return "[1, 0]"
        elif probabilities[1] >= threshold:
            return "[0, 1]"
        return "Other"
    
    combination_counts = Counter()

    for results in results_list:
        # Extract the last mixed strategies
        m1_p0 = classify(results['last_mixed_strategy'][0]['m1'], threshold)
        m1_p1 = classify(results['last_mixed_strategy'][1]['m1'], threshold)
        m2_p0 = classify(results['last_mixed_strategy'][0]['m2'], threshold)
        m2_p1 = classify(results['last_mixed_strategy'][1]['m2'], threshold)
        
        # Store combinations of interest
        combination_1 = (m1_p0, m1_p1)
        combination_2 = (m1_p0, m2_p1)
        combination_3 = (m2_p0, m1_p1)
        
        # Increment counts
        combination_counts[combination_1] += 1
        combination_counts[combination_2] += 1
        combination_counts[combination_3] += 1
    
    return combination_counts

# Example Usage:
# Assuming `results_list` is a list of dictionaries obtained from your simulations
# results_list = [results_sim1, results_sim2, ..., results_simN]





def count_combinations_simple(results_list, threshold=0.995):
    """
    Count occurrences of distinct patterns of the 6 results in the simulations.
    
    Parameters:
        results_list (list): A list of results dictionaries from the simulations.
        threshold (float): Threshold to classify probabilities as [1, 0] or [0, 1].
        
    Returns:
        Counter: A count of all unique patterns.
    """
    def classify(probabilities, threshold):
        """Classify probabilities as [1, 0] or [0, 1] if close to the extremes."""
        probabilities = np.array(probabilities)
        if probabilities[0] >= threshold:
            return "[1, 0]"
        elif probabilities[1] >= threshold:
            return "[0, 1]"
        return "Other"

    patterns = []

    for results in results_list:
        # Classify the three key combinations
        combination_1 = (
            classify(results['last_mixed_strategy'][0]['m1'], threshold),
            classify(results['last_mixed_strategy'][1]['m1'], threshold),
        )
        combination_2 = (
            classify(results['last_mixed_strategy'][0]['m1'], threshold),
            classify(results['last_mixed_strategy'][1]['m2'], threshold),
        )
        combination_3 = (
            classify(results['last_mixed_strategy'][0]['m2'], threshold),
            classify(results['last_mixed_strategy'][1]['m1'], threshold),
        )
        
        # Group all three combinations as a tuple (to handle them together)
        patterns.append((combination_1, combination_2, combination_3))

    # Count all unique patterns
    return Counter(map(tuple, patterns))

def simulate_and_count(beta0_values, k_values, T, payoffs, messages, message_probabilities, threshold=0.995):
    correlated_eq_pattern = (
        ('[1, 0]', '[1, 0]'),
        ('[1, 0]', '[0, 1]'),
        ('[0, 1]', '[1, 0]')
    )
    nash_eq_patterns = [
        correlated_eq_pattern,
        (
            ('[0, 1]', '[1, 0]'),
            ('[0, 1]', '[1, 0]'),
            ('[0, 1]', '[1, 0]')
        ),
        (
            ('[1, 0]', '[0, 1]'),
            ('[1, 0]', '[0, 1]'),
            ('[1, 0]', '[0, 1]')
        )
    ]
    
    correlated_eq_percentages = np.zeros((len(beta0_values), len(k_values)))
    correlated_nash_percentages = np.zeros((len(beta0_values), len(k_values)))
    
    for i, beta0 in enumerate(beta0_values):
        for j, k in enumerate(k_values):
            correlated_count = 0
            nash_count = 0
            
            for _ in range(100):  # 100 simulations
                results = q_learning_2x2_game_with_pushforward(
                    payoffs=payoffs,
                    T=T,
                    alpha=0,
                    beta0=beta0,
                    k=k,
                    messages=messages,
                    message_probabilities=message_probabilities
                )
                patterns = count_combinations_simple([results], threshold)
                
                # Track matched patterns to avoid double counting
                matched_patterns = set()
                
                # Check correlated equilibrium
                if correlated_eq_pattern in patterns:
                    correlated_count += 1
                    matched_patterns.add(correlated_eq_pattern)
                    
                # Check Nash equilibria
                for pattern in nash_eq_patterns:
                    if pattern in patterns and pattern not in matched_patterns:
                        nash_count += 1
                        matched_patterns.add(pattern)
            
            correlated_eq_percentages[i, j] = (correlated_count / 100) * 100
            correlated_nash_percentages[i, j] = ((nash_count + correlated_count) / 100) * 100
    
    return correlated_eq_percentages, correlated_nash_percentages


def simulate_alpha_beta(payoffs, T, messages, message_probabilities, alpha_values, beta_values, threshold=0.995):
    """
    Simulate the game for varying alpha and beta with k = 0.
    
    Parameters:
        payoffs (list): Payoff matrix for the 2x2 game.
        T (int): Number of iterations for each simulation.
        messages (list): List of messages.
        message_probabilities (list): Probabilities for each message.
        alpha_values (list): List of alpha values to test.
        beta_values (list): List of beta values to test.
        threshold (float): Threshold to classify probabilities as [1, 0] or [0, 1].
    
    Returns:
        tuple: Two heatmaps as 2D numpy arrays for correlated equilibrium and correlated/Nash equilibrium percentages.
    """
    correlated_eq_pattern = (
        ('[1, 0]', '[1, 0]'),
        ('[1, 0]', '[0, 1]'),
        ('[0, 1]', '[1, 0]')
    )
    nash_eq_patterns = [
        correlated_eq_pattern,
        (
            ('[0, 1]', '[1, 0]'),
            ('[0, 1]', '[1, 0]'),
            ('[0, 1]', '[1, 0]')
        ),
        (
            ('[1, 0]', '[0, 1]'),
            ('[1, 0]', '[0, 1]'),
            ('[1, 0]', '[0, 1]')
        )
    ]
    
    correlated_eq_percentages = np.zeros((len(alpha_values), len(beta_values)))
    correlated_nash_percentages = np.zeros((len(alpha_values), len(beta_values)))
    
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            correlated_count = 0
            nash_count = 0
            
            for _ in range(100):  # 100 simulations per (alpha, beta)
                results = q_learning_2x2_game_with_pushforward(
                    payoffs=payoffs,
                    T=T,
                    alpha=alpha,
                    beta0=beta,  # Constant beta
                    k=0,  # No growth for beta
                    messages=messages,
                    message_probabilities=message_probabilities
                )
                patterns = count_combinations_simple([results], threshold)
                
                for pattern in patterns.keys():
                    if pattern == correlated_eq_pattern:
                        correlated_count += patterns[pattern]
                    if pattern in nash_eq_patterns:
                        nash_count += patterns[pattern]
            
            correlated_eq_percentages[i, j] = (correlated_count / 100) * 100
            correlated_nash_percentages[i, j] = (nash_count / 100) * 100
    
    return correlated_eq_percentages, correlated_nash_percentages

def heatmap_varying_beta0_k(payoffs, T, messages, message_probabilities, beta0_values, k_values, threshold=0.995):
    """
    Generate heatmaps for correlated equilibrium and correlated/Nash equilibrium percentages
    while varying beta0 and k.

    Parameters:
        payoffs (list): Payoff matrix for the 2x2 game.
        T (int): Number of iterations for each simulation.
        messages (list): List of messages.
        message_probabilities (list): Probabilities for each message.
        beta0_values (list): List of beta0 values to test.
        k_values (list): List of k values to test.
        threshold (float): Threshold to classify probabilities as [1, 0] or [0, 1].

    Returns:
        tuple: Two heatmaps as 2D numpy arrays for correlated equilibrium and correlated/Nash equilibrium percentages.
    """
    correlated_eq_percentages, correlated_nash_percentages = simulate_and_count(
        beta0_values, k_values, T, payoffs, messages, message_probabilities, threshold
    )

    return correlated_eq_percentages, correlated_nash_percentages


def heatmap_varying_alpha_beta(payoffs, T, messages, message_probabilities, alpha_values, beta_values, threshold=0.995):
    """
    Generate heatmaps for correlated equilibrium and correlated/Nash equilibrium percentages
    while varying alpha and beta with k = 0.

    Parameters:
        payoffs (list): Payoff matrix for the 2x2 game.
        T (int): Number of iterations for each simulation.
        messages (list): List of messages.
        message_probabilities (list): Probabilities for each message.
        alpha_values (list): List of alpha values to test.
        beta_values (list): List of beta values to test.
        threshold (float): Threshold to classify probabilities as [1, 0] or [0, 1].

    Returns:
        tuple: Two heatmaps as 2D numpy arrays for correlated equilibrium and correlated/Nash equilibrium percentages.
    """
    correlated_eq_percentages, correlated_nash_percentages = simulate_alpha_beta(
        payoffs, T, messages, message_probabilities, alpha_values, beta_values, threshold
    )

    return correlated_eq_percentages, correlated_nash_percentages



def plot_heatmaps(correlated_eq_percentages, correlated_nash_percentages, x_values, y_values, x_label, y_label, extra=None, filename="heatmaps_clean.pdf"):
    """
    Plot clean heatmaps for correlated equilibrium and correlated/Nash equilibrium percentages,
    without any gridlines or additional marks.

    Parameters:
        correlated_eq_percentages (np.ndarray): Heatmap data for correlated equilibrium percentages.
        correlated_nash_percentages (np.ndarray): Heatmap data for correlated/Nash equilibrium percentages.
        x_values (list or np.ndarray): Values for the x-axis.
        y_values (list or np.ndarray): Values for the y-axis.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        extra (str): Extra text to add to the title.
        filename (str): Filename to save the heatmaps.
    """
    plt.style.use('seaborn-v0_8-dark')
    extra_text = f" ({extra})" if extra else ""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Compute exact extent for proper axis alignment
    x_extent = [x_values[0], x_values[-1]]
    y_extent = [y_values[0], y_values[-1]]

    # Correlated Equilibrium Heatmap
    im1 = axes[0].imshow(correlated_eq_percentages, aspect='auto', origin='lower', 
                          extent=x_extent + y_extent, cmap='viridis')
    axes[0].set_title(f"Correlated Equilibrium Percentage{extra_text}")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].tick_params(left=False, bottom=False)  # Remove ticks
    plt.colorbar(im1, ax=axes[0])

    # Correlated Equilibrium/Nash Heatmap
    im2 = axes[1].imshow(correlated_nash_percentages, aspect='auto', origin='lower', 
                          extent=x_extent + y_extent, cmap='plasma')
    axes[1].set_title(f"Correlated/Nash Equilibrium Percentage{extra_text}")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].tick_params(left=False, bottom=False)  # Remove ticks
    plt.colorbar(im2, ax=axes[1])

    # Adjust layout and save the figure
    plt.tight_layout()
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Clean heatmaps saved to {filename}")
    plt.show()


payoffs = [
    [[6, 2], [7, 0]],  # Payoffs for Player 0
    [[6, 2], [7, 0]]   # Payoffs for Player 1
]
messages = [['m1', 'm1'], ['m1', 'm2'], ['m2', 'm1'], ['m2', 'm2']]
num_players = 2
num_actions = 2
ms = ['m1', 'm2']




#############################################################################################
# Prototype 1

"""
# Define a smaller test setup
beta0_values_test = np.linspace(0, 1.0, 3)  # Reduced range for testing
k_values_test = np.linspace(0, 1.0, 3)      # Reduced range for testing
T_test = 30  # Reduced number of iterations for testing
message_probabilities_test = [1/3, 1/3, 1/3, 0]  # Example message probabilities for testing

# Run the smaller test for beta0 and k
correlated_eq_test, correlated_nash_test = heatmap_varying_beta0_k(
    payoffs=payoffs,
    T=T_test,
    messages=messages,
    message_probabilities=message_probabilities_test,
    beta0_values=beta0_values_test,
    k_values=k_values_test
)

# Plot the results of the smaller test
plot_heatmaps(
    correlated_eq_test,
    correlated_nash_test,
    beta0_values_test,
    k_values_test,
    x_label="k (test)",
    y_label="beta0 (test)",
    extra="Test Run",
    filename="heatmaps_test.pdf"
)
"""
# Correlated Eq, same Beta
#"""
#T, alpha, beta0, k= 5000, 0, 1, 0
#message_probabilities = [0.5, 0.25, 0.25, 0]

message_probabilities = [1/3, 1/3, 1/3, 0]



beta0_values = np.linspace(0, 1.0, 11)
k_values = np.linspace(0, 1.0, 11)

correlated_eq, correlated_nash = heatmap_varying_beta0_k(
    payoffs=payoffs,  # Your payoff matrix
    #T=5000,
    T = 20000,
    messages=messages,  # Your messages
    message_probabilities=message_probabilities,  # Your probabilities
    beta0_values=beta0_values,
    k_values=k_values
)
save_to_pickle(correlated_eq, "correlated_eq1.pkl")
save_to_pickle(correlated_nash, "correlated_nash1.pkl")

correlated_eq = load_from_pickle("correlated_eq1.pkl")
correlated_nash = load_from_pickle("correlated_nash1.pkl")

plot_heatmaps(correlated_eq, correlated_nash, beta0_values, k_values, "k", "beta0", extra = " [v.1]", filename="heatmaps_varying_k [v.1].pdf")


alpha_values = np.linspace(0, 1.0, 11)
beta_values = np.linspace(0, 1.0, 11)
"""
correlated_eq, correlated_nash = heatmap_varying_alpha_beta(
    payoffs=payoffs,  # Your payoff matrix
    T=5000,
    messages=messages,  # Your messages
    message_probabilities=message_probabilities,  # Your probabilities
    alpha_values=alpha_values,
    beta_values=beta_values
)

save_to_pickle(correlated_eq, "correlated_eq2.pkl")
save_to_pickle(correlated_nash, "correlated_nash2.pkl")
"""
correlated_eq = load_from_pickle("correlated_eq2.pkl")
correlated_nash = load_from_pickle("correlated_nash2.pkl")

plot_heatmaps(correlated_eq, correlated_nash, beta_values, alpha_values, "Beta", "Alpha",  extra = " [v.1]", filename="heatmaps_varying_alpha [v.1].pdf")




message_probabilities = [1/2, 1/4, 1/4, 0]
beta0_values = np.linspace(0, 1.0, 11)
k_values = np.linspace(0, 1.0, 11)
"""
correlated_eq, correlated_nash = heatmap_varying_beta0_k(
    payoffs=payoffs,  # Your payoff matrix
    T=5000,
    messages=messages,  # Your messages
    message_probabilities=message_probabilities,  # Your probabilities
    beta0_values=beta0_values,
    k_values=k_values
)

save_to_pickle(correlated_eq, "correlated_eq3.pkl")
save_to_pickle(correlated_nash, "correlated_nash3.pkl")
"""
correlated_eq = load_from_pickle("correlated_eq3.pkl")
correlated_nash = load_from_pickle("correlated_nash3.pkl")

plot_heatmaps(correlated_eq, correlated_nash, beta0_values, k_values, "k", "beta0", extra = " [v.2]", filename = "heatmaps_varying_k [v.2].pdf")


alpha_values = np.linspace(0, 1.0, 11)
beta_values = np.linspace(0, 1.0, 11)
"""
correlated_eq, correlated_nash = heatmap_varying_alpha_beta(
    payoffs=payoffs,  # Your payoff matrix
    T=5000,
    messages=messages,  # Your messages
    message_probabilities=message_probabilities,  # Your probabilities
    alpha_values=alpha_values,
    beta_values=beta_values
)

save_to_pickle(correlated_eq, "correlated_eq4.pkl")
save_to_pickle(correlated_nash, "correlated_nash4.pkl")
"""
correlated_eq = load_from_pickle("correlated_eq4.pkl")
correlated_nash = load_from_pickle("correlated_nash4.pkl")
plot_heatmaps(correlated_eq, correlated_nash, beta_values, alpha_values, "Beta", "Alpha",  extra = " [v.2]", filename="heatmaps_varying_alpha [v.2].pdf")



##



alpha_values = np.linspace(0, 1.0, 11)
beta_values = np.linspace(0, 2.0, 11)

correlated_eq, correlated_nash = heatmap_varying_alpha_beta(
    payoffs=payoffs,  # Your payoff matrix
    T=5000,
    messages=messages,  # Your messages
    message_probabilities=message_probabilities,  # Your probabilities
    alpha_values=alpha_values,
    beta_values=beta_values
)

save_to_pickle(correlated_eq, "correlated_eq5.pkl")
save_to_pickle(correlated_nash, "correlated_nash5.pkl")

correlated_eq = load_from_pickle("correlated_eq5.pkl")
correlated_nash = load_from_pickle("correlated_nash5.pkl")

plot_heatmaps(correlated_eq, correlated_nash, beta_values, alpha_values, "Beta", "Alpha",  extra = " [v.1]", filename="heatmaps_varying_alpha [v.1].pdf")



########
#Do these!













#"""

"""
results_list = []


start = time()
for i in range(100):
    results = q_learning_2x2_game_with_pushforward(payoffs, T, alpha, beta0, k, messages, message_probabilities)
    results_list.append(results)
end = time()
time_difference = end - start
print(f"Time difference is {time_difference}")
    
    
pattern_counts = count_combinations_simple(results_list)

for pattern, count in pattern_counts.items():
    print(f"Pattern {pattern}: {count}")
    
"""


results_list = []

T, alpha, beta0, k= 5000, 0, 0.5, 0
#message_probabilities = [0.5, 0.25, 0.25, 0]

message_probabilities = [1/3, 1/3, 1/3, 0]


start = time()
for i in range(100):
    results = q_learning_2x2_game_with_pushforward(payoffs, T, alpha, beta0, k, messages, message_probabilities)
    results_list.append(results)
end = time()
time_difference = end - start
print(f"Time difference is {time_difference}")
    
    
pattern_counts = count_combinations_simple(results_list)

for pattern, count in pattern_counts.items():
    print(f"Pattern {pattern}: {count}")
    
###################
T, alpha, beta0, k= 5000, 0, 0.5, 0


message_probabilities = [1/3, 1/3, 1/3, 0]
q_plot(payoffs, T, alpha, beta0, k, messages, message_probabilities, 10)

z = 10

mixed_strategy_history = [{'m1': [], 'm2': []},{'m1': [], 'm2': []}]
for j in range(10):
    results = q_learning_2x2_game_with_pushforward(payoffs, T, alpha, beta0, k, messages, message_probabilities)
    for i in range(num_players):
        for m in ms:
            mixed_strategy_history[i][m].append(results['strategies'][i][m])
            
plt.style.use('seaborn-v0_8-dark')
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, 10))

for i in range(num_players):
    for j, m in enumerate(ms):
        ax = axes[i, j]

        # Plot each of the 10 simulation time-series:
        for sim_idx, sim_data in enumerate(mixed_strategy_history[i][m]):
            # sim_data is assumed shape (T, 2). 
            # We'll plot the probability of playing action 1 vs. time.
            sim_data = np.array(sim_data)  # Now sim_data is T×2
            prob_action_0 = sim_data[:, 0]
            ax.plot(prob_action_0,
                    color=colors[sim_idx],
                    alpha=0.4,
                    linewidth=2,
                    label=f"Simulation {sim_idx+1}" if sim_idx == 0 else None)

        # Make it look nice
        ax.set_title(f"Player {i}, Message '{m}'", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Probability of Action 0", fontsize=12)
        ax.set_ylim(0, 1)  # Probabilities should be between 0 and 1
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

# If you want a single legend outside, you can do this:
# fig.legend(*axes[0,0].get_legend_handles_labels(), loc="upper center",
#            bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=10)

plt.tight_layout()
fig.savefig(f"qlearning_results{z}.pdf", format="pdf", bbox_inches="tight")
plt.show()

################################

T, alpha, beta0, k= 5000, 0, 0.5, 0
results_list = []

for z in range(500):
    results = q_learning_2x2_game_with_pushforward(payoffs, T, alpha, beta0, k, messages, message_probabilities)
    if (
        np.array_equal(results['last_mixed_strategy'][0]['m1'], np.array([1., 0.])) and
        np.array_equal(results['last_mixed_strategy'][1]['m1'], np.array([1., 0.])) and
        np.array_equal(results['last_mixed_strategy'][0]['m1'], np.array([1., 0.])) and
        np.array_equal(results['last_mixed_strategy'][1]['m2'], np.array([0., 1.])) and
        np.array_equal(results['last_mixed_strategy'][0]['m2'], np.array([0., 1.])) and
        np.array_equal(results['last_mixed_strategy'][1]['m1'], np.array([1., 0.]))
    ):
        structured_result = [
            {
                "m1": results['strategies'][0]['m1'],  # Convert to list for JSON compatibility
                "m2": results['strategies'][0]['m2']
            },
            {
                "m1": results['strategies'][1]['m1'],
                "m2": results['strategies'][1]['m2']
            }
        ]
        results_list.append(structured_result)


results_list 



def q_plot_from_results(results_list, T):
    """
    Plot the mixed strategy histories from the given results_list.

    Parameters:
    results_list (list): A list of structured results containing strategies.
    T (int): Number of time steps.
    """
    mixed_strategy_history = [{'m1': [], 'm2': []}, {'m1': [], 'm2': []}]

    # Process results_list to extract mixed strategies
    for results in results_list:
        for i in range(2):  # Two players
            for m in ['m1', 'm2']:
                mixed_strategy_history[i][m].append(results[i][m])

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))

    for i in range(2):  # Players
        for j, m in enumerate(['m1', 'm2']):
            ax = axes[i, j]

            # Plot each simulation time-series:
            for sim_idx, sim_data in enumerate(mixed_strategy_history[i][m]):
                # sim_data is assumed shape (T, 2). We'll plot the probability of action 0 vs. time.
                sim_data = np.array(sim_data)  # Now sim_data is T×2
                prob_action_0 = sim_data[:, 0]
                ax.plot(prob_action_0,
                        color=colors[sim_idx],
                        alpha=0.6,
                        linewidth=2,
                        label=f"Simulation {sim_idx+1}" if sim_idx == 0 else None)

            # Make it look nice
            ax.set_title(f"Player {i+1}, Message '{m}'", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Probability of Action 0", fontsize=12)
            ax.set_ylim(0, 1)  # Probabilities should be between 0 and 1
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("qlearning_results.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# Example usage:
# Assuming results_list contains the first 10 structured results
q_plot_from_results(results_list, T)


########################################################################################################



def compute_social_welfare(avg_mixed_strategy, payoffs, message_probabilities, messages):
    """
    Compute social welfare (utilitarian aggregation) using average mixed strategies.

    Parameters:
        avg_mixed_strategy (list): Average mixed strategies for both players.
        payoffs (list): Payoff matrices for both players.
        message_probabilities (list): Probabilities for each message.
        messages (list): List of message pairs.

    Returns:
        float: Computed social welfare.
    """
    social_welfare = 0.0

    for i, (m0, m1) in enumerate(messages):
        prob_m = message_probabilities[i]
        if prob_m == 0:
            continue  # Skip messages with zero probability

        # Retrieve each player's mixed strategy for these messages
        p0 = avg_mixed_strategy[0][m0]  # e.g., [p(0), p(1)] for player 0
        p1 = avg_mixed_strategy[1][m1]  # e.g., [p(0), p(1)] for player 1

        # Compute expected social welfare for this message profile
        for a0 in range(len(p0)):
            for a1 in range(len(p1)):
                joint_prob = p0[a0] * p1[a1]
                social_welfare += prob_m * joint_prob * (payoffs[0][a0][a1] + payoffs[1][a0][a1])

    return social_welfare

def heatmap_social_welfare_alpha_beta(payoffs, T, alpha_values, beta_values, messages, message_probabilities):
    """
    Generate a heatmap for social welfare over varying alpha and beta values.

    Parameters:
        payoffs (list): Payoff matrices for both players.
        T (int): Number of iterations.
        alpha_values (list): Values for alpha.
        beta_values (list): Values for beta.
        messages (list): List of message pairs.
        message_probabilities (list): Probabilities for each message.
    """
    welfare_matrix = np.zeros((len(alpha_values), len(beta_values)))

    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            # Run the simulation
            results = q_learning_2x2_game_with_pushforward(
                payoffs, T, alpha, beta, 0, messages, message_probabilities
            )

            # Compute social welfare using avg_mixed_strategy
            social_welfare = compute_social_welfare(
                results['avg_mixed_strategy'], payoffs, message_probabilities, messages
            )
            welfare_matrix[i, j] = social_welfare

    # Plot the heatmap
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(10, 8))
    plt.imshow(welfare_matrix, origin='lower', aspect='auto', cmap='viridis',
               extent=[beta_values[0], beta_values[-1], alpha_values[0], alpha_values[-1]])
    plt.colorbar(label="Social Welfare")
    plt.title("Social Welfare Heatmap (Alpha vs. Beta)", fontsize=16)
    plt.xlabel("Beta", fontsize=14)
    plt.ylabel("Alpha", fontsize=14)
    plt.tight_layout()
    plt.savefig("social_welfare_heatmap_alpha_beta.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def heatmap_social_welfare_beta0_k(payoffs, T, beta0_values, k_values, messages, message_probabilities):
    """
    Generate a heatmap for social welfare over varying beta0 and k values.

    Parameters:
        payoffs (list): Payoff matrices for both players.
        T (int): Number of iterations.
        beta0_values (list): Values for beta0.
        k_values (list): Values for k.
        messages (list): List of message pairs.
        message_probabilities (list): Probabilities for each message.
    """
    welfare_matrix = np.zeros((len(beta0_values), len(k_values)))

    for i, beta0 in enumerate(beta0_values):
        for j, k in enumerate(k_values):
            # Run the simulation
            results = q_learning_2x2_game_with_pushforward(
                payoffs, T, 0, beta0, k, messages, message_probabilities
            )

            # Compute social welfare using avg_mixed_strategy
            social_welfare = compute_social_welfare(
                results['avg_mixed_strategy'], payoffs, message_probabilities, messages
            )
            welfare_matrix[i, j] = social_welfare

    # Plot the heatmap
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(10, 8))
    plt.imshow(welfare_matrix, origin='lower', aspect='auto', cmap='plasma',
               extent=[k_values[0], k_values[-1], beta0_values[0], beta0_values[-1]])
    plt.colorbar(label="Social Welfare")
    plt.title("Social Welfare Heatmap (Beta0 vs. k)", fontsize=16)
    plt.xlabel("k", fontsize=14)
    plt.ylabel("Beta0", fontsize=14)
    plt.tight_layout()
    plt.savefig("social_welfare_heatmap_beta0_k.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# Parameters for simulation
alpha_values = np.linspace(0, 1.0, 11)
beta_values = np.linspace(0, 1.0, 11)
beta0_values = np.linspace(0, 1.0, 11)
k_values = np.linspace(0, 0.01, 11)

# Generate and plot the heatmaps
heatmap_social_welfare_alpha_beta(
    payoffs=payoffs,
    T=5000,
    alpha_values=alpha_values,
    beta_values=beta_values,
    messages=messages,
    message_probabilities=[1/3, 1/3, 1/3, 0]
)

heatmap_social_welfare_beta0_k(
    payoffs=payoffs,
    T=5000,
    beta0_values=beta0_values,
    k_values=k_values,
    messages=messages,
    message_probabilities=[1/3, 1/3, 1/3, 0]
)

#######################


# Do these!


alpha_values = np.linspace(0, 1.0, 11)
beta_values = np.linspace(0, 1.0, 11)

message_probabilities = [1, 0, 0, 0]

heatmap_social_welfare_alpha_beta(
    payoffs=payoffs,
    T=5000,
    alpha_values=alpha_values,
    beta_values=beta_values,
    messages=messages,
    message_probabilities=[1, 0, 0, 0]
)

#####


alpha_values = np.linspace(0, 1.0, 11)
beta_values = np.linspace(0, 20.0, 100)

message_probabilities = [1/2, 1/4, 1/4, 0]
beta0_values = np.linspace(0, 1.0, 11)
k_values = np.linspace(0, 1.0, 11)
"""
correlated_eq, correlated_nash = heatmap_varying_beta0_k(
    payoffs=payoffs,  # Your payoff matrix
    T=500,
    messages=messages,  # Your messages
    message_probabilities=message_probabilities,  # Your probabilities
    beta0_values=beta0_values,
    k_values=k_values
)

save_to_pickle(correlated_eq, "correlated_eq3.pkl")
save_to_pickle(correlated_nash, "correlated_nash3.pkl")
"""
correlated_eq = load_from_pickle("correlated_eq3.pkl")
correlated_nash = load_from_pickle("correlated_nash3.pkl")

plot_heatmaps(correlated_eq, correlated_nash, beta0_values, k_values, "k", "beta0", extra = " [v.2]", filename = "heatmaps_varying_k [v.2].pdf")

########


alpha_values = np.linspace(0, 1.0, 11)
beta_values = np.linspace(0, 20.0, 100)
message_probabilities = [1/2, 1/4, 1/4, 0]

correlated_eq, correlated_nash = heatmap_varying_alpha_beta(
    payoffs=payoffs,  # Your payoff matrix
    T=500,
    messages=messages,  # Your messages
    message_probabilities=message_probabilities,  # Your probabilities
    alpha_values=alpha_values,
    beta_values=beta_values
)

save_to_pickle(correlated_eq, "correlated_eq12.pkl")
save_to_pickle(correlated_nash, "correlated_nash12.pkl")

correlated_eq = load_from_pickle("correlated_eq12.pkl")
correlated_nash = load_from_pickle("correlated_nash12.pkl")

plot_heatmaps(correlated_eq, correlated_nash, beta_values, alpha_values, "Beta", "Alpha",  extra = " [v.12]", filename="heatmaps_varying_alpha [v.12].pdf")

###################################################################################################


alpha_values = np.linspace(0, 1.0, 11)
beta_values = np.linspace(0, 1.0, 100)
message_probabilities = [1/2, 1/4, 1/4, 0]

correlated_eq, correlated_nash = heatmap_varying_alpha_beta(
    payoffs=payoffs,  # Your payoff matrix
    T=500,
    messages=messages,  # Your messages
    message_probabilities=message_probabilities,  # Your probabilities
    alpha_values=alpha_values,
    beta_values=beta_values
)

save_to_pickle(correlated_eq, "correlated_eq10.pkl")
save_to_pickle(correlated_nash, "correlated_nash10.pkl")

correlated_eq = load_from_pickle("correlated_eq10.pkl")
correlated_nash = load_from_pickle("correlated_nash10.pkl")

plot_heatmaps(correlated_eq, correlated_nash, beta_values, alpha_values, "Beta", "Alpha",  extra = " [v.10]", filename="heatmaps_varying_alpha [v.10].pdf")


