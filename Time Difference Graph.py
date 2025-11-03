#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Sat May 10 10:22:59 2025

@author: hiro

Here, I use the original algorithm and graph the time-difference graph. 

"""


import os
os.chdir('/Users/hiro/Documents/RL Project with Giordano')
os.listdir()

import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import Counter, defaultdict
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import random

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
    """
    Outputs:"avg_mixed_strategy","avg_social_welfare","action_history","strategies": strategies

    """
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
    start_time = time()
    correlated_eq_percentages, correlated_nash_percentages = simulate_alpha_beta(
        payoffs, T, messages, message_probabilities, alpha_values, beta_values, threshold
    )
    end_time = time()
    print(f"{end_time - start_time} seconds")

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


###################


T, alpha, beta0, k = 500, 0, 1, 0
message_probabilities = [1/3, 1/3, 1/3, 0]





Results = q_learning_2x2_game_with_pushforward(payoffs, T, alpha, beta0, k, messages, message_probabilities)
Results['strategies'][0]['m1']




a = np.vstack(Results['strategies'][0]['m1'])
b = np.vstack(Results['strategies'][0]['m2'])
c = np.vstack(Results['strategies'][1]['m1'])
d = np.vstack(Results['strategies'][1]['m1'])

a = a[:, 0]
b = b[:, 0]
c = c[:, 0]
d = d[:, 0]

########################################################################################################################



##############################################################################################################
def run_simulations_and_plot_by_convergence(payoffs, T, alpha, beta, k, messages, 
                                           message_probabilities, num_runs=100, threshold=0.995):
    """
    Run multiple Q-learning simulations, classify their convergence patterns,
    and create separate trajectory plots for each pattern.
    """
    # Storage for results organized by convergence pattern
    pattern_results = defaultdict(lambda: {
        'p0_m1': [], 'p0_m2': [], 'p1_m1': [], 'p1_m2': [], 'count': 0
    })
    
    # Define the equilibrium patterns of interest
    correlated_eq_pattern = (
        ('[1, 0]', '[1, 0]'),
        ('[1, 0]', '[0, 1]'),
        ('[0, 1]', '[1, 0]')
    )
    
    nash_eq_patterns = [
        (('[0, 1]', '[1, 0]'), ('[0, 1]', '[1, 0]'), ('[0, 1]', '[1, 0]')),
        (('[1, 0]', '[0, 1]'), ('[1, 0]', '[0, 1]'), ('[1, 0]', '[0, 1]'))
    ]
    
    pattern_names = {
        correlated_eq_pattern: "Correlated Equilibrium",
        nash_eq_patterns[0]: "Nash Equilibrium 1",
        nash_eq_patterns[1]: "Nash Equilibrium 2"
    }
    
    # Helper function to classify probability vectors
    def classify(probabilities, threshold):
        """Classify a probability vector as [1, 0] or [0, 1] if close enough."""
        if probabilities[0] >= threshold:
            return "[1, 0]"
        elif probabilities[1] >= threshold:
            return "[0, 1]"
        return "Other"
    
    # Run simulations and classify convergence patterns
    for run in range(num_runs):
        print(f"Running simulation {run+1}/{num_runs}...")
        
        # Run the simulation
        results = q_learning_2x2_game_with_pushforward(
            payoffs, T, alpha, beta, k, messages, message_probabilities
        )
        
        # Extract strategies
        a = np.vstack(results['strategies'][0]['m1'])[:, 0]
        b = np.vstack(results['strategies'][0]['m2'])[:, 0]
        c = np.vstack(results['strategies'][1]['m1'])[:, 0]
        d = np.vstack(results['strategies'][1]['m2'])[:, 0]
        
        # Find minimum length
        min_length = min(len(a), len(b), len(c), len(d))
        
        # Truncate all arrays to the minimum length
        a = a[:min_length]
        b = b[:min_length]
        c = c[:min_length]
        d = d[:min_length]
        
        # Classify the last mixed strategy
        combination_1 = (
            classify(results['last_mixed_strategy'][0]['m1'], threshold),
            classify(results['last_mixed_strategy'][1]['m1'], threshold)
        )
        combination_2 = (
            classify(results['last_mixed_strategy'][0]['m1'], threshold),
            classify(results['last_mixed_strategy'][1]['m2'], threshold)
        )
        combination_3 = (
            classify(results['last_mixed_strategy'][0]['m2'], threshold),
            classify(results['last_mixed_strategy'][1]['m1'], threshold)
        )
        
        # Create the pattern tuple
        pattern = (combination_1, combination_2, combination_3)
        
        # Determine which predefined pattern this matches (if any)
        matched_pattern = None
        if pattern == correlated_eq_pattern:
            matched_pattern = correlated_eq_pattern
        else:
            for nash_pattern in nash_eq_patterns:
                if pattern == nash_pattern:
                    matched_pattern = nash_pattern
                    break
        
        # If no match, classify as "Other"
        if matched_pattern is None:
            matched_pattern = "Other"
        
        # Store the results under the appropriate pattern
        pattern_results[matched_pattern]['p0_m1'].append(a)
        pattern_results[matched_pattern]['p0_m2'].append(b)
        pattern_results[matched_pattern]['p1_m1'].append(c)
        pattern_results[matched_pattern]['p1_m2'].append(d)
        pattern_results[matched_pattern]['count'] += 1
    
    # Print summary of patterns found
    print("\nConvergence Pattern Summary:")
    for pattern, data in pattern_results.items():
        if pattern != "Other":
            pattern_name = pattern_names.get(pattern, str(pattern))
            print(f"{pattern_name}: {data['count']} runs ({data['count']/num_runs*100:.1f}%)")
        else:
            print(f"Other patterns: {data['count']} runs ({data['count']/num_runs*100:.1f}%)")
    
    # For each convergence pattern, create trajectory plots
    for pattern, data in pattern_results.items():
        if data['count'] > 0:  # Only plot if we have data for this pattern
            if pattern != "Other":
                pattern_name = pattern_names.get(pattern, str(pattern))
                print(f"\nPlotting {pattern_name} trajectories ({data['count']} runs)...")
            else:
                print(f"\nPlotting other pattern trajectories ({data['count']} runs)...")
            
            plot_pattern_trajectories(data, pattern, alpha, beta, k, 
                                     pattern_names.get(pattern, "Other Patterns"))
    
    return pattern_results

def plot_pattern_trajectories(pattern_data, pattern, alpha, beta, k, pattern_name):
    """
    Plot strategy trajectories for a specific convergence pattern.
    """
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Define the four plot combinations
    plot_combinations = [
        (gs[0, 0], 'p0_m1', 'p1_m1', "Player 0's m1 vs Player 1's m1"),
        (gs[0, 1], 'p0_m1', 'p1_m2', "Player 0's m1 vs Player 1's m2"),
        (gs[1, 0], 'p0_m2', 'p1_m1', "Player 0's m2 vs Player 1's m1"),
        (gs[1, 1], 'p0_m2', 'p1_m2', "Player 0's m2 vs Player 1's m2")
    ]
    
    # Get number of runs for this pattern
    num_runs = pattern_data['count']
    
    # Create a colormap for different runs
    colors = cm.rainbow(np.linspace(0, 1, num_runs))
    
    # Create each subplot
    for pos, key_x, key_y, title in plot_combinations:
        ax = fig.add_subplot(pos)
        
        # Plot each run as a different colored trajectory
        for i in range(num_runs):
            x_data = pattern_data[key_x][i]
            y_data = pattern_data[key_y][i]
            
            # Plot the trajectory line
            ax.plot(x_data, y_data, color=colors[i], linewidth=0.5, alpha=0.2)
            
            # Mark the starting point (bold)
            ax.scatter(x_data[0], y_data[0], color=colors[i], s=40, marker='o', edgecolor='black', linewidth=0.5, alpha=0.6)
            ax.scatter(x_data[-1], y_data[-1], color=colors[i], s=50, marker='*', edgecolor='black', linewidth=0.5, alpha=0.7)
        
        # Set titles and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"{key_x.split('_')[0].upper()} {key_x.split('_')[1]} Strategy", fontsize=12)
        ax.set_ylabel(f"{key_y.split('_')[0].upper()} {key_y.split('_')[1]} Strategy", fontsize=12)
        
        # Set axes limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add grid and equal aspect ratio
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add reference diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Add overall title
    fig.suptitle(f"Strategy Trajectories for {pattern_name}\n(α={alpha}, β={beta}, k={k}, {num_runs} runs)", 
                fontsize=16)
    
    # Create a safe filename
    safe_pattern_name = pattern_name.replace(" ", "_").lower()
    filename = f"strategy_trajectories_pattern_{safe_pattern_name}_alpha_{alpha}_beta_{beta}_k_{k}.pdf"
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

# Example usage
alpha, beta, k = 0, 0.5, 0
message_probabilities = [1/3, 1/3, 1/3, 0]

pattern_results = run_simulations_and_plot_by_convergence(
    payoffs=payoffs,
    T=500,
    alpha=alpha,
    beta=beta,
    k=k,
    messages=messages,
    message_probabilities=message_probabilities,
    num_runs=100,  # Increased to get more samples of each pattern
    threshold=0.995
)


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



alpha_values = np.linspace(0, 1.0, 11)
beta_values = np.linspace(0, 20.0, 100)
#"""
correlated_eq, correlated_nash = heatmap_varying_alpha_beta(
    payoffs=payoffs,  # Your payoff matrix
    T=500,
    messages=messages,  # Your messages
    message_probabilities=message_probabilities,  # Your probabilities
    alpha_values=alpha_values,
    beta_values=beta_values
)

save_to_pickle(correlated_eq, "correlated_eq2.pkl")
save_to_pickle(correlated_nash, "correlated_nash2.pkl")
#"""
correlated_eq = load_from_pickle("correlated_eq2.pkl")
correlated_nash = load_from_pickle("correlated_nash2.pkl")

plot_heatmaps(correlated_eq, correlated_nash, beta_values, alpha_values, "Beta", "Alpha",  extra = " [v.1]", filename="heatmaps_varying_alpha [v.1].pdf")


######################################################################################################
# ---------- #
#This should be the last version



######################################################################################################

def run_simulations_and_plot_by_convergence(payoffs, T, alpha, beta, k, messages, 
                                           message_probabilities, num_runs=300, 
                                           threshold=0.995, max_runs_per_pattern=20):
    """
    Run multiple Q-learning simulations and create separate trajectory plots for each
    convergence pattern, with equal representation across patterns.
    """
    # Storage for results organized by convergence pattern
    pattern_results = defaultdict(lambda: {
        'p0_m1': [], 'p0_m2': [], 'p1_m1': [], 'p1_m2': [], 'count': 0
    })
    
    # Define the patterns of interest (separating each correlated equilibrium component)
    correlated_eq_components = [
        (('[1, 0]', '[1, 0]'), ('[1, 0]', '[0, 1]'), ('[0, 1]', '[1, 0]')),  # Full correlated
        (('[1, 0]', '[1, 0]'), 'Any', 'Any'),  # Component 1
        (('[1, 0]', '[0, 1]'), 'Any', 'Any'),  # Component 2
        (('[0, 1]', '[1, 0]'), 'Any', 'Any')   # Component 3
    ]
    
    nash_eq_patterns = [
        (('[0, 1]', '[1, 0]'), ('[0, 1]', '[1, 0]'), ('[0, 1]', '[1, 0]')),  # Nash 1
        (('[1, 0]', '[0, 1]'), ('[1, 0]', '[0, 1]'), ('[1, 0]', '[0, 1]'))   # Nash 2
    ]
    
    pattern_names = {
        correlated_eq_components[0]: "Complete Correlated Equilibrium",
        correlated_eq_components[1]: "Correlated Component (P0-m1, P1-m1)",
        correlated_eq_components[2]: "Correlated Component (P0-m1, P1-m2)",
        correlated_eq_components[3]: "Correlated Component (P0-m2, P1-m1)",
        nash_eq_patterns[0]: "Nash Equilibrium ([0,1], [1,0])",
        nash_eq_patterns[1]: "Nash Equilibrium ([1,0], [0,1])"
    }
    
    # Helper function to classify probability vectors
    def classify(probabilities, threshold):
        """Classify a probability vector as [1, 0] or [0, 1] if close enough."""
        if probabilities[0] >= threshold:
            return "[1, 0]"
        elif probabilities[1] >= threshold:
            return "[0, 1]"
        return "Other"
    
    # Run simulations until we have enough examples of each pattern
    runs_completed = 0
    max_runs = num_runs  # Safety cap to prevent infinite loops
    
    while runs_completed < max_runs:
        print(f"Running simulation {runs_completed+1}/{max_runs}...")
        
        # Run the simulation
        results = q_learning_2x2_game_with_pushforward(
            payoffs, T, alpha, beta, k, messages, message_probabilities
        )
        
        # Extract strategies
        a = np.vstack(results['strategies'][0]['m1'])[:, 0]
        b = np.vstack(results['strategies'][0]['m2'])[:, 0]
        c = np.vstack(results['strategies'][1]['m1'])[:, 0]
        d = np.vstack(results['strategies'][1]['m2'])[:, 0]
        
        # Find minimum length
        min_length = min(len(a), len(b), len(c), len(d))
        
        # Truncate all arrays to the minimum length
        a = a[:min_length]
        b = b[:min_length]
        c = c[:min_length]
        d = d[:min_length]
        
        # Classify the last mixed strategy
        combination_1 = (
            classify(results['last_mixed_strategy'][0]['m1'], threshold),
            classify(results['last_mixed_strategy'][1]['m1'], threshold)
        )
        combination_2 = (
            classify(results['last_mixed_strategy'][0]['m1'], threshold),
            classify(results['last_mixed_strategy'][1]['m2'], threshold)
        )
        combination_3 = (
            classify(results['last_mixed_strategy'][0]['m2'], threshold),
            classify(results['last_mixed_strategy'][1]['m1'], threshold)
        )
        
        # Create the pattern tuple
        full_pattern = (combination_1, combination_2, combination_3)
        
        # Determine which predefined patterns this matches
        matched_patterns = []
        
        # Check complete correlated equilibrium
        if full_pattern == correlated_eq_components[0]:
            matched_patterns.append(correlated_eq_components[0])
        
        # Check individual correlated components
        if combination_1 == correlated_eq_components[1][0]:
            matched_patterns.append(correlated_eq_components[1])
        if combination_2 == correlated_eq_components[2][0]:
            matched_patterns.append(correlated_eq_components[2])
        if combination_3 == correlated_eq_components[3][0]:
            matched_patterns.append(correlated_eq_components[3])
        
        # Check Nash equilibria
        for nash_pattern in nash_eq_patterns:
            if full_pattern == nash_pattern:
                matched_patterns.append(nash_pattern)
        
        # If no match, classify as "Other"
        if not matched_patterns:
            matched_patterns = ["Other"]
        
        # Store the results under all matched patterns, unless we already have enough examples
        for pattern in matched_patterns:
            if pattern_results[pattern]['count'] < max_runs_per_pattern:
                pattern_results[pattern]['p0_m1'].append(a)
                pattern_results[pattern]['p0_m2'].append(b)
                pattern_results[pattern]['p1_m1'].append(c)
                pattern_results[pattern]['p1_m2'].append(d)
                pattern_results[pattern]['count'] += 1
        
        runs_completed += 1
        
        # Check if we have reached the desired number of examples for each pattern
        if all(pattern_results[pattern]['count'] >= max_runs_per_pattern 
              for pattern in correlated_eq_components + nash_eq_patterns + ["Other"]):
            print(f"Collected sufficient examples of each pattern after {runs_completed} runs.")
            break
    
    # Print summary of patterns found
    print("\nConvergence Pattern Summary:")
    for pattern, data in pattern_results.items():
        if pattern != "Other":
            pattern_name = pattern_names.get(pattern, str(pattern))
            print(f"{pattern_name}: {data['count']} runs")
        else:
            print(f"Other patterns: {data['count']} runs")
    
    # For each convergence pattern, create trajectory plots (limiting to max_runs_per_pattern)
    for pattern, data in pattern_results.items():
        if data['count'] > 0:
            if pattern != "Other":
                pattern_name = pattern_names.get(pattern, str(pattern))
                print(f"\nPlotting {pattern_name} trajectories...")
            else:
                print(f"\nPlotting other pattern trajectories...")
            
            # Ensure we only use max_runs_per_pattern runs for each plot
            if data['count'] > max_runs_per_pattern:
                # Randomly select max_runs_per_pattern indices
                selected_indices = random.sample(range(data['count']), max_runs_per_pattern)
                plot_data = {
                    'p0_m1': [data['p0_m1'][i] for i in selected_indices],
                    'p0_m2': [data['p0_m2'][i] for i in selected_indices],
                    'p1_m1': [data['p1_m1'][i] for i in selected_indices],
                    'p1_m2': [data['p1_m2'][i] for i in selected_indices],
                    'count': max_runs_per_pattern
                }
            else:
                plot_data = data
            
            plot_pattern_trajectories(plot_data, pattern, alpha, beta, k, 
                                     pattern_names.get(pattern, "Other Patterns"))
    
    return pattern_results

def plot_pattern_trajectories(pattern_data, pattern, alpha, beta, k, pattern_name):
    """
    Plot strategy trajectories for a specific convergence pattern.
    """
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Define the four plot combinations
    plot_combinations = [
        (gs[0, 0], 'p0_m1', 'p1_m1', "Player 0's m1 vs Player 1's m1"),
        (gs[0, 1], 'p0_m1', 'p1_m2', "Player 0's m1 vs Player 1's m2"),
        (gs[1, 0], 'p0_m2', 'p1_m1', "Player 0's m2 vs Player 1's m1"),
        (gs[1, 1], 'p0_m2', 'p1_m2', "Player 0's m2 vs Player 1's m2")
    ]
    
    # Get number of runs for this pattern
    num_runs = pattern_data['count']
    
    # Create a colormap for different runs
    colors = cm.rainbow(np.linspace(0, 1, num_runs))
    
    # Create each subplot
    for pos, key_x, key_y, title in plot_combinations:
        ax = fig.add_subplot(pos)
        
        # Plot each run as a different colored trajectory
        for i in range(num_runs):
            x_data = pattern_data[key_x][i]
            y_data = pattern_data[key_y][i]
            
            # Plot the trajectory line (more subtle)
            ax.plot(x_data, y_data, color=colors[i], linewidth=0.8, alpha=0.4)
            
            # Mark intermediate points (smaller, more subtle)
            ax.scatter(x_data[1:-1], y_data[1:-1], color=colors[i], s=15, alpha=0.4)
            
            # Mark the starting point (pronounced)
            ax.scatter(x_data[0], y_data[0], color=colors[i], s=70, marker='o', edgecolor='black', linewidth=0.5)
            
            # Mark the ending point (pronounced)
            ax.scatter(x_data[-1], y_data[-1], color=colors[i], s=90, marker='*', edgecolor='black', linewidth=0.5)
        
        # Set titles and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"{key_x.split('_')[0].upper()} {key_x.split('_')[1]} Strategy", fontsize=12)
        ax.set_ylabel(f"{key_y.split('_')[0].upper()} {key_y.split('_')[1]} Strategy", fontsize=12)
        
        # Set axes limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add grid and equal aspect ratio
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add reference diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Add overall title
    fig.suptitle(f"Strategy Trajectories for {pattern_name}\n(α={alpha}, β={beta}, k={k}, {num_runs} runs)", 
                fontsize=16)
    
    # Create a safe filename
    safe_pattern_name = pattern_name.replace(" ", "_").replace("[", "").replace("]", "").replace(",", "").lower()
    filename = f"strategy_trajectories_{safe_pattern_name}_alpha_{alpha}_beta_{beta}_k_{k}.pdf"
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

# Example usage
alpha, beta, k = 0, 1, 0
message_probabilities = [1/3, 1/3, 1/3, 0]

pattern_results = run_simulations_and_plot_by_convergence(
    payoffs=payoffs,
    T=500,
    alpha=alpha,
    beta=beta,
    k=k,
    messages=messages,
    message_probabilities=message_probabilities,
    num_runs=1000,  # Increased to ensure finding enough examples of each pattern
    threshold=0.995,
    max_runs_per_pattern=10  # Equal representation per pattern
)


