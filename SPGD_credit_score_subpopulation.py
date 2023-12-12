import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

# simulation for gaming credit score
NUM_FEATURES = 3
PAYMENT_HISTORY_IDX = 0
CREDIT_UTILIZATION_IDX = 1
LENGTH_OF_CREDIT_HISTORY_IDX = 2

# starting point for both general population and sub-populations
theta_initial = np.array([0.1, 0.1, 0.1]) + np.random.randn(NUM_FEATURES) * 0.02

def linear_loss(z, theta):

    return -np.dot(z, theta)

def mean_update(theta, mu, delta, A, b):
   
    return delta * (A @ theta + b) + (1 - delta) * mu

def performative_loss(theta, mu_star):
    #Performative loss given model parameters and stateful mean.
    return -np.dot(mu_star, theta)

def behavior_change(z, theta, behavior_change_factor):
    #Simulate strategic behavior change based on model (payment history and credit utilization).

    for i in range(2):  
        z[i] += behavior_change_factor * (np.random.randn() * theta[i])
        z[i] = np.clip(z[i], 0, None) 
    return z

def SPGD_tracking_distribution(delta, A, b, eta, num_deployments, theta_initial, behavior_change_factor=1.0):
    #SPGD algorithm with behavior change simulation and distribution tracking.
    theta = theta_initial.copy()
    theta_values = [theta.copy()]
    z = np.array([0.5, 0.5, 0])  # Initialize random credit score features
    z_values = [z.copy()]  # Track the feature distribution

    for deployment in range(num_deployments):
        # Update theta using a gradient step
        mu_current = mean_update(theta, np.zeros(NUM_FEATURES), delta, A, b)
        grad_performative_loss = -mu_current
        theta -= eta * grad_performative_loss
        theta_values.append(theta.copy())
        
        z = behavior_change(z, theta, behavior_change_factor)
        z[LENGTH_OF_CREDIT_HISTORY_IDX] += 1 / num_deployments
        z_values.append(z.copy())

    return theta_values, z_values



# SPGD algorithm with sub-population behavior
def SPGD_subpops(delta, A, b, eta, num_deployments, theta_initial, sub_pops):
    theta = theta_initial.copy()
    theta_values = [theta.copy()]
    z_values_subpops = {name: [info['initial_z'].copy()] for name, info in sub_pops.items()}

    for deployment in range(num_deployments):
        mu_previous = np.zeros(theta_initial.shape)  # Reset 
        for name, info in sub_pops.items():
            z = z_values_subpops[name][-1].copy()
            # Apply behavior change based on the sub-population
            z = behavior_change(z, theta, info['behavior_change_factor'])
            z[LENGTH_OF_CREDIT_HISTORY_IDX] = deployment / num_deployments  
            z_values_subpops[name].append(z)
        
        mu_current = mean_update(theta, mu_previous, delta, A, b)
        grad_performative_loss = -mu_current
        theta -= eta * grad_performative_loss
        theta_values.append(theta.copy())

    return theta_values, z_values_subpops


# SPGD parameters
A = -np.eye(NUM_FEATURES)
b = np.ones(NUM_FEATURES) * 0.5  
delta = 0.1  #  mean reversion
eta = 0.1  # learning rate
num_deployments = 50


# General SPGD algorithm
theta_values_general, z_values_general = SPGD_tracking_distribution(
    delta, A, b, eta, num_deployments, theta_initial
)

# Define sub-populations with different initial conditions and behavior change rates
sub_pops = {
    'Subpop1': {'initial_z': np.array([0.5, 0.5, 0]), 'behavior_change_factor': 1.5},
    'Subpop2': {'initial_z': np.array([0.5, 0.5, 0]), 'behavior_change_factor': 0.5}
}
# SPGD algorithm with sub-populations
theta_values_subpops, z_values_subpops = SPGD_subpops(
    delta, A, b, eta, num_deployments, theta_initial, sub_pops
)


def plot_features(z_values, label_prefix):
    z_values = np.array(z_values)
    time_steps = np.arange(num_deployments + 1)
    plt.plot(time_steps, z_values[:, PAYMENT_HISTORY_IDX], label=f'Payment History ({label_prefix})')
    plt.plot(time_steps, z_values[:, CREDIT_UTILIZATION_IDX], label=f'Credit Utilization ({label_prefix})')
    plt.plot(time_steps, z_values[:, LENGTH_OF_CREDIT_HISTORY_IDX], label=f'Length of Credit History ({label_prefix})')

# Plot the distribution shift for the general population
plt.figure(figsize=(12, 6))
plot_features(z_values_general, "General Population")
plt.xlabel('Time Step (Deployment)')
plt.ylabel('Feature Value')
plt.title('Distribution Shift of Credit Score Features Over Time for General Population')
plt.legend()
plt.show()

# Plot the distribution shift for each sub-population
plt.figure(figsize=(14, 7))
for name, z_values in z_values_subpops.items():
    plot_features(z_values, name)
plt.xlabel('Time Step (Deployment)')
plt.ylabel('Feature Value')
plt.title('Distribution Shift of Credit Score Features Over Time by Sub-Population')
plt.legend()
plt.show()
