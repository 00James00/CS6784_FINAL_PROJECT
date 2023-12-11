import numpy as np
import matplotlib.pyplot as plt

def linear_loss(z, theta):
    """Linear loss function for credit scoring."""
    return -np.dot(z, theta)

def mean_update(theta, mu, delta, A, b):
    """Stateful mean update function for the linear map."""
    return delta * (A @ theta + b) + (1 - delta) * mu

def performative_loss(theta, mu_star):
    """Performative loss given model parameters and stateful mean."""
    return -np.dot(mu_star, theta)

def behavior_change(z, theta):
    """Simulate strategic behavior change based on model understanding."""
    for i in range(len(z)):
        if theta[i] > 0: 
            z[i] += 0.01 * theta[i]  
        else:
            z[i] *= 0.99 * abs(theta[i])  
    return z

def SPGD_tracking_distribution(delta, A, b, eta, num_deployments, theta_initial):
    """SPGD algorithm with behavior change simulation and distribution tracking."""
    mu_previous = np.zeros(theta_initial.shape)
    theta = theta_initial.copy()
    theta_values = [theta.copy()]
    z = np.random.rand(3)  # Initialize random credit score features
    z_values = [z.copy()]  # Track the feature distribution

    for _ in range(num_deployments):
        z = behavior_change(z, theta)
        z_values.append(z.copy())
        mu_current = mean_update(theta, mu_previous, delta, A, b)
        grad_performative_loss = -mu_current
        theta -= eta * grad_performative_loss
        theta_values.append(theta.copy())
        mu_previous = mu_current

    return theta_values, z_values

# SPGD parameters
A = -np.eye(3)  
b = np.ones(3) 
delta = 0.5 
eta = 0.01  
num_deployments = 50  
theta_initial = np.random.randn(3)

#  tracking the distribution
theta_values, z_values = SPGD_tracking_distribution(delta, A, b, eta, num_deployments, theta_initial)

# Plotting the distribution shift over time
z_values = np.array(z_values)
time_steps = np.arange(num_deployments + 1)

plt.figure(figsize=(12, 6))
plt.plot(time_steps, z_values[:, 0], label='Payment History')
plt.plot(time_steps, z_values[:, 1], label='Credit Utilization')
plt.plot(time_steps, z_values[:, 2], label='Length of Credit History')
plt.xlabel('Time Step (Deployment)')
plt.ylabel('Feature Value')
plt.title('Distribution Shift of Credit Score Features Over Time')
plt.legend()
plt.show()
