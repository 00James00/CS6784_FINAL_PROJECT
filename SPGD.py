import numpy as np



#This code simulates the SPGD algorithm over 50 deployments and calculates the performance fraction compared to the optimal performance.
#  The performance fraction informs how well the SPGD algorithm has approximated the optimal solution.​
# Define the linear point loss function as specified in the paper (linear map 5.2)
def linear_loss(z, theta):
    """Linear loss function."""
    return -np.dot(z, theta)

# Define the mean update function according to the linear map specification
def mean_update(theta, mu, delta, A, b):
    """Stateful mean update function for the linear map."""
    return delta * (A @ theta + b) + (1 - delta) * mu

# Define the long-term performative loss
def performative_loss(theta, mu_star):
    """Performative loss given the model parameters and stateful mean."""
    return -np.dot(mu_star, theta)

# Define the SPGD algorithm with linear map evaluation
def SPGD(delta, A, b, eta, num_deployments, theta_initial):
    """Stateful Performative Gradient Descent (SPGD) algorithm."""
    mu_previous = np.zeros(theta_initial.shape)  # Start with a zero vector for the stateful mean
    theta = theta_initial.copy()  # Copy of the initial parameters

    # Store the updated parameters for each deployment
    theta_values = [theta.copy()]

    for _ in range(num_deployments):
        # Update the stateful mean based on the current model parameters
        mu_current = mean_update(theta, mu_previous, delta, A, b)
        
        # Compute the gradient of the performative loss (gradient of linear loss w.r.t. theta)
        grad_performative_loss = -mu_current
        
        # Update the model parameters using the estimated gradient
        theta -= eta * grad_performative_loss
        
        # Store the updated parameters
        theta_values.append(theta.copy())
        
        # Update the previous mean
        mu_previous = mu_current

    return theta_values

# Parameters for the linear map and SPGD
A = -np.eye(3)  # Simple negative identity matrix 
b = np.ones(3)  # Simple vector 
delta = 0.5  # Set delta to 0.5 
eta = 0.01  # Learning rate
num_deployments = 50  # Number of model deployments for evaluation
theta_initial = np.random.randn(3)  # Initial model parameters

# Run the SPGD algorithm
theta_values = SPGD(delta, A, b, eta, num_deployments, theta_initial)

# Compute the optimal theta for evaluation
theta_opt = -np.linalg.inv(2 * A) @ b
optimal_loss = performative_loss(theta_opt, A @ theta_opt + b)
final_loss = performative_loss(theta_values[-1], A @ theta_values[-1] + b)
performance_fraction = final_loss / optimal_loss

theta_values[-1], performance_fraction  # Output the final model parameters and performance fraction


final_theta, performance_ratio = theta_values[-1], performance_fraction

print ("The final model parameters ", final_theta)
print ("The performance fraction, which compares the final loss to the optimal loss,is ", performance_ratio)