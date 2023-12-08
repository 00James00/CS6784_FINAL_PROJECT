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
# A = -np.eye(3)  # Simple negative identity matrix 
# b = np.ones(3)  # Simple vector 
# delta = 0.5  # Set delta to 0.5 
# eta = 0.01  # Learning rate
# num_deployments = 50  # Number of model deployments for evaluation
# theta_initial = np.random.randn(3)  # Initial model parameters

# # Run the SPGD algorithm
# theta_values = SPGD(delta, A, b, eta, num_deployments, theta_initial)

# # Compute the optimal theta for evaluation
# theta_opt = -np.linalg.inv(2 * A) @ b
# optimal_loss = performative_loss(theta_opt, A @ theta_opt + b)
# final_loss = performative_loss(theta_values[-1], A @ theta_values[-1] + b)
# performance_fraction = final_loss / optimal_loss

# theta_values[-1], performance_fraction  # Output the final model parameters and performance fraction


# final_theta, performance_ratio = theta_values[-1], performance_fraction

# print ("The final model parameters ", final_theta)
# print ("The performance fraction, which compares the final loss to the optimal loss, is ", performance_ratio)

class SubPopulation:

    def __init__(self, mus, sigmas, fn):
        self.mus = mus
        self.sigmas = sigmas
        self.fn = fn

    def apply(self, n, theta):
        samples = self.sample(n)
        out = theta @ samples
        self.mus, self.sigmas = self.fn(out)

    def sample(self, n):
        samples = []
        for mu, sigma in zip(self.mus, self.sigmas):
            samples.append(np.random.normal(mu, sigma, n))
        samples = np.stack(samples)
        return samples

class MeanWorld:
    # the mean of each subpop evolves, new individuals are sampled
    def __init__(self, populations, n, weights):
        self.pops = populations
        self.n = n
        self.mus = []
        self.weights = weights

    def step(self, theta):
        for pop in self.pops:
            pop.apply(self.n * 10, theta)
        self.estimate_mu()

    def sample(self):
        samples = []
        for pop, w in zip(self.pops, self.weights):
            samples.append(pop.sample(self.n * w))
        samples = np.stack(samples)
        return samples
    
    def estimate_mu(self):
        samples = self.sample()
        self.mus.append(np.mean(samples, axis=0))

class StatefulPerfGD:

    def __init__(self, world, theta_init, H):
        self.world = world
        self.thetas = [theta_init]
        self.mu_hats = []
        self.H = H

    def estimate_mean(self, samples):
        mu_hat = np.mean(samples, axis=0)
        self.mu_hats.append(mu_hat)

    def deploy_sample(self, n):
        self.world.step(self.thetas[-1])
        samples = self.world.sample(n)
        self.estimate_mean(samples)

    def estimate_partials(self, psi, mu_hat, t):
        psis = []
        mu_hats = []
        for t in range(self.H, 0, -1):
            psis.append(np.concatenate(self.world.mus[-t], self.thetas[-t]))
            mu_hats.append(self.mu_hats[t])
        psi_H = np.stack(psis)
        mu_hats = np.stack(mu_hats)
        grad_psi = psi_H - psi          # subtract psi from each prev timestep psi
        grad_mu_hat = mu_hats - mu_hat
        adj = (grad_psi @ grad_mu_hat).H    # unsure on this line
        print(adj.shape)
        # extract estimates of first and second order derivative of m
        # UNSURE ABOUT SHAPES + DIMENSIONS
        half = -1
        d1, d2 = adj[:,:half], adj[:,half:]
        return d1, d2

    def estimateLTJac(self, d1, d2):
        # estimate derivative of long term mu w.r.t theta
        term1 = np.eye(d2.shape[0]) - d2
        dmu_star = np.linalg.pinv(term1) @ d1
        return dmu_star

    def estimateLTGrad(self, theta, mu_hat, dmu_star):
        pass

    def spgd(self, lr, max_steps, sample_size):
        converged = False
        while not converged:
            self.deploy_sample(sample_size)
            d1, d2 = self.estimate_partials()
            self.estimateLTJac(d1, d2)
            grad = self.estimateLTGrad()

            # estimation noise: optional
            noise = np.random.normal(loc=0, scale=1, size=grad.shape)
            theta = self.thetas[-1] - lr * (grad + noise)
            self.thetas.append(theta)
            converged = True        # replace with convergence condition


# example setup with 2 simple populations
# identical starting states
# pop2 has higher barrier of entry (see fn definitions)
fn1 = lambda x: x > 0
pop1 = SubPopulation([1], [1], fn1)
fn2 = lambda x: x > 5
pop2 = SubPopulation([1], [1], fn2)
world = MeanWorld([pop1, pop2], 10, [.5, .5])

# init simulation, Horizon = 1
theta_init = np.ones(1)
simulation = StatefulPerfGD(world, theta_init, 1)
simulation.spgd(.01, 5, 10)