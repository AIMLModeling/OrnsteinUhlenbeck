import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.integrate import quad

np.random.seed(seed=42)

N = 20000  # time steps
paths = 5000  # number of paths
T = 5
T_vec, dt = np.linspace(0, T, N, retstep=True)

kappa = 3                # mean reversion coefficient
theta = 0.5              # long term mean
sigma = 0.5              # volatility coefficient
std_asy = np.sqrt(sigma**2 / (2 * kappa))  # asymptotic standard deviation

# Simulation of OU Paths
X0 = 2
X = np.zeros((N, paths))
X[0, :] = X0
W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
for t in range(0, N - 1):
    X[t + 1, :] = theta + np.exp(-kappa * dt) * (X[t, :] - theta) + std_dt * W[t, :]
# Analysis of the Simulated Data
X_T = X[-1, :]  # values of X at time T
X_1 = X[:, 1]  # a single path
mean_T = theta + np.exp(-kappa * T) * (X0 - theta)
std_T = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * T)))

param = ss.norm.fit(X_T)  # FIT from data
print(f"Theoretical mean={mean_T.round(6)} and theoretical STD={std_T.round(6)}")
print("Parameters from the fit: mean={0:.6f}, STD={1:.6f}".format(*param))  # these are MLE parameters
N_processes = 10  # number of processes
x = np.linspace(X_T.min(), X_T.max(), 100)
pdf_fitted = ss.norm.pdf(x, *param)

fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(T_vec, X[:, :N_processes], linewidth=0.5)
ax1.plot(T_vec, (theta + std_asy) * np.ones_like(T_vec), label="1 asymptotic std dev", color="black")
ax1.plot(T_vec, (theta - std_asy) * np.ones_like(T_vec), color="black")
ax1.plot(T_vec, theta * np.ones_like(T_vec), label="Long term mean")
ax1.legend(loc="upper right")
ax1.set_title(f"{N_processes} OU processes")
ax1.set_xlabel("T")
ax2.plot(x, pdf_fitted, color="r", label="Normal density")
ax2.hist(X_T, density=True, bins=50, facecolor="LightBlue", label="Frequency of X(T)")
ax2.legend()
ax2.set_title("Histogram vs Normal distribution")
ax2.set_xlabel("X(T)")
plt.show()
# Covariance Calculation
n1 = 5950
n2 = 6000
t1 = n1 * dt
t2 = n2 * dt

cov_th = sigma**2 / (2 * kappa) * (np.exp(-kappa * np.abs(t1 - t2)) - np.exp(-kappa * (t1 + t2)))
print(f"Theoretical COV[X(t1), X(t2)] = {cov_th.round(4)} with t1 = {t1.round(4)} and t2 = {t2.round(4)}")
print(f"Computed covariance from data COV[X(t1), X(t2)] = {np.cov( X[n1, :], X[n2, :] )[0,1].round(4)}")
# Estimation of parameters from a single path
XX = X_1[:-1]
YY = X_1[1:]
beta, alpha, _, _, _ = ss.linregress(XX, YY)  # OLS
kappa_ols = -np.log(beta) / dt
theta_ols = alpha / (1 - beta)
res = YY - beta * XX - alpha  # residuals
std_resid = np.std(res, ddof=2)
sig_ols = std_resid * np.sqrt(2 * kappa_ols / (1 - beta**2))

print("\n\nOLS theta = ", theta_ols)
print("Given theta = ", theta)
print("OLS kappa = ", kappa_ols)
print("Given kappa = ", kappa)
print("OLS sigma = ", sig_ols)
print("Given sigma = ", sigma)
Sx = np.sum(XX)
Sy = np.sum(YY)
Sxx = XX @ XX
Sxy = XX @ YY
Syy = YY @ YY

theta_mle = (Sy * Sxx - Sx * Sxy) / (N * (Sxx - Sxy) - (Sx**2 - Sx * Sy))
kappa_mle = -(1 / dt) * np.log(
    (Sxy - theta_mle * Sx - theta_mle * Sy + N * theta_mle**2) / (Sxx - 2 * theta_mle * Sx + N * theta_mle**2)
)
sigma2_hat = (
    Syy
    - 2 * np.exp(-kappa_mle * dt) * Sxy
    + np.exp(-2 * kappa_mle * dt) * Sxx
    - 2 * theta_mle * (1 - np.exp(-kappa_mle * dt)) * (Sy - np.exp(-kappa_mle * dt) * Sx)
    + N * theta_mle**2 * (1 - np.exp(-kappa_mle * dt)) ** 2
) / N
sigma_mle = np.sqrt(sigma2_hat * 2 * kappa_mle / (1 - np.exp(-2 * kappa_mle * dt)))
print("\n\ntheta MLE = ", theta_mle)
print("Given theta = ", theta)
print("kappa MLE = ", kappa_mle)
print("Given kappa = ", kappa)
print("sigma MLE = ", sigma_mle)
print("Given sigma = ", sigma)

# First Passage Time Analysis
T_to_theta = np.argmax(X <= theta if (X0 > theta) else X >= theta, axis=0) * dt  # first passage time
print(f"The expected time from X0 to theta is: {T_to_theta.mean()} with std error: {ss.sem(T_to_theta)}")
print("The standard deviation of the first time the process touches theta is: ", T_to_theta.std())
# The function `density_T_to_theta` computes the density function of 
# the first hitting time for the OU process to reach the long-term mean level theta. 
def density_T_to_theta(t, C):
    return (
        np.sqrt(2 / np.pi)
        * np.abs(C)
        * np.exp(-t)
        / (1 - np.exp(-2 * t)) ** (3 / 2)
        * np.exp(-((C**2) * np.exp(-2 * t)) / (2 * (1 - np.exp(-2 * t))))
    )
# The constant C represents the standardized initial deviation from the mean theta.
# By normalizing X0 in this manner, the first hitting time density can be expressed
# in a simpler form that is easier to handle analytically and numerically.
# This standardization allows the use of dimensionless variables
# and facilitates the derivation and computation of the first hitting time distribution
# for the OU process.
C = (X0 - theta) * np.sqrt(2 * kappa) / sigma  # new starting point
fig = plt.figure(figsize=(10, 4))
x = np.linspace(T_to_theta.min(), T_to_theta.max(), 100)
plt.plot(x, kappa * density_T_to_theta(kappa * x, C), color="red", label="OU hitting time density")
plt.hist(T_to_theta, density=True, bins=100, facecolor="LightBlue", label="frequencies of T")
plt.title("First passage time distribution from X0 to theta")
plt.legend()
plt.show()
theoretical_T = quad(lambda t: t * kappa * density_T_to_theta(kappa * t, C), 0, 1000)[0]
theoretical_std = np.sqrt(
    quad(lambda t: (t - theoretical_T) ** 2 * kappa * density_T_to_theta(kappa * t, C), 0, 1000)[0]
)
print("Theoretical expected hitting time: ", theoretical_T)
print("Theoretical standard deviation of the hitting time: ", theoretical_std)
