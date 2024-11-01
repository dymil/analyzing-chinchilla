# %%
import warnings
# from os import CLD_CONTINUED
from itertools import product
from functools import partial

import jax.numpy as jnp
from jax import grad
import jax
import pandas as pd
from scipy.optimize import curve_fit, basinhopping, OptimizeWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics._plot.confusion_matrix import unique_labels
import matplotlib.pyplot as plt
import numpy as np

np.seterr(over='ignore')
np.seterr(invalid='ignore')
warnings.filterwarnings("ignore", category=OptimizeWarning)

jax.config.update("jax_enable_x64", True)
x = jax.random.uniform(jax.random.key(0), (1000,), dtype=np.float64)
x.dtype # --> dtype('float64')

# %% [markdown]
# # Loading files

# %%
# Use pandas to read the CSV file directly from the URL
training_df = pd.read_csv('data/svg_extracted_data.csv')

# The DataFrame 'training_df' now contains the data from the CSV file

training_df['Training Tokens'] = training_df['Training FLOP']/(6.0*training_df['Model Size'])
training_df = training_df[['Model Size', 'Training Tokens', 'Training FLOP', 'loss']].dropna()
training_df

# %%
training_df.head(5)

# %% [markdown]
# # Plot of the data


# %%
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from matplotlib.transforms import Bbox

# Adjusting default font sizes for better readability
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Creating the scatter plot
fig, ax = plt.subplots(figsize=(12/1.2, 6/1.2))
sc = ax.scatter(training_df["Training FLOP"], training_df["Model Size"],
                c=training_df["loss"], cmap='magma', norm=LogNorm(vmin=1.804501, vmax=5.0), marker='o', edgecolors='black', linewidths=0.5)

# Adding a color bar on the right, but with ticks and text to the left of the bar
cbar = fig.colorbar(sc, ax=ax, location='right', pad=0.15)
cbar.set_label('Loss', rotation=90, labelpad=15)
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')

# Adjusting subplot parameters to remove whitespace
fig.subplots_adjust(left=0.1, right=0.8)  # Adjust the right margin here

# Removing the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Directly setting color bar tick labels to avoid scientific notation
cbar_ticks = [2.00, 3.00, 4.00, 5.00]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(['{:.2f}'.format(tick) for tick in cbar_ticks])

# Setting the plot labels and title
ax.set_xlabel('Training FLOP')
ax.set_ylabel('Model Size')

# Applying log scale to both axes
ax.set_xscale('log')
ax.set_yscale('log')

# Custom formatter for axes
ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))

# Setting axes limits
ax.set_xlim([min(training_df['Training FLOP'])*0.85, max(training_df['Training FLOP'])*1.15])
ax.set_ylim([min(training_df['Model Size'])*0.85, max(training_df['Model Size'])*1.15])

# Saving the plot to a PDF
# Get the current bounding box of the figure
fig_bbox = fig.get_tightbbox(fig.canvas.get_renderer())

# Create a new bounding box by modifying the original
# Increase left and right margins by specifying a shift
padding = 0.85 # Change the padding value as needed (in inches)
new_fig_bbox = Bbox.from_extents(
    fig_bbox.x0 - padding,  # Extend left by padding inches
    fig_bbox.y0,            # Keep bottom the same
    fig_bbox.x1 + padding+0.25,  # Extend right by padding inches
    fig_bbox.y1             # Keep top the same
)

# Save the figure using the new bounding box
plt.savefig('training_flop_vs_model_size.pdf', bbox_inches=new_fig_bbox)

plt.show()

# %%
# outlier datapoints
training_df['d_n_ratio'] = training_df['Training Tokens']/training_df['Model Size']

# %%
training_df

# %% [markdown]
# # Replicate methodology from Chinchilla paper
# 
# 

# %% [markdown]
# Define things

# %%
nr_of_models_excluded = 5

# %%
N = training_df['Model Size'].values
D = training_df['Training Tokens'].values
losses = training_df['loss'].values
bootstraps = 4000

sorted_losses = sorted(losses)
if nr_of_models_excluded == 0:
    indices = np.arange(0, N)
else:
    sorted_losses = sorted(losses)
    indices = np.array([i for i in range(len(N)) if losses[i] < sorted_losses[-nr_of_models_excluded]])

np.random.seed(42)
random_indices = [np.random.choice(indices, size=len(indices), replace=True) for _ in range(bootstraps)]

# %%
from itertools import product
from operator import itemgetter
from jax.scipy.stats import norm
from jax.scipy.special import erf, logsumexp
from scipy.optimize import minimize
from jax.scipy.optimize import minimize as jax_minimize
from optax import huber_loss as huber_optax

# true_params = np.array([np.log(406.4), np.log(410.7), np.log(1.69), 0.34, 0.28])
true_params = np.array([6.0073404, 6.0179186, 0.5267228, 0.33917084, 0.2849083])
true_params_rounded = np.array([np.log(406.4), np.log(410.7), np.log(1.69), 0.34, 0.28])

# Define the log-sum-exp function
def log_sum_exp(a, b, e, alpha, beta, N, D):
    """Axis"""
    return logsumexp(jnp.stack((a - alpha * jnp.log(N), b - beta * jnp.log(D), jnp.tile(e, N.shape[0]))), axis=0)
    return jnp.log(jnp.exp(a - alpha * jnp.log(N)) + jnp.exp(b - beta * jnp.log(D)) + jnp.exp(e))

def huber_normalizing_factor(delta=1e-3):
    return jnp.sqrt(2*jnp.pi) * (1 - 2*norm.sf(delta)) + 2 * jnp.exp(-0.5*delta**2)/delta

def huber_logpdf(x, delta=1e-3, loc=0, scale=1):
    x = (x-loc)/scale

    cond = jnp.abs(x) <= delta
    loss = jnp.where(cond, 0.5 * x**2, delta * (jnp.abs(x) - 0.5 * delta))
    return -loss - jnp.log(huber_normalizing_factor(delta=delta)) - jnp.log(scale)

def huber_pdf(x, delta=1e-3, loc=0, scale=1):
    return jnp.exp(huber_logpdf(x, delta=delta, loc=loc, scale=scale))

# Define the objective function to be minimized
@jax.jit
def objective(params, N, D, losses):
    a, b, e, alpha, beta, sigma = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return -jnp.sum(huber_logpdf(jnp.log(losses), loc=predictions, scale=jnp.exp(sigma), delta=1e-3))
    # return custom_huber_loss(jnp.log(losses), predictions, delta=1e-3)

@jax.jit
def scale_objective(sigma, params, N, D, losses):
    a, b, e, alpha, beta = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return -jnp.sum(huber_logpdf(jnp.log(losses), loc=predictions, scale=jnp.exp(sigma), delta=1e-3))
    # return custom_huber_loss(jnp.log(losses), predictions, delta=1e-3)

def constant_term_objective(params, a, b, alpha, beta, N, D, losses):
    e, sigma = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return -jnp.sum(huber_logpdf(jnp.log(losses), loc=predictions, scale=jnp.exp(sigma), delta=1e-3))

@partial(jax.jit, static_argnums=4)
def huber_loss_objective(params, N, D, losses, reduce_fn=jnp.sum):
    a, b, e, alpha, beta = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return reduce_fn(huber_optax(predictions, jnp.log(losses), delta=1e-3))
jac = jax.jit(grad(huber_loss_objective), static_argnums=4)

# Define the parameter untransform
def untransform_params(param_array):
    if len(jnp.shape(param_array)) == 2:
      return jnp.hstack((jnp.exp(param_array[:, :3]), param_array[:, 3:]))
    else:
      return jnp.hstack((jnp.exp(param_array[:3]), param_array[3:]))

# Define the Huber loss function on residuals
def huber_loss(residuals, delta=1e-3):
    return huber_optax(residuals, delta=delta)
    # Calculate the difference
    diff = residuals
    # Calculate the condition for Huber loss
    cond = jnp.abs(diff) <= delta
    # Apply Huber loss formula
    loss = jnp.where(cond, 0.5 * diff**2, delta * (jnp.abs(diff) - 0.5 * delta))
    return loss


jax_min = jax.jit(partial(jax_minimize, method="BFGS"), static_argnums=0)

def improve_min(acc, init_params, args):
    result = jax_min(huber_loss_objective, jnp.array(init_params), args)
    best_params = jnp.where(result.fun < acc[0], result.x, acc[1])
    best_loss = jnp.where(result.fun < acc[0], result.fun, acc[0])
    return (best_loss, best_params), None

@jax.jit
def jax_grid(alpha_vals, beta_vals, e_vals, a_vals, b_vals, args):
    """Tried a bunch of ways to do minimizations in parallel, doesn't work"""
    # a_s, b_s, e_s, alphas, betas = jnp.meshgrid(a_vals, b_vals, e_vals, alpha_vals, beta_vals)
    # bfgs = lambda alpha, beta, e, a, b: jax_min(huber_loss_objective, jnp.array([a, b, e, alpha, beta]), args).fun
    # for i in range(5):
    #     bfgs = jax.vmap(bfgs)
    # res = bfgs(alphas, betas, e_s, a_s, b_s)
    # # print(res)
    # i = jnp.unravel_index(res.argmin(), a_s.shape)
    # return res[i], jnp.array([a_s[i], b_s[i], e_s[i], alphas[i], betas[i]])

    best_loss = jnp.inf
    best_params = jnp.array(list(map(itemgetter(0), (a_vals, b_vals, e_vals, alpha_vals, beta_vals))))

    results_dict = {}
    # print(list(product(a_vals, b_vals, e_vals, alpha_vals, beta_vals)))
    # print(jax.lax.scan(partial(improve_min, args=args), (best_loss, best_params), jnp.array(list(product(a_vals, b_vals, e_vals, alpha_vals, beta_vals))))[0])
    return jax.lax.scan(partial(improve_min, args=args), (best_loss, best_params), jnp.array(list(product(a_vals, b_vals, e_vals, alpha_vals, beta_vals))))[0]
    for a, b, e, alpha, beta in (tuple(v) for v in product(a_vals, b_vals, e_vals, alpha_vals, beta_vals)):
        init_params = [a, b, e, alpha, beta]
        # result = minimize(huber_loss_objective, init_params, args=(N[indices], D[indices], losses[indices]), method='L-BFGS-B') #, jac=grad(huber_loss_objective)
        result = jax_min(huber_loss_objective, jnp.array(init_params), args)
        # results_dict[tuple(init_params)] = {'params': result.x, 'loss': result.fun}
        best_params = jnp.where(result.fun < best_loss, result.x, best_params)
        best_loss = jnp.where(result.fun < best_loss, result.fun, best_loss)
    return best_loss, best_params

# %% [markdown]
# ### Replicate exactly

# %%
N = np.array(training_df['Model Size'].values)
D = np.array(training_df['Training Tokens'].values)
losses = np.array(training_df['loss'].values)

# Set up the grid for initial parameter values
alpha_vals = np.arange(0, 2.5, 0.5)
# print(alpha_vals)
beta_vals = np.arange(0, 2.5, 0.5)
e_vals = np.arange(-1, 1.5, 0.5)
a_vals = np.arange(0, 30, 5)
b_vals = np.arange(0, 30, 5)
# alpha_vals = np.array([20, 10])
# beta_vals = np.array([20, 10])
# e_vals = np.array([-1.0, 1])
# a_vals = np.array([1.5, 0.5])
# b_vals = np.array([1.0, 20])

# Perform the optimization using L-BFGS over the grid of initial values
best_loss = np.inf
best_params = None
best_init = None

from itertools import product
results_dict = {}
for alpha, beta, e, a, b in (tuple(x.item() for x in v) for v in product(alpha_vals, beta_vals, e_vals, a_vals, b_vals)):
    init_params = [a, b, e, alpha, beta]
    # result = minimize(huber_loss_objective, init_params, args=(N[indices], D[indices], losses[indices]), method='BFGS', jac=jac)
    result = jax_min(huber_loss_objective, jnp.array(init_params), (N[indices], D[indices], losses[indices]))
    results_dict[tuple(init_params)] = {'params': result.x, 'loss': result.fun}
    if result.success and result.fun < best_loss:
        best_loss = result.fun
        best_params = result.x
        best_init = jnp.array(init_params)
        print(f"New best loss: {best_loss}")
        print(f"Best params: {best_params}")
        print(f"Initial guess: {init_params}")
        print(f"nfev: {result.nfev}")
# best_loss, best_params = jax_grid(alpha_vals, beta_vals, e_vals, a_vals, b_vals, (N[indices], D[indices], losses[indices]))

# Transform the fitted parameters a, b, e to A, B, E
if best_params is not None:
    A = np.exp(best_params[0])
    B = np.exp(best_params[1])
    E = np.exp(best_params[2])
    alpha = best_params[3]
    beta = best_params[4]
    print(f"Best loss: {best_loss}, best fit parameters: A={A}, B={B}, E={E}, alpha={alpha}, beta={beta}")
else:
    print("Optimization failed to converge.")

# %% [markdown]
# # Chinchilla parametric fit fits the data poorly

# %% [markdown]
# ### Our estimates

# %%
# Set up the grid for initial parameter values
param_list = []
param_list_grid = []

BEST_OF = 5
for num, indices in enumerate(random_indices):
    # Perform the optimization using BFGS
    best_loss = np.inf
    best_params = None

    init_params = true_params
    #   result = minimize(huber_loss_objective, init_params, args=(N[indices], D[indices], losses[indices]), \
    #                     jac=grad(huber_loss_objective), method='BFGS')
    result = jax_min(huber_loss_objective, init_params, args=(N[indices], D[indices], losses[indices]))
    param_list.append(result.x)
    for _ in range(BEST_OF):
        init_params = list(map(lambda vals: np.random.choice(vals), [a_vals, b_vals, e_vals, alpha_vals, beta_vals]))
        # result = minimize(huber_loss_objective, init_params, args=(N[indices], D[indices], losses[indices]), method='BFGS', jac=jac)
        result = jax_min(huber_loss_objective, jnp.array(init_params), (N[indices], D[indices], losses[indices]))
        results_dict[tuple(init_params)] = {'params': result.x, 'loss': result.fun}
    if result.success and result.fun < best_loss:
        best_loss = result.fun
        best_params = result.x
    if best_params is not None: # being generous
        param_list_grid.append(best_params)
    #print(f"New best loss: {best_loss}")
    #print(f"Best params: {best_params}")

    if num % 200 == 199:
        print("Bootstrap step %d completed" % (num+1))



param_list = np.array(param_list)
cov_matrix = np.cov(param_list.T)

param_list_grid = np.array(param_list_grid)
cov_matrix_grid = np.cov(np.transpose(param_list_grid))

# %%
import seaborn as sns

# Applying the given transformations
transformed_params = np.exp(param_list[:, :3])  # Apply exp() to the first three parameters
alpha_beta = param_list[:, 3:]  # The last two parameters remain unchanged
transformed_params = np.hstack([transformed_params, alpha_beta])  # Combine the transformed and untransformed parameters

transformed_params_grid = np.exp(param_list_grid[:, :3])  # Apply exp() to the first three parameters
transformed_params_grid = np.hstack([transformed_params_grid, param_list_grid[:, 3:]])  # Combine the transformed and untransformed parameters

# Creating a DataFrame for the transformed parameters
transformed_params_df = pd.DataFrame(
    transformed_params,
    columns=['A', 'B', 'E', r'$\alpha$', r'$\beta$']
)

transformed_params_grid_df = pd.DataFrame(
    transformed_params_grid,
    columns=['A', 'B', 'E', r'$\alpha$', r'$\beta$']
)

pd.DataFrame(transformed_params_grid, columns=['A', 'B', 'E', r'$\alpha$', r'$\beta$']).assign(setting='grid')
np.hstack([np.exp([A,B,E]), [alpha, beta]])
big_df = pd.concat([transformed_params_grid_df.assign(setting=f'grid best-of-{BEST_OF}'),
                                   transformed_params_df.assign(setting='og'),
                                #    pd.DataFrame(transformed_params_grid, columns=['A', 'B', 'E', r'$\alpha$', r'$\beta$']).assign(setting='grid'),
                                   pd.DataFrame(np.array([[A,B,E,alpha, beta]]), columns=['A', 'B', 'E', r'$\alpha$', r'$\beta$']).assign(setting='best'),
                                   pd.DataFrame(np.hstack([np.exp(true_params[:3]), true_params[3:]]).reshape(1, -1), columns=['A', 'B', 'E', r'$\alpha$', r'$\beta$']).assign(setting='true'),#])
                                   ])

# Plotting the distribution of each transformed parameter across bootstraps
grid = sns.pairplot(big_df, diag_kind='kde', corner=True, hue='setting')#, kind='kde')
print(true_params)
print(pd.DataFrame(np.hstack([np.exp(true_params[:3]), true_params[3:]]).reshape(1, -1), columns=['A', 'B', 'E', r'$\alpha$', r'$\beta$']))
plt.suptitle('Distribution of Transformed Parameters Across Bootstraps', y=1.02)
plt.show()

# %%
param_list = np.array(param_list)
cov_matrix = np.cov(param_list.T)

param_list_untransformed = untransform_params(param_list)
cov_matrix_untransformed = np.cov(np.transpose(param_list_untransformed))

# %%
init_params = np.concatenate((true_params, np.array([0])))

indices = np.arange(len(N)) if nr_of_models_excluded == 0 else np.array([i for i in range(len(N)) if losses[i] < sorted(losses)[-nr_of_models_excluded]])

result = minimize(objective, init_params, args=(N[indices], D[indices], losses[indices]), method='BFGS',
                  jac=jax.jit(grad(objective)))
# result = jax_min(objective, init_params, args=(N[indices], D[indices], losses[indices]))

print(result)
print(result.x)

estimated_params = result.x[:5]
estimated_params_untransformed = untransform_params(estimated_params)

# %%
standard_errors = np.sqrt(np.diag(cov_matrix[:5, :5]))
standard_errors_untransformed = np.sqrt(np.diag(cov_matrix_untransformed[:5, :5]))

parameter_labels = ["A", "B", "E", "alpha", "beta"]
print("Parameter estimates and their standard errors\n")
for index, label in enumerate(parameter_labels):
  print("%s: %.5f (%.5f)" % (label, estimated_params_untransformed[index], standard_errors_untransformed[index]))

# %%
# Calculating 95% Confidence Intervals for each parameter
confidence_intervals = {}

# For each column in the DataFrame, calculate the 2.5th and 97.5th percentiles
for column in transformed_params_df.columns:
    lower_bound = np.percentile(transformed_params_df[column], 2.5)
    upper_bound = np.percentile(transformed_params_df[column], 97.5)
    confidence_intervals[column] = (lower_bound, upper_bound)

# Printing out the 95% Confidence Intervals for each parameter
print("95% Confidence Intervals for Parameter Estimates\n")
for parameter, (lower, upper) in confidence_intervals.items():
    print(f"{parameter}: ({lower:.3f}, {upper:.3f})")

# %%
true_params_unlogged = np.array([np.exp(6.0073404), np.exp(6.0179186), np.exp(0.5267228), 0.33917084, 0.2849083])
true_params_rounded_unlogged = np.array([406.4, 410.7, 1.69, 0.34, 0.28])

# %%
from scipy.stats import t

# Calculate t-statistics
t_statistics = (estimated_params_untransformed - true_params_unlogged) / standard_errors_untransformed

# Degrees of freedom
degrees_of_freedom = len(indices) -5

# Calculate two-tailed p-values
p_values = t.sf(np.abs(t_statistics), degrees_of_freedom) * 2  # times 2 for two-tailed test

# Print parameter names alongside p-values
for label, p_value in zip(parameter_labels, p_values):
    print(f"{label}: P-value = {p_value:.1e}")

# %% [markdown]
# ### Chi squared test

# %%
transformed_params_df

# %% [markdown]
# Chi squared for equality of all parameters

# %%
best_params

# %%
#baseline_fitted_params = np.array([  6.17795598,   7.64273236,   0.59711196,   0.34781303,   0.36585412  ])

baseline_fitted_params = np.array([  6.17795598,   7.64273236,   0.59711196,   0.34781303,   0.36585412  ])

#fitted_params_with_outliers = np.array([6.13834459, 9.43584213, 0.63412782, 0.3453568 , 0.45189211])

# %%
from scipy.stats import chi2

def chi_squared_stat(params_1, params_2, cov_matrix):
  return np.transpose(params_1 - params_2) @ np.linalg.inv(cov_matrix) @ (params_1 - params_2)

print("Difference between Hoffmann et al. (2022) params and our params:", true_params - baseline_fitted_params)
chi_squared = chi_squared_stat(true_params, estimated_params, cov_matrix[:5, :5])

print("Implied chi^2 (df=5) test statistic: %.2f" % (chi_squared))
print("Implied chi^2 (df=5) p-value: %.2e\n" % (chi2.sf(chi_squared, df=5)))

print("Difference between our default (excluding five outliers) params and our params without any outlier exclusion:", baseline_fitted_params - estimated_params)
chi_squared = chi_squared_stat(baseline_fitted_params, estimated_params, cov_matrix[:5, :5])

print("Implied chi^2 (df=5) test statistic: %.2f" % (chi_squared))
print("Implied chi^2 (df=5) p-value: %.2e" % (chi2.sf(chi_squared, df=5)))

# %% [markdown]
# # Our fit is better

# %% [markdown]
# ## Plots

# %%
def scaling_law(N, D, params):
    logA, logB, logE, alpha, beta = params
    A, B, E = np.exp(np.array([logA, logB, logE]))
    return E + A/N**alpha + B/D**beta

# Your residuals calculation
A, B, E, alpha, beta = true_params_unlogged
residuals = losses[indices] - scaling_law(N[indices], D[indices], true_params_rounded)  # Hoffmann estimates
mse = np.mean(residuals**2)

residuals_ours = losses[indices] - np.exp(log_sum_exp(*estimated_params, N[indices], D[indices]))
mse_ours = np.mean(residuals_ours**2)

# %%
from matplotlib.path import Path

# Assuming residuals_2 and residuals_2_ours are numpy arrays with your data
# Define the data

color_ours = 0.35
color_chinchilla = 0.65

data_control = residuals  # replace with your actual data for population-matched controls
data_active = residuals_ours  # replace with your actual data for musically active cases

# Create a new figure
fig, ax = plt.subplots(figsize=(6, 4))

# Check for the largest residual and set the y-limit if necessary
largest_residual = max(np.max(data_control), np.max(data_active))
if largest_residual > 0.6:
    ax.set_ylim(-0.15, 0.2)

# Scatter plots with a small random noise to x-values for jitter effect
scatter_control = np.random.normal(1, 0.04, size=len(data_control))
scatter_active = np.random.normal(2, 0.04, size=len(data_active))
ax.scatter(scatter_control, data_control, alpha=0.25, color=plt.cm.viridis(color_chinchilla))
ax.scatter(scatter_active, data_active, alpha=0.25, color=plt.cm.viridis(color_ours))

# Offset for the violin plots to be right of the scatter points
violin_offset = 0.1  # Adjust as needed

# Create violin plots on the same axis as scatter plots, slightly offset to the right
violin_parts = ax.violinplot([data_control, data_active], positions=[1 + violin_offset, 2 + violin_offset],
                             widths=0.8, showmeans=False, showextrema=False, showmedians=True)

# Make the violin plot one-sided by adjusting its paths
colors = [plt.cm.viridis(color_chinchilla), plt.cm.viridis(color_ours)]  # Use the same colors as scatter plots
for pc, color in zip(violin_parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
    m = np.mean(pc.get_paths()[0].vertices[:, 0])  # Find the center of the violin
    pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)  # Clip to the right

# Calculate means and confidence intervals for both groups
mean_control = np.mean(data_control)
mean_active = np.mean(data_active)
ci_control = np.std(data_control) * 1.96 / np.sqrt(len(data_control))
ci_active = np.std(data_active) * 1.96 / np.sqrt(len(data_active))

# Set the x-tick labels
ax.set_xticks([1, 2])
ax.set_xticklabels(['Hoffmann et al.', 'Ours'])

# Set y-label
ax.set_ylabel('Residuals')

# Add grid to the plot
ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a horizontal line at y=0
ax.axhline(y=0, color='black',  linestyle='--', linewidth=0.8)

# Set layout to be tight
plt.tight_layout()

plt.savefig("distributions_1.pdf")
#plt.savefig("distributions2.pdf")

# Show plot
plt.show()

# %%
# Your residuals calculation
A, B, E, alpha, beta = true_params_unlogged
residuals = losses[indices] - scaling_law(N[indices], D[indices], true_params)  # Hoffmann estimates
mse = np.mean(residuals**2)

residuals_ours = losses[indices] - np.exp(log_sum_exp(*estimated_params, N[indices], D[indices]))
mse_ours = np.mean(residuals_ours**2)

from scipy import stats

t_statistic, p_value = stats.ttest_ind(residuals, residuals_ours, equal_var=True)

print(f"T-statistic: {t_statistic}, P-value: {p_value}")

# %%
# Assuming residuals_2 and residuals_2_ours are numpy arrays with your data
# Define the data

color_ours = 0.35
color_chinchilla = 0.65

data_control = residuals  # replace with your actual data for population-matched controls
data_active = residuals_ours  # replace with your actual data for musically active cases

# Create a new figure
fig, ax = plt.subplots(figsize=(6, 4))

# Check for the largest residual and set the y-limit if necessary
largest_residual = max(np.max(data_control), np.max(data_active))
if largest_residual > 0.6:
    ax.set_ylim(-0.15, 0.2)

# Scatter plots with a small random noise to x-values for jitter effect
scatter_control = np.random.normal(1, 0.04, size=len(data_control))
scatter_active = np.random.normal(2, 0.04, size=len(data_active))
ax.scatter(scatter_control, data_control, alpha=0.25, color=plt.cm.viridis(color_chinchilla))
ax.scatter(scatter_active, data_active, alpha=0.25, color=plt.cm.viridis(color_ours))

# Offset for the violin plots to be right of the scatter points
violin_offset = 0.1  # Adjust as needed

# Create violin plots on the same axis as scatter plots, slightly offset to the right
violin_parts = ax.violinplot([data_control, data_active], positions=[1 + violin_offset, 2 + violin_offset],
                             widths=0.8, showmeans=False, showextrema=False, showmedians=True)

# Make the violin plot one-sided by adjusting its paths
colors = [plt.cm.viridis(color_chinchilla), plt.cm.viridis(color_ours)]  # Use the same colors as scatter plots
for pc, color in zip(violin_parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)
    m = np.mean(pc.get_paths()[0].vertices[:, 0])  # Find the center of the violin
    pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, jnp.inf)  # Clip to the right

# Calculate means and confidence intervals for both groups
mean_control = np.mean(data_control)
mean_active = np.mean(data_active)
ci_control = np.std(data_control) * 1.96 / np.sqrt(len(data_control))
ci_active = np.std(data_active) * 1.96 / np.sqrt(len(data_active))

# Set the x-tick labels
ax.set_xticks([1, 2])
ax.set_xticklabels(['Hoffmann et al.', 'Ours'])

# Set y-label
ax.set_ylabel('Residuals')

# Add grid to the plot
ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a horizontal line at y=0
ax.axhline(y=0, color='black',  linestyle='--', linewidth=0.8)

# Set layout to be tight
plt.tight_layout()

plt.savefig("distributions_2.pdf")
#plt.savefig("distributions2.pdf")

# Show plot
plt.show()

# %% [markdown]
# ## Tests

# %% [markdown]
# ## Likelihood test

# %%
param_list = np.array(param_list)
cov_matrix = np.cov(np.transpose(param_list))

param_list_untransformed = untransform_params(param_list)
cov_matrix_untransformed = np.cov(np.transpose(param_list_untransformed))

init_params = list(true_params) + [0]
indices = np.array([i for i in range(len(N)) if losses[i] < sorted(losses)[-5]])

# estimate parameters
result = minimize(objective, init_params, args=(N[indices], D[indices], losses[indices]), method='BFGS', jac=jax.jit(grad(objective)), tol=1e-1)

estimated_params = result.x[:5]
estimated_params_untransformed = untransform_params(estimated_params)

indices_with_outliers = [i for i in range(len(N))]

# fit scale param for estimated (without outliers)
result = minimize(scale_objective, init_params[-1], args=(estimated_params, N[indices], D[indices], losses[indices]), method='BFGS', jac=jax.jit(grad(scale_objective)), tol=1e-1)
estimated_params_scale_adjusted_no_outliers = list(estimated_params) + [result.x[0]]

# fit scale param for estimated (with outliers)
result = minimize(scale_objective, init_params[-1], args=(estimated_params, N[indices_with_outliers], D[indices_with_outliers], losses[indices_with_outliers]), method='BFGS', jac=jax.jit(grad(scale_objective)), tol=1e-1)
estimated_params_scale_adjusted_with_outliers = list(estimated_params) + [result.x[0]]

# %%
# fit scale param for rounded Chinchilla parameters (without outliers)
result = minimize(scale_objective, init_params[-1], args=(true_params_rounded, N[indices], D[indices], losses[indices]), \
                  method='BFGS', jac=grad(scale_objective), tol=1e-9)

rounded_chinchilla_params_with_scale_no_outliers = list(true_params_rounded) + [result.x[0]]


# fit scale param for rounded Chinchilla parameters (with outliers)
result = minimize(scale_objective, init_params[-1], args=(true_params_rounded, N[indices_with_outliers], D[indices_with_outliers], losses[indices_with_outliers]), \
                  method='BFGS', jac=grad(scale_objective), tol=1e-9)

rounded_chinchilla_params_with_scale_with_outliers = list(true_params_rounded) + [result.x[0]]


# fit scale param for unrounded Chinchilla parameters (with outliers)
result = minimize(scale_objective, init_params[-1], args=(true_params, N[indices_with_outliers], D[indices_with_outliers], losses[indices_with_outliers]), method='BFGS', jac=grad(scale_objective), tol=1e-1)

unrounded_chinchilla_params_with_scale_with_outliers = list(true_params) + [result.x[0]]


# fit scale param for unrounded Chinchilla parameters (with outliers)
result = minimize(scale_objective, init_params[-1], args=(true_params, N[indices], D[indices], losses[indices]), method='BFGS', jac=grad(scale_objective), tol=1e-1)

unrounded_chinchilla_params_with_scale_no_outliers = list(true_params) + [result.x[0]]

# %%
# Print the arrays
print("estimated_params_scale_adjusted_no_outliers:")
print(estimated_params_scale_adjusted_no_outliers)
print()

print("estimated_params_scale_adjusted_with_outliers:")
print(estimated_params_scale_adjusted_with_outliers)
print()

print("rounded_chinchilla_params_with_scale_no_outliers:")
print(rounded_chinchilla_params_with_scale_no_outliers)
print()

print("rounded_chinchilla_params_with_scale_with_outliers:")
print(rounded_chinchilla_params_with_scale_with_outliers)
print()

print("unrounded_chinchilla_params_with_scale_with_outliers:")
print(unrounded_chinchilla_params_with_scale_with_outliers)
print()

print("unrounded_chinchilla_params_with_scale_no_outliers:")
print(unrounded_chinchilla_params_with_scale_no_outliers)

# %% [markdown]
# Log likelihoods (left column)

# %%
log_likelihood_chinchilla_rounding_no_outliers = -objective(rounded_chinchilla_params_with_scale_no_outliers, N[indices], D[indices], losses[indices])

print(f"Likelihood ratios (Chichila, rounded, no outliers): {log_likelihood_chinchilla_rounding_no_outliers:.2f}")

log_likelihood_chinchilla_no_outliers = -objective(unrounded_chinchilla_params_with_scale_no_outliers, N[indices], D[indices], losses[indices])

print(f"Likelihood ratios (Chichila, unrounded, no outliers): {log_likelihood_chinchilla_no_outliers:.2f}")

log_likelihood_our_best_fit_no_outliers = -objective(estimated_params_scale_adjusted_no_outliers, N[indices], D[indices], losses[indices])

print(f"Likelihood ratios (our fit, no outliers): {log_likelihood_our_best_fit_no_outliers:.2f}")

# %%
lambda_LR = -2*(log_likelihood_chinchilla_no_outliers - log_likelihood_our_best_fit_no_outliers)
lr_test_df = 5 # 6 parameters fitted for best fit - 1 parameter (scale) fit for chinchilla = 5 degrees of freedom

lr_test_p_value = chi2.sf(lambda_LR, df=lr_test_df)
print("Likelihood ratio test statistic: %.2f\nWilks distribution (chi^2 with df=%d) p-value: %.2e" % (lambda_LR, lr_test_df, lr_test_p_value))

# %% [markdown]
# Log likelihoods (right column)

# %%
log_likelihood_chinchilla_rounding_with_outliers = -objective(rounded_chinchilla_params_with_scale_with_outliers, N[indices_with_outliers], D[indices_with_outliers], losses[indices_with_outliers])

print(f"Likelihood ratios (Chichila, rounded, no outliers): {log_likelihood_chinchilla_rounding_with_outliers:.2f}")

log_likelihood_chinchilla_with_outliers = -objective(unrounded_chinchilla_params_with_scale_with_outliers, N[indices_with_outliers], D[indices_with_outliers], losses[indices_with_outliers])

print(f"Likelihood ratios (Chichila, unrounded, no outliers): {log_likelihood_chinchilla_with_outliers:.2f}")

log_likelihood_our_best_fit_with_outliers = -objective(estimated_params_scale_adjusted_with_outliers, N[indices_with_outliers], D[indices_with_outliers], losses[indices_with_outliers])

print(f"Likelihood ratios (our fit, no outliers): {log_likelihood_our_best_fit_with_outliers:.2f}")

# %%
lambda_LR = -2*(log_likelihood_chinchilla_with_outliers - log_likelihood_our_best_fit_with_outliers)
lr_test_df = 5 # 6 parameters fitted for best fit - 1 parameter (scale) fit for chinchilla = 5 degrees of freedom

lr_test_p_value = chi2.sf(lambda_LR, df=lr_test_df)
print("Likelihood ratio test statistic: %.2f\nWilks distribution (chi^2 with df=%d) p-value: %.2e" % (lambda_LR, lr_test_df, lr_test_p_value))

# %% [markdown]
# Estimates and standard errors

# %%
a_low = 0.454
a_high = 0.455
a_mid = np.mean(np.array([a_low, a_high]))

estimated_params_with_outliers = np.array([6.28204169, 9.51269708, 0.63748901, 0.35286066, 0.45596155])
#estimated_params = np.median(param_list, axis=0)
standard_errors = np.sqrt(np.diag(cov_matrix[:5, :5]))

parameter_labels = ["A", "B", "E", "alpha", "beta"]
print("Parameter estimates and their standard errors\n")
for index, label in enumerate(parameter_labels):
  print("%s: %.3f (%.3f)" % (label, estimated_params[index], standard_errors[index]))

# %%
from scipy.stats import chi2

def chi_squared_stat(params_1, params_2, cov_matrix):
  return np.transpose(params_1 - params_2) @ np.linalg.inv(cov_matrix) @ (params_1 - params_2)

print("Difference between Hoffmann et al. (2022) params and our params:", true_params - estimated_params)
chi_squared = chi_squared_stat(true_params, estimated_params, cov_matrix[:5, :5])

print("Implied chi^2 (df=5) test statistic: %.2f" % (chi_squared))
print("Implied chi^2 (df=5) p-value: %.2e" % (chi2.sf(chi_squared, df=5)))

# %% [markdown]
# # How much data is needed to get confidence bands around a and b that are as tight as Hoffmann et al. report

# %%
from scipy.stats import norm

# Extract alpha and beta from the parameter list
alpha_samples = param_list[:, -2]
beta_samples = param_list[:, -1]

# Calculate the mean values of alpha and beta
mean_alpha = np.mean(alpha_samples)
mean_beta = np.mean(beta_samples)

# Extract the variances and covariance for alpha and beta
var_alpha = cov_matrix_untransformed[-2, -2]
var_beta = cov_matrix_untransformed[-1, -1]
cov_alpha_beta = cov_matrix_untransformed[-2, -1]

# Calculate the partial derivatives of g(alpha, beta) = alpha / (alpha + beta)
# with respect to alpha and beta, evaluated at the mean values of alpha and beta
partial_g_alpha = -mean_beta / (mean_alpha + mean_beta)**2
partial_g_beta = mean_alpha / (mean_alpha + mean_beta)**2

# Calculate the variance of the ratio using the delta method
var_ratio = (partial_g_alpha**2 * var_alpha +
             partial_g_beta**2 * var_beta +
             2 * partial_g_alpha * partial_g_beta * cov_alpha_beta)

# Calculate the standard error of the ratio
se_ratio = np.sqrt(var_ratio)

# Calculate the width of the 80% confidence interval
width_of_80_ci_band_a = 2 * norm.ppf(0.9) * se_ratio

# Assuming you want to maintain a fixed standard error (se_ratio) for a different sample size
# and you have a desired width for the confidence interval, calculate the required sample size
desired_width = 0.001

existing_sample_size = len(N[indices])
multiple_by_which_n_needs_to_increase = (width_of_80_ci_band_a / desired_width)**2
required_n = existing_sample_size*multiple_by_which_n_needs_to_increase

print(required_n)

# %% [markdown]
# Confirm with bootstrap

# %%
#estimated_params = np.median(param_list, axis=0)
standard_errors_untransformed = np.sqrt(np.diag(cov_matrix_untransformed[:5, :5]))

b = param_list[:, -2]/(param_list[:, -2] + param_list[:, -1])
a = 1-b

b_point_estimate = estimated_params[-2]/(estimated_params[-2] + estimated_params[-1])
a_point_estimate = 1 - b_point_estimate

parameter_labels = ["A", "B", "E", "alpha", "beta"]
print("Parameter estimates and their standard errors\n")
for index, label in enumerate(parameter_labels):
  print("%s: %.3f (%.3f)" % (label, estimated_params_untransformed[index], standard_errors_untransformed[index]))

print("a = beta/(alpha_beta): %.3f (%.3f)" % (a_point_estimate, np.std(a)))
chinchilla_conf_int_width = desired_width

a_std_err = np.std(a)
a_conf_int_width = np.percentile(a, 90) - np.percentile(a, 10)
chinchilla_a_conf_int_width = 1e-3
our_sample_size = len(N[indices])
required_sample_size = our_sample_size * (a_conf_int_width/chinchilla_a_conf_int_width)**2

print("""Our sample size is %d, and a has a standard error of %.3f
      and a 80%% conf int width of %.3f at this sample size""" % (our_sample_size, a_std_err, a_conf_int_width))

print("To reach 80%% conf int width of %.3f, we would need a sample size of %d" % (chinchilla_conf_int_width, required_sample_size))

# %% [markdown]
# ### What if they used intermediate losses and clustered standard errors?

# %%
import math

def calculate_new_se(rho, se_original=0.02, G=500, N=500000): #suppose they have 1k loss values per training run
    """
    Calculate the new standard error using all observations and accounting for clustering.
    """
    n = N / G  # Number of observations per group
    effective_N = N / (1 + (n - 1) * rho)  # Effective number of independent observations
    new_se = se_original * math.sqrt(G / effective_N)
    return new_se

# Define rho values from 0.05 to 0.5
rho_values = np.linspace(0.25, 0.95, 100)
new_se_values = [calculate_new_se(rho) for rho in rho_values]
confidence_interval_width = [se * 1.282 * 2 for se in new_se_values]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(rho_values, confidence_interval_width, marker='o', linestyle='-', color='b')
plt.title('Standard Errors vs. Intra-group Correlation Coefficient (rho)')
plt.xlabel('Intra-group Correlation Coefficient (rho)')
plt.ylabel('80% CI interval width')
plt.grid(True)
plt.show()

# %% [markdown]
# # Comparing optimal scaling policies

# %%
from scipy.stats import multivariate_normal

@jax.jit
def scaling_law_reducible(N, D, params):
  a, b, e, alpha, beta = params
  A, B, E = jnp.exp(jnp.array([a, b, e]))

  return A/N**alpha + B/D**beta

def G(params):
  a, b, e, alpha, beta = params
  A, B, E = np.exp(np.array([a, b, e]))

  return ((alpha*A)/(beta*B))**(1/(alpha+beta))

@jax.jit
def compute_optimal_allocation(compute, params):
  a, b, e, alpha, beta = params
  A, B, E = jnp.exp(jnp.array([a, b, e]))

  G = ((alpha*A)/(beta*B))**(1/(alpha+beta))
  a = beta/(alpha+beta)
  b = 1 - a

  return G*(compute/6)**a, G**(-1) * (compute/6)**b

def compute_optimal_reducible_loss(compute, params):
  N_opt, D_opt = compute_optimal_allocation(compute, params)
  return scaling_law_reducible(N_opt, D_opt, params)

@jax.jit
def optimal_compute_from_reducible_loss(loss, params):
  a, b, e, alpha, beta = params
  A, B, E = jnp.exp(jnp.array([a, b, e]))

  G = ((alpha*A)/(beta*B))**(1/(alpha+beta))
  a = beta/(alpha+beta)
  b = 1 - a

  return 6 * (loss/(G**(-alpha) * A + G**beta * B))**(-(alpha + beta)/(alpha*beta))

@jax.jit
def compute_optimal_allocation_from_shares(compute, G, a):
  b = 1-a
  return G*(compute/6)**a, G**(-1) * (compute/6)**b

def ratio(params_and_tokens):
  params, tokens = params_and_tokens
  return tokens/params

def inner(simulated_params, threshold, N_true_opt, D_true_opt):
  """Find the optimal allocation given a set of parameters, and compare with Chinchilla.
  
  Packaged into a function to vectorize the inner loop, hence the name."""
  N_opt, D_opt = compute_optimal_allocation(threshold, simulated_params)

  loss_achieved_by_chinchilla = scaling_law_reducible(N_true_opt, D_true_opt, simulated_params)
  compute_needed_for_loss = optimal_compute_from_reducible_loss(loss_achieved_by_chinchilla, simulated_params)
  return jnp.array([D_opt/N_opt, threshold/compute_needed_for_loss])

inner_vec = jax.vmap(inner, (0, None, None, None), 0)

# %%
# compute_thresholds = 10.0**np.array([18, 28])
compute_thresholds = 10.0**np.arange(18, 28, .05)
conf_int_percentile = 80
low, high = (100-conf_int_percentile)/2, 100 - (100-conf_int_percentile)/2

D_N_ratio_conf_int = [[], [], []]
D_N_ratios = []
chinchilla_D_N_ratio = []

compute_loss_factors = []

simulated_params_list = multivariate_normal.rvs(mean=estimated_params, cov=cov_matrix[:5, :5], size=10000)

for threshold in compute_thresholds:
  N_true_opt, D_true_opt = compute_optimal_allocation_from_shares(threshold, G(true_params), a_mid)
  D_N_true_ratio = D_true_opt/N_true_opt

  D_N_ratio, compute_loss_factor = inner_vec(simulated_params_list, threshold, N_true_opt, D_true_opt).T

  D_N_ratio_conf_int[0].append(np.percentile(D_N_ratio, low))
  D_N_ratio_conf_int[1].append(np.median(D_N_ratio))
  D_N_ratio_conf_int[2].append(np.percentile(D_N_ratio, high))

  chinchilla_D_N_ratio.append(D_N_true_ratio)

  D_N_ratios.append(D_N_ratio)
  compute_loss_factors.append(compute_loss_factor)

D_N_ratios = np.array(D_N_ratios)
compute_loss_factors = np.array(compute_loss_factors)

# %%
def log_format(val, pos):
    """Format the tick labels on logarithmic scale."""
    val_str = '{:g}'.format(val)
    if float(val_str) >= 1.0:
        # If the value is a whole number, return it as an integer.
        return str(int(val))
    else:
        # Otherwise, return the string as is (useful for fractional values).
        return val_str

# %%
# Assuming your previous variables and data (compute_thresholds, D_N_ratio_conf_int, etc.) are defined
chinchilla_lower = ratio(compute_optimal_allocation_from_shares(compute_thresholds, G(true_params), a_low))
chinchilla_upper = ratio(compute_optimal_allocation_from_shares(compute_thresholds, G(true_params), a_high))

# %%
from matplotlib.ticker import FuncFormatter

chinchilla_compute = (1.4*10**12)*(70*10**9)*6

color_ours = 0.35
color_chinchilla = 0.65

# Define line width for better visibility
line_width = 2.5

plt.figure(figsize=(15/2, 7/1.85))  # Width: 10 inches, Height: 6 inches

plt.plot(compute_thresholds, D_N_ratio_conf_int[1], label="Optimal policy (ours)", color=plt.cm.viridis(color_ours), linewidth=line_width)
#plt.plot(compute_thresholds, D_N_ratio_conf_int[0], color=plt.cm.viridis(color_ours), linestyle="dashed")
#plt.plot(compute_thresholds, D_N_ratio_conf_int[2], color=plt.cm.viridis(color_ours), linestyle="dashed")

plt.plot(compute_thresholds, chinchilla_D_N_ratio, \
         label="Optimal policy (Hoffmann et al.)", color=plt.cm.viridis(color_chinchilla), linewidth=line_width)
# Assuming chinchilla_lower and chinchilla_upper are defined previously along with their respective function calculations
#plt.plot(compute_thresholds, chinchilla_lower, label="", color=plt.cm.viridis(color_chinchilla), linestyle="dashed")
#plt.plot(compute_thresholds, chinchilla_upper, label="", color=plt.cm.viridis(color_chinchilla), linestyle="dashed")

plt.fill_between(compute_thresholds, D_N_ratio_conf_int[0], D_N_ratio_conf_int[2], color=plt.cm.viridis(color_ours), alpha=0.25, label='')
plt.fill_between(compute_thresholds, chinchilla_lower, chinchilla_upper, color=plt.cm.viridis(color_chinchilla), alpha=0.25, label='')

plt.axhline(y=20, color='gray', linestyle='--')
plt.text(x=7e24, y=20, s=r"$D/N = 20$ rule of thumb", color='gray', fontsize = 10, verticalalignment='bottom')

# Adding the round filled marker at (chinchilla_compute, 20)
plt.plot(chinchilla_compute, 20, 'o', color='black', markersize=8, label="Chinchilla model", alpha=0.75)  # 'o' is the marker style for a filled circle

plt.xscale("log")
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(FuncFormatter(log_format))

plt.xlim([min(compute_thresholds), max(compute_thresholds)])
plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to make space for the x-axis

plt.xlabel("Training compute (FLOP)")
plt.ylabel("Tokens per parameters ratio")
plt.legend(loc='upper left')

plt.savefig("tokens_to_params_ratio_plot.pdf")
plt.show()

# %%
# Assuming compute_thresholds and chinchilla_D_N_ratio are numpy arrays or can be converted into numpy arrays
compute_thresholds = np.array(compute_thresholds)  # Convert to numpy array if not already
chinchilla_D_N_ratio = np.array(chinchilla_D_N_ratio)  # Convert to numpy array if not already

# Calculate the absolute difference between each element in compute_thresholds and chinchilla_compute
abs_difference = np.abs(compute_thresholds - chinchilla_compute)

# Find the index of the smallest difference
index_closest = np.argmin(abs_difference)

# Retrieve and print the value of chinchilla_D_N_ratio at this index
value_closest = chinchilla_D_N_ratio[index_closest]
print(f"The value of chinchilla_D_N_ratio closest to chinchilla_compute is: {value_closest}")

# %%
# Assuming compute_thresholds and chinchilla_D_N_ratio are numpy arrays or can be converted into numpy arrays
compute_thresholds = np.array(D_N_ratio_conf_int[1])  # Convert to numpy array if not already
chinchilla_D_N_ratio = np.array(D_N_ratio_conf_int[1])  # Convert to numpy array if not already

# Calculate the absolute difference between each element in compute_thresholds and chinchilla_compute
abs_difference = np.abs(compute_thresholds - chinchilla_compute)

# Find the index of the smallest difference
index_closest = np.argmin(abs_difference)

# Retrieve and print the value of chinchilla_D_N_ratio at this index
value_closest = chinchilla_D_N_ratio[index_closest]
print(f"The value of chinchilla_D_N_ratio closest to chinchilla_compute is: {value_closest}")

# %% [markdown]
# # Replicate with lower loss scale

# %%
true_params = jnp.array([np.log(406.4), np.log(410.7), np.log(1.69), 0.34, 0.28])

# %%
import functools

# Set up the grid for initial parameter values
alpha_vals = np.linspace(0., 2., 10)
beta_vals = np.arange(0, 2.5, 0.5)
e_vals = np.linspace(0., 2., 10)
a_vals = np.linspace(6, 30, 10)
b_vals = np.arange(0, 30, 5)

# Perform the optimization using L-BFGS over the grid of initial values
best_loss = np.inf
best_params = None

results_dict = {}
for alpha, e, a in product(alpha_vals, e_vals, a_vals):
    init_params = [a, a, e, alpha, alpha]
    # result = minimize(partial(huber_loss_objective, reduce_fn=jnp.sum),
    #                   init_params, args=(N[indices], D[indices], losses[indices]),
    #                   method='L-BFGS-B')
    result = jax_min(huber_loss_objective,
                    jnp.array(init_params), args=(N[indices], D[indices], losses[indices]))
    # TODO: why does partial make it so much slower??
    # print(result.message)
    results_dict[tuple(init_params)] = {'params': result.x, 'loss': result.fun}
    if result.success and result.fun < best_loss:
        best_loss = result.fun
        best_params = result.x
        print(f"New best loss: {best_loss}")
        print(f"Best params: {best_params}")
        print(f"Initial guess: {init_params}")
        print(f"nfev: {result.nfev}")

# Transform the fitted parameters a, b, e to A, B, E
if best_params is not None:
    A = np.exp(best_params[0])
    B = np.exp(best_params[1])
    E = np.exp(best_params[2])
    alpha = best_params[3]
    beta = best_params[4]
    print(f"Best fit parameters: A={A}, B={B}, E={E}, alpha={alpha}, beta={beta}")
else:
    print("Optimization failed to converge.")

A = np.exp(true_params[0])
B = np.exp(true_params[1])
E = np.exp(true_params[2])
alpha = true_params[3]
beta = true_params[4]
print(f"\nParameters from Chinchilla paper: A={A}, B={B}, E={E}, alpha={alpha}, beta={beta}")

# %% [markdown]
# Show effect of lower loss scale

# %%
# true_params = np.array([np.log(406.4), np.log(410.7), np.log(1.69), 0.34, 0.28])
true_params = np.array([6.0073404, 6.0179186, 0.5267228, 0.33917084, 0.2849083])
true_params_rounded = np.array([np.log(406.4), np.log(410.7), np.log(1.69), 0.34, 0.28])

@jax.jit
def huber_loss_objective_mean(params, N, D, losses):
    a, b, e, alpha, beta = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return jnp.mean(huber_optax(predictions, jnp.log(losses), delta=1e-3))

# %%
# Initialize parameters
init_params = true_params

# Store loss values
sum_loss_values_for_sum_opt = []
sum_loss_values_for_mean_opt = []

# Callback functions to log loss values
def callback_sum(params):
    sum_loss = huber_loss_objective(params, N[indices], D[indices], losses[indices])
    sum_loss_values_for_sum_opt.append(sum_loss)

def callback_mean(params):
    # Calculate the sum loss even though the mean loss is being optimized
    sum_loss = huber_loss_objective(params, N[indices], D[indices], losses[indices])
    sum_loss_values_for_mean_opt.append(sum_loss)

# Perform optimization with logging
result_sum = minimize(huber_loss_objective, init_params, args=(N[indices], D[indices], losses[indices]),
                      jac=jac, method='BFGS', callback=callback_sum)

result_mean = minimize(huber_loss_objective_mean, init_params, args=(N[indices], D[indices], losses[indices]),
                       method='L-BFGS-B', callback=callback_mean)

# Plotting the loss values
plt.plot(sum_loss_values_for_sum_opt, label='Loss scale (corrected)')
plt.plot(sum_loss_values_for_mean_opt, label='Loss scale (Hoffmann et al.)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
#plt.title('Sum Loss vs. Iterations')
plt.show()

# %%



