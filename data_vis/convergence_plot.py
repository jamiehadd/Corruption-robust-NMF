import numpy as np
import matplotlib.pylab as plt

def plot_experiment(errors, log_y_axis=False, runtimes=None,
                    y_lab=r"\text{Relative Error} $\displaystyle\frac{\lVert \tilde{D} - WH \rVert}{\lVert \tilde{D} \rVert}$"):
    """
    Plots the evolution of relative error over iterations for multiple experiments.

    Parameters:
        errors (dict): Dictionary mapping method labels (str) to numpy arrays of shape
                       (num_runs, num_iterations+1) containing relative error values.
        log_y_axis (bool): If True, sets the y-axis to a logarithmic scale.
        runtimes (dict): Dictionary mapping method labels to list of experiment runtimes
        y_lab (str): Label for the y-axis (supports LaTeX formatting).
    """
    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (19, 11)
    plt.rcParams["font.size"] = 32
    plt.rcParams["xtick.color"] = 'black'
    plt.rcParams["ytick.color"] = 'black'
    plt.rcParams["axes.edgecolor"] = 'black'
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    labels = list(errors.keys())
    num_iterations = errors[labels[0]].shape[1]
    domain = range(num_iterations)

    dashes = [
        [1, 0],      # Solid
        [6, 2],      # Dashed
        [1, 2],      # Dotted
        [4, 2, 1, 2],# Dash-Dot
        [8, 2, 2, 2],# Custom
        [3, 1, 1, 1]
    ]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'teal', 'yellow', 'pink']

    plt.figure()

    for i, label in enumerate(labels):
        data_matrix = errors[label]
        mean_error = np.mean(data_matrix, axis=0)
        min_error = np.min(data_matrix, axis=0)
        max_error = np.max(data_matrix, axis=0)

        plt.plot(domain, mean_error, label=label, color=colors[i], dashes=dashes[i], linewidth=3)
        plt.fill_between(domain, min_error, max_error, color=colors[i], alpha=0.2)
        
        # if runtimes provided, add to end of line plot
        if runtimes is not None:
            avg_runtime = np.mean(runtimes[label])
            final_x = domain[-1]
            final_y = mean_error[-1]
            plt.text(final_x+3, final_y, f"{avg_runtime:.2f}s", fontsize=24, va='center', ha='left')
            

    plt.xlabel("Iteration", color='black')
    plt.ylabel(y_lab, color='black')

    if log_y_axis:
        plt.yscale('log')

    plt.xticks(ticks=np.arange(0, num_iterations, step=max(1, num_iterations // 8)))
    plt.xlim(right=domain[-1] + 25)
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend(fancybox=True, handlelength=1.5, shadow=False, loc='best', ncol=1,
               fontsize=36, framealpha=1.0, edgecolor='black', borderpad=0.4, borderaxespad=0.1)
    
    plt.show()