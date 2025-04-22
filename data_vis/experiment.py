import numpy as np

def run_experiments(num_runs, num_iterations, experiment, data_gen_func, base_seed=42, output="Error Plot",
                    y_lab=r"\text{Relative Error} $\displaystyle\frac{\lVert \tilde{D} - WH \rVert}{\lVert \tilde{D} \rVert}$"):
    """
    Runs multiple experiments and collects error values.

    Parameters:
        num_runs (int): Number of independent experimental runs.
        num_iterations (int): Number of iterations per run.
        experiments (dict): Dictionary mapping experiment labels to experiment specifications.
            Each specification is a dict with the following (optional) keys:
                - 'alg_func': (required) Function handle for the algorithm (e.g., StandardNMF or CorruptionRobustNMF).
                - 'alg_params': (optional) Dictionary of additional parameters to pass to the algorithm.
                - 'data_params': (optional) Dictionary of parameters to pass to the data generation function.
                - 'data_input': (optional) Tuple (ref_choice, train_choice) where each element is either
                                "X" (uncorrupted data) or "X_corrupted" (corrupted data). Default is ("X", "X_corrupted").
                - 'model_rank': (optional) Target rank. If provided, it will be passed as the fourth argument.
        data_gen_func (function): Function to generate data. It should accept keyword arguments (if any)
                                  and return a tuple (X, X_corrupted).
        base_seed (int): Base random seed for reproducibility.
        output string: Strings indicating what output to generate. Can be "Error Plot" or "Swimmer Image" (coming soon).

    Returns:
        results (dict): Dictionary mapping experiment labels to numpy arrays of error values with shape
                        (num_runs, T), where T is the length of the error vector returned by the algorithm.
    """
    results = {label: [] for label in experiment}
    runtimes = {label: [] for label in experiment}
    seed = base_seed

    for run in range(num_runs):
        np.random.seed(seed)
        for label, exp in experiment.items():
            # Get data generation parameters (if any) and generate data.
            data_params = exp.get('data_params', {})
            D, D_tilde = data_gen_func(**data_params)

            # Determine which data to use as reference and for training.
            # data_input should be a tuple: (ref_choice, train_choice), each either "X" or "X_corrupted".
            data_input = exp.get('data_input', ("X", "X_corrupted"))
            D_ref = D_tilde if data_input[0] == "D_tilde" else D
            D_train = D_tilde if data_input[1] == "D_tilde" else D

            # Retrieve the algorithm function and any extra parameters.
            alg_func = exp['alg_func']
            alg_params = exp.get('alg_params', {})
            model_rank = exp.get('model_rank', None)

            # Run the algorithm. If model_rank is specified, pass it as the fourth argument.
            outputs = alg_func(D_ref, D_train, num_iterations, model_rank, **alg_params)
            # Assume that the error vector is the last output.
            errors = outputs[-2]
            results[label].append(errors)
            runtime = outputs[-1]
            runtimes[label].append(runtime)
        seed += 1

    # Convert list of error vectors to numpy arrays.
    for label in results:
        results[label] = np.array(results[label])

    if output == "Error Plot":
          plot_experiment(results, log_y_axis=True, y_lab=y_lab, runtimes=runtimes)
    
    return results