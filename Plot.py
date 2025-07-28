import matplotlib.pyplot as plt

def plot_simulation_results(time, sim_steps, reference, output, measured_output, control_input):
    """
    Plot the simulation results including reference, output, measured output, and control input.

    Parameters:
        time (np.ndarray): Array of time points.
        sim_steps (int): Number of simulation steps.
        reference (np.ndarray): Reference signal array.
        output (np.ndarray): System output array.
        measured_output (np.ndarray): Measured system output array.
        control_input (np.ndarray): Control input signal array.
    """
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time[:sim_steps], reference[:sim_steps], 'y', label='Reference')
    plt.plot(time[:sim_steps], output[:sim_steps], 'b', label='Output')
    plt.plot(time[:sim_steps], measured_output[:sim_steps], 'r--', label='Measured')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time[:sim_steps], control_input[:sim_steps], 'r', label='Control Input')
    plt.grid(True)
    plt.tight_layout()
    plt.show()