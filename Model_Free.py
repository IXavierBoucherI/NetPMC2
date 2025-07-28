def model_free_control(
    step,                 # Current simulation step (index)
    internal_ref_signal,  # Array storing internal reference values over time
    measured_output,      # Array of measured system outputs
    reference_signal,     # Desired target/reference signal
    pma_integral,         # PMA's internal control accumulation (proportional + integral)
    integrated_error_log, # Stores past integrated errors (Ki * error)
    trapz_integral_log,   # Stores the numerical trapezoidal integration results
    pma_params,           # Dictionary of PMA controller parameters (Kp, Ki, K_alpha, etc.)
    time_step,            # Simulation time step (e.g., 1e-5)
    final_scale           # Scaling factor for the final control signal
):
    """
    Perform one iteration of model-free control signal calculation using a PMA controller.
    
    Returns updated values for:
        - final_control_signal: computed control action for this step
        - internal_ref_value: internal reference signal at this step
        - updated_pma_integral: updated internal controller state
        - integrated_error: current step's integrated error
        - trapz_integral: updated trapezoidal integration
        - tracking_error: error between desired and measured output
    """

    # Evaluate internal reference signal at current time (exponentially decaying function)
    internal_ref_value = pma_params['y_internal_start_func'](
        step * time_step, pma_params['K_alpha'], pma_params['K_beta'])

    # Error between internal reference and measured output
    internal_error = internal_ref_value - measured_output[step - 1]

    # Error between external reference signal and measured output (tracking error)
    tracking_error = reference_signal[step] - measured_output[step - 1]

    # Update internal proportional-integral control accumulator
    updated_pma_integral = pma_integral[step - 1] + pma_params['Kp'] * internal_error

    # Compute current integrated error
    integrated_error = pma_params['Kint'] * tracking_error

    # Update trapezoidal integration of error signal
    trapz_integral = trapz_integral_log[step - 1] + time_step * (
        integrated_error + integrated_error_log[step - 1]) / 2

    # Final control signal scaled by trapezoidal integral
    final_control_signal = updated_pma_integral * trapz_integral / final_scale

    return (final_control_signal,
            internal_ref_value,
            updated_pma_integral,
            integrated_error,
            trapz_integral,
            tracking_error)