from Utils import physical_model

def runge_kutta_4_step(step_idx, state_array, control_input, dt, A, B):
    """
    Perform one integration step using the 4th-order Runge-Kutta (RK4) method for a linear system:
        dx/dt = A * x + B * u

    Parameters:
        step_idx (int): Current time step index
        state_array (np.ndarray): Array of state values over time
        control_input (float): Control input u at current step
        dt (float): Time step size
        A (float): System dynamic coefficient
        B (float): Input gain

    Returns:
        float: State at next time step
    """

    x = state_array[step_idx]

    k1 = physical_model(x, control_input, A, B)
    k2 = physical_model(x + 0.5 * dt * k1, control_input, A, B)
    k3 = physical_model(x + 0.5 * dt * k2, control_input, A, B)
    k4 = physical_model(x + dt * k3, control_input, A, B)

    next_x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return next_x