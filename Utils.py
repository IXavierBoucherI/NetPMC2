import numpy as np

def physical_model(state, control, A, B):
    """
    Compute the derivative of the system state.
    
    Parameters:
        state (float): Current system state
        control (float): Control input
        A (float): System parameter
        B (float): Input gain

    Returns:
        float: Time derivative of state
    """
    return A * state + B * control

def internal_ref_func(t, alpha, beta):
    """
    Compute the internal reference signal at time t as an exponentially decaying function.

    This function models an internal reference trajectory that starts at amplitude `alpha` and decays 
    exponentially over time with a rate defined by `beta`. It's often used to shape the expected 
    system behavior or to generate a target signal within a control algorithm.

    Parameters:
        t (float): The current time instant at which to evaluate the internal reference.
        alpha (float): Initial amplitude scaling factor of the reference signal.
        beta (float): Exponential decay rate (higher values cause faster decay).

    Returns:
        float: The value of the internal reference signal at time t.
    """
    return alpha * np.exp(-beta * t)