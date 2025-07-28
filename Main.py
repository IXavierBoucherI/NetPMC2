import numpy as np
from Model_Free import model_free_control
from Plot import plot_simulation_results
from Runge_Kutta import runge_kutta_4_step
from Utils import internal_ref_func

TEST = True  # Enable/disable progress printing

# Simulation parameters
SIM_STEPS = 1000            # Number of simulation time steps
TIME_FINAL = 0.001          # Total simulation time (seconds)
TIME_STEP = 1e-5            # Time step size (seconds)
REF_VALUE = 8.55            # Constant reference signal value

# System parameters (linear system dx/dt = A*x + B*u, output = C*x)
A, B, C = -1e5, 1, 1e3

# Preallocate arrays
time = np.zeros(SIM_STEPS + 1)
internal_ref = np.zeros(SIM_STEPS)
output = np.zeros(SIM_STEPS + 2)
measured_output = np.zeros(SIM_STEPS + 2)
tracking_error = np.zeros(SIM_STEPS + 1)
control_proportional = np.zeros(SIM_STEPS + 1)
control_integral = np.zeros(SIM_STEPS + 1)
integral_trapz = np.zeros(SIM_STEPS + 1)
control_final = np.zeros(SIM_STEPS + 1)
state_RK = np.zeros(SIM_STEPS + 2)
control_log = np.zeros(SIM_STEPS + 2)

# Reference signal
reference = REF_VALUE * np.ones(SIM_STEPS + 1)

def main():
    t_progress = 0

    PMA_params = {
        'Kp': 10.0,  # Proportional gain for the PMA controller: scales the immediate error correction
        'Kint': 1.0,  # Integral gain: scales the accumulated error for steady-state error correction
        'K_alpha': 1e3,  # Initial amplitude (alpha) for the internal reference exponential decay function
        'K_beta': 40,  # Decay rate (beta) controlling how fast the internal reference signal decays
        'FinalScale': 10.0,  # Scaling factor to adjust the final computed control signal magnitude
        'y_internal_start_func': internal_ref_func  # Function generating the internal reference signal over time
    }

    for step in range(1, SIM_STEPS + 1):
        time[step] = step * TIME_STEP

        # Model-free control calculation
        u_f, y_int_val, u_p_val, u_i_val, integral_val, error = model_free_control(
            step, internal_ref_func, measured_output, reference,
            control_proportional, control_integral, integral_trapz, PMA_params,
            TIME_STEP, PMA_params['FinalScale']
        )

        # Store updated values
        internal_ref[step - 1] = y_int_val
        control_proportional[step] = u_p_val
        control_integral[step] = u_i_val
        integral_trapz[step] = integral_val
        control_final[step] = u_f
        tracking_error[step] = error
        control_log[step] = u_f

        # State update via RK4
        state_RK[step + 1] = runge_kutta_4_step(step, state_RK, u_f, TIME_STEP, A, B)
        output[step] = C * state_RK[step + 1]
        measured_output[step + 1] = output[step]

        # Progress display
        if TEST:
            progress = 100 * time[step] / TIME_FINAL
            if progress >= t_progress:
                t_progress += 25
                print(f"Simulation progress: {int(progress)}%")
            print(f"Step: {step}")

    # Plot results
    plot_simulation_results(time, SIM_STEPS, reference, output, measured_output, control_final)


if __name__ == "__main__":
    main()