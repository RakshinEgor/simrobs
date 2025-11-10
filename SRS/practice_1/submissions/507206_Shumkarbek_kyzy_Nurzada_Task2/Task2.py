import numpy as np
import matplotlib.pyplot as plt

def mass_spring_damper_vertical(x):
    """
    Vertical mass-spring-damper system: m·ẍ + b·ẋ + k·x = -m·g
    """
    m, b, k, g = 0.6, 0.045, 6.8, 9.81
    x_pos, x_vel = x
    F_gravity = -m * g  # Constant force
    x_acc = (F_gravity - b*x_vel - k*x_pos) / m
    return np.array([x_vel, x_acc])

def analytic_solution(t):
    """
    Analytical solution for vertical mass-spring-damper
    x(t) = e^(-δt)·[C₁·cos(ωt) + C₂·sin(ωt)] + x_steady
    """
    delta = 0.0375
    omega = 3.366
    C1 = 1.6956
    C2 = 0.01889
    x_steady = -0.8656
    return np.exp(-delta * t) * (C1 * np.cos(omega * t) + C2 * np.sin(omega * t)) + x_steady

def analytic_derivative(t):
    """
    Derivative of analytical solution
    """
    delta = 0.0375
    omega = 3.366
    C1 = 1.6956
    C2 = 0.01889
    x_steady = -0.8656
    
    term1 = -delta * np.exp(-delta * t) * (C1 * np.cos(omega * t) + C2 * np.sin(omega * t))
    term2 = np.exp(-delta * t) * (-C1 * omega * np.sin(omega * t) + C2 * omega * np.cos(omega * t))
    return term1 + term2

# Numerical integration methods (same as before)
def forward_euler(fun, x0, Tf, h):
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k] + h * fun(x_hist[:, k])
    return x_hist, t

def backward_euler(fun, x0, Tf, h, tol=1e-8, max_iter=100):
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k]
        for i in range(max_iter):
            x_next = x_hist[:, k] + h * fun(x_hist[:, k + 1])
            error = np.linalg.norm(x_next - x_hist[:, k + 1])
            x_hist[:, k + 1] = x_next
            if error < tol:
                break
    return x_hist, t

def runge_kutta4(fun, x0, Tf, h):
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    for k in range(len(t) - 1):
        k1 = fun(x_hist[:, k])
        k2 = fun(x_hist[:, k] + 0.5 * h * k1)
        k3 = fun(x_hist[:, k] + 0.5 * h * k2)
        k4 = fun(x_hist[:, k] + h * k3)
        x_hist[:, k + 1] = x_hist[:, k] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_hist, t

# Parameters
x0 = np.array([0.83, 0.0])  # x(0) = 0.83, x'(0) = 0.0
Tf = 20.0  # Longer time to see oscillations
h = 0.01

# Solve numerically
print("Solving vertical mass-spring-damper system...")
x_fe, t_fe = forward_euler(mass_spring_damper_vertical, x0, Tf, h)
x_be, t_be = backward_euler(mass_spring_damper_vertical, x0, Tf, h)
x_rk4, t_rk4 = runge_kutta4(mass_spring_damper_vertical, x0, Tf, h)

# Analytical solution
t_analytical = np.linspace(0, Tf, 1000)
x_analytical = analytic_solution(t_analytical)
dx_analytical = analytic_derivative(t_analytical)

# Plot results
plt.figure(figsize=(20, 12))

# Plot 1: Position comparison
plt.subplot(2, 2, 1)
plt.plot(t_analytical, x_analytical, 'k-', label='Analytical', linewidth=2)
plt.plot(t_fe, x_fe[0, :], 'r--', label='Forward Euler', alpha=0.8)
plt.plot(t_be, x_be[0, :], 'g--', label='Backward Euler', alpha=0.8)
plt.plot(t_rk4, x_rk4[0, :], 'b--', label='RK4', alpha=0.8)
plt.axhline(y=-0.8656, color='m', linestyle=':', label='Steady state = -0.8656', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Position x(t) (m)')
plt.legend()
plt.title('Vertical Mass-Spring-Damper: Position vs Time')
plt.grid(True)

# Plot 2: Velocity comparison
plt.subplot(2, 2, 2)
plt.plot(t_analytical, dx_analytical, 'k-', label='Analytical', linewidth=2)
plt.plot(t_fe, x_fe[1, :], 'r--', label='Forward Euler', alpha=0.8)
plt.plot(t_be, x_be[1, :], 'g--', label='Backward Euler', alpha=0.8)
plt.plot(t_rk4, x_rk4[1, :], 'b--', label='RK4', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Velocity dx/dt (m/s)')
plt.legend()
plt.title('Vertical Mass-Spring-Damper: Velocity vs Time')
plt.grid(True)

# Plot 3: Phase portrait
plt.subplot(2, 2, 3)
plt.plot(x_analytical, dx_analytical, 'k-', label='Analytical', linewidth=1)
plt.plot(x_fe[0, :], x_fe[1, :], 'r-', label='Forward Euler', alpha=0.6)
plt.plot(x_be[0, :], x_be[1, :], 'g-', label='Backward Euler', alpha=0.6)
plt.plot(x_rk4[0, :], x_rk4[1, :], 'b-', label='RK4', alpha=0.6)
plt.plot(-0.8656, 0, 'mo', markersize=8, label='Steady State')
plt.xlabel('Position x(t) (m)')
plt.ylabel('Velocity dx/dt (m/s)')
plt.legend()
plt.title('Phase Portrait')
plt.grid(True)

# Plot 4: Errors
plt.subplot(2, 2, 4)
error_fe = np.abs(x_fe[0, :] - analytic_solution(t_fe))
error_be = np.abs(x_be[0, :] - analytic_solution(t_be))
error_rk4 = np.abs(x_rk4[0, :] - analytic_solution(t_rk4))

plt.semilogy(t_fe, error_fe, 'r-', label='Forward Euler')
plt.semilogy(t_be, error_be, 'g-', label='Backward Euler')
plt.semilogy(t_rk4, error_rk4, 'b-', label='RK4')
plt.xlabel('Time (s)')
plt.ylabel('Absolute Error (m)')
plt.legend()
plt.title('Numerical Errors (Log Scale)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Error analysis
print("\n" + "="*50)
print("ERROR ANALYSIS")
print("="*50)
print(f"Maximum Absolute Errors:")
print(f"Forward Euler:  {np.max(error_fe):.6e} m")
print(f"Backward Euler: {np.max(error_be):.6e} m")
print(f"RK4:            {np.max(error_rk4):.6e} m")

print(f"\nFinal positions at t = {Tf} s:")
print(f"Analytical:     {analytic_solution(Tf):.6f} m")
print(f"Forward Euler:  {x_fe[0, -1]:.6f} m")
print(f"Backward Euler: {x_be[0, -1]:.6f} m")
print(f"RK4:            {x_rk4[0, -1]:.6f} m")