import numpy as np

# Define cost functions
def m_s_e(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)

def m_a_e(y_actual, y_predicted):
    return np.mean(np.abs(y_actual - y_predicted))

def r_squared(y_actual, y_predicted):
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

def gradient_descent(x, y, learning_rate=0.01, num_iterations=1000,
                     error_func=m_s_e, regularization=None, lambda_reg=0.1, 
                     single_step=False, verbose=False):
    m, b = 0, 0  # Initialize parameters
    n = len(x)
    
    # Lists to store cost and predictions over time for visualization
    cost_history = []
    predictions_history = []

    if verbose:
        print(f"Starting gradient descent with learning_rate={learning_rate}, num_iterations={num_iterations}, "
              f"regularization={regularization}, lambda_reg={lambda_reg}")

    for i in range(num_iterations):
        y_predicted = m * x + b
        
        if error_func == m_s_e:
            m_gradient = (-2 / n) * np.sum(x * (y - y_predicted))
            b_gradient = (-2 / n) * np.sum(y - y_predicted)
        elif error_func == m_a_e:
            m_gradient = (-1 / n) * np.sum(x * np.sign(y - y_predicted))
            b_gradient = (-1 / n) * np.sum(np.sign(y - y_predicted))
        else:
            raise ValueError("Unsupported error function. Use 'm_s_e' or 'm_a_e'.")

        # Apply regularization if specified
        if regularization == 'L1':
            m_gradient += lambda_reg * np.sign(m)
        elif regularization == 'L2':
            m_gradient += lambda_reg * m
        
        # Update parameters
        m -= learning_rate * m_gradient
        b -= learning_rate * b_gradient

        # Calculate and store cost for this iteration
        cost = error_func(y, y_predicted)
        cost_history.append(cost)
        predictions_history.append(y_predicted)

        # Debugging information at each step if verbose is enabled
        if verbose:
            print(f"Iteration {i + 1}/{num_iterations}: m={m:.4f}, b={b:.4f}, cost={cost:.4f}")
        
        # If in single-step mode, pause and wait for user input
        if single_step:
            input(f"Press Enter to continue to iteration {i + 2}/{num_iterations}...")

    if verbose:
        print(f"Completed gradient descent. Final values: m={m}, b={b}, final cost={cost_history[-1]:.4f}")
        print("Returning m, b, predictions_history, cost_history")

    return m, b, predictions_history, cost_history
