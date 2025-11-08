import numpy as np
import matplotlib.pyplot as plt

# ===================================================================
# PHASE 1: THE ENGINE (`landscape.py` logic, now inside the notebook)
# ===================================================================

class LossSurface:
    """
    A class to analyze and visualize 2D loss surfaces.
    It computes geometry numerically from first principles.
    """
    def __init__(self, func):
        self.func = func

    def compute_gradient_at(self, point: np.ndarray, h=1e-6):
        """ Computes the gradient vector using the central finite difference method. """
        x, y = point
        df_dx = (self.func(x + h, y) - self.func(x - h, y)) / (2 * h)
        df_dy = (self.func(x, y + h) - self.func(x, y - h)) / (2 * h)
        return np.array([df_dx, df_dy])

    def compute_hessian_at(self, point: np.ndarray, h=1e-6):
        """ Computes the 2x2 Hessian matrix using central finite differences. """
        x, y = point
        f_xx = (self.func(x + h, y) - 2 * self.func(x, y) + self.func(x - h, y)) / (h**2)
        f_yy = (self.func(x, y + h) - 2 * self.func(x, y) + self.func(x, y - h)) / (h**2)
        f_xy = (self.func(x + h, y + h) - self.func(x + h, y - h) - self.func(x - h, y + h) + self.func(x - h, y - h)) / (4 * h**2)
        return np.array([[f_xx, f_xy], [f_xy, f_yy]])

    def analyze_curvature_at(self, point: np.ndarray):
        """ Analyzes the local geometry at a point using the Hessian's eigendecomposition. """
        hessian = self.compute_hessian_at(point)
        eigenvalues, eigenvectors = np.linalg.eig(hessian)
        
        order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        condition_number = np.abs(eigenvalues[0]) / (np.abs(eigenvalues[1]) + 1e-8)

        point_type = "Undetermined"
        if np.all(eigenvalues > 0.1): point_type = "Steep Valley"
        elif np.all(eigenvalues > 0): point_type = "Flat Basin (Minimum)"
        elif np.all(eigenvalues < 0): point_type = "Ridge (Maximum)"
        elif eigenvalues[0] * eigenvalues[1] < 0: point_type = "Saddle Point"

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "condition_number": condition_number,
            "point_type": point_type
        }

    def run_optimizer(self, start_point: np.ndarray, optimizer_type='sgd', 
                        lr=0.001, momentum=0.9, num_steps=5000):
        """ Simulates the path of an optimizer on the surface. """
        path = [start_point]
        position = np.copy(start_point)
        velocity = np.zeros_like(position)

        for _ in range(num_steps):
            grad = self.compute_gradient_at(position)
            
            if optimizer_type == 'sgd':
                position = position - lr * grad
            elif optimizer_type == 'momentum':
                velocity = momentum * velocity - lr * grad
                position = position + velocity
            else:
                raise ValueError("Unsupported optimizer type")
                
            path.append(np.copy(position))
            
            if np.linalg.norm(position) > 1e3 or np.linalg.norm(path[-1] - path[-2]) < 1e-9:
                break
        
        return np.array(path)

