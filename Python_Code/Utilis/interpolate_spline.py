import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Small epsilon value to prevent numerical issues with log(0) and sqrt(0)
EPSILON = 0.0000000001


def interpolate_spline(train_points, train_values, query_points, order, regularization_weight=0.0):
    """Interpolate signal using polyharmonic interpolation.
    
    The interpolant has the form:
    f(x) = sum_{i=1}^n w_i * phi(||x - c_i||) + v^T * x + b
    
    This is a sum of two terms:
    1. A weighted sum of radial basis function (RBF) terms with centers c_i
    2. A linear term with bias
    
    The coefficients w and v are estimated such that:
    - The interpolant exactly fits the training data
    - The vector w is orthogonal to each training point
    - The vector w sums to 0
    
    Args:
        train_points: Tensor of shape [batch_size, n, d] containing n d-dimensional
            training locations. These do not need to be regularly-spaced.
        train_values: Tensor of shape [batch_size, n, k] containing n k-dimensional
            values evaluated at train_points.
        query_points: Tensor of shape [batch_size, m, d] containing m d-dimensional
            locations where we will output the interpolant's values.
        order: Order of the interpolation. Common values:
            - 1 for phi(r) = r
            - 2 for phi(r) = r^2 * log(r) (thin-plate spline)
            - 3 for phi(r) = r^3
        regularization_weight: Weight placed on the regularization term.
            Should be tuned for each problem. Use 0.0 for exact interpolation,
            or small values like 0.001 for regularized interpolation.
    
    Returns:
        Tensor of shape [batch_size, m, k] containing interpolated values
        at the query points.
    """
    # Solve for interpolation coefficients
    w, v = _solve_interpolation(train_points, train_values, order, regularization_weight)
    
    # Apply interpolation to query points
    query_values = _apply_interpolation(query_points, train_points, w, v, order)
    
    return query_values


def _phi(r, order):
    """Radial basis function that defines the order of interpolation.
    
    Implements the polyharmonic spline basis functions as defined in:
    https://en.wikipedia.org/wiki/Polyharmonic_spline
    
    Args:
        r: Input tensor containing squared distances
        order: Interpolation order
    
    Returns:
        Tensor with phi_k evaluated coordinate-wise on r
    """
    # Clamp r to prevent numerical issues with log(0) and sqrt(0)
    # Note: sqrt(0) is well-defined, but its gradient is not
    
    if order == 1:
        r = torch.clamp(r, EPSILON, np.inf)
        return torch.sqrt(r)
    elif order == 2:
        # Thin-plate spline: 0.5 * r * log(r)
        return 0.5 * r * torch.log(torch.clamp(r, EPSILON, np.inf))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(torch.clamp(r, EPSILON, np.inf))
    elif order % 2 == 0:
        # Even orders: 0.5 * r^(k/2) * log(r)
        r = torch.clamp(r, EPSILON, np.inf)
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        # Odd orders: r^(k/2)
        r = torch.clamp(r, EPSILON, np.inf)
        return torch.pow(r, 0.5 * order)


def _cross_squared_distance_matrix(x, y):
    """Compute squared Euclidean distances between points in x and y.
    
    Args:
        x: Tensor of shape [batch_size, n, d]
        y: Tensor of shape [batch_size, m, d]
    
    Returns:
        Tensor of shape [batch_size, n, m] containing squared distances
        between each point in x and each point in y
    """
    # Compute ||x||^2 for each point in x
    x_norm = (x**2).sum(2).view(x.shape[0], x.shape[1], 1)
    
    # Transpose y for matrix multiplication
    y_t = y.permute(0, 2, 1).contiguous()
    
    # Compute ||y||^2 for each point in y
    y_norm = (y**2).sum(2).view(y.shape[0], 1, y.shape[1])
    
    # Use the identity ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    
    # Replace any NaN values with 0 (can occur due to numerical precision)
    dist[dist != dist] = 0
    
    # Ensure all distances are non-negative
    return torch.clamp(dist, 0.0, np.inf)


def _pairwise_squared_distance_matrix(x):
    """Compute pairwise squared distances within a set of points.
    
    Args:
        x: Tensor of shape [batch_size, n, d]
    
    Returns:
        Tensor of shape [batch_size, n, n] containing pairwise squared distances
    """
    return _cross_squared_distance_matrix(x, x)


def _solve_interpolation(train_points, train_values, order, regularization_weight):
    """Solve for interpolation coefficients.
    
    Computes the coefficients of the polyharmonic interpolant using the
    linear system described in the polyharmonic spline formulation.
    
    Args:
        train_points: Tensor of shape [batch_size, n, d] - interpolation centers
        train_values: Tensor of shape [batch_size, n, k] - function values
        order: Order of the interpolation
        regularization_weight: Weight for smoothness regularization
    
    Returns:
        w: Tensor of shape [batch_size, n, k] - weights on each interpolation center
        v: Tensor of shape [batch_size, d+1, k] - weights on input dimensions plus bias
    """
    b, n, d = train_points.shape
    _, _, k = train_values.shape
    
    # Rename variables to match Wikipedia notation for polyharmonic splines
    c = train_points  # Centers
    f = train_values  # Function values
    
    # Construct matrix A: RBF kernel matrix between training points
    matrix_a = _phi(_pairwise_squared_distance_matrix(c), order)  # [b, n, n]
    
    # Add regularization term if specified
    if regularization_weight > 0:
        batch_identity_matrix = torch.eye(n, device=device).unsqueeze(0)
        matrix_a += regularization_weight * batch_identity_matrix
    
    # Construct matrix B: training points augmented with ones for bias term
    ones = torch.ones([b, n, 1], device=device)
    matrix_b = torch.cat([c, ones], 2)  # [b, n, d + 1]
    
    # Construct the full linear system matrix
    # Left block: [A; B^T]
    left_block = torch.cat([matrix_a, matrix_b.permute([0, 2, 1])], 1)
    
    # Right block: [B; 0]
    num_b_cols = matrix_b.shape[2]  # d + 1
    lhs_zeros = torch.zeros([b, num_b_cols, num_b_cols], device=device)
    right_block = torch.cat([matrix_b, lhs_zeros], 1)  # [b, n + d + 1, d + 1]
    
    # Full left-hand side matrix
    lhs = torch.cat([left_block, right_block], 2)  # [b, n + d + 1, n + d + 1]
    
    # Construct right-hand side vector
    rhs_zeros = torch.zeros([b, d + 1, k], device=device)
    rhs = torch.cat([f, rhs_zeros], 1)  # [b, n + d + 1, k]
    
    # Solve the linear system
    w_v = torch.linalg.solve(lhs, rhs)
    
    # Extract weights for RBF terms and linear terms
    w = w_v[:, :n, :]      # RBF weights
    v = w_v[:, n:, :]      # Linear weights (including bias)
    
    return w, v


def _apply_interpolation(query_points, train_points, w, v, order):
    """Apply polyharmonic interpolation model to query points.
    
    Given coefficients w and v, evaluate the interpolated function values
    at the specified query points.
    
    Args:
        query_points: Tensor of shape [batch_size, m, d] - points to evaluate at
        train_points: Tensor of shape [batch_size, n, d] - interpolation centers
        w: Tensor of shape [batch_size, n, k] - RBF weights
        v: Tensor of shape [batch_size, d+1, k] - linear weights (including bias)
        order: Order of the interpolation
    
    Returns:
        Tensor of shape [batch_size, m, k] containing interpolated values
        at the query points
    """
    batch_size = train_points.shape[0]
    num_query_points = query_points.shape[1]
    
    # Compute RBF term: sum of weighted basis functions
    pairwise_dists = _cross_squared_distance_matrix(query_points, train_points)
    phi_pairwise_dists = _phi(pairwise_dists, order)
    rbf_term = torch.bmm(phi_pairwise_dists, w)
    
    # Compute linear term: weighted sum of coordinates plus bias
    # Pad query points with ones for the bias term
    query_points_pad = torch.cat([
        query_points, 
        torch.ones([batch_size, num_query_points, 1], device=device)
    ], 2)
    linear_term = torch.bmm(query_points_pad, v)
    
    # Final interpolated values = RBF term + linear term
    return rbf_term + linear_term





















