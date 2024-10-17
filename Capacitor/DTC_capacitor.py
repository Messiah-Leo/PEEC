import math
import gmsh
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Constants
eps_0 = 8.854e-12  # Vacuum permittivity, F/m
eps_d = 1.006 * eps_0  # Relative permittivity of the dielectric
C = 1 / (4 * math.pi * eps_d)  # Constant for capacitance calculation

LAMDA_MAX = 1.0e-8  # Minimum lambda for regularization


# Function to compute the surface normal vector of a triangle
def surface_normal_vector(Point_v):
    # Cross product of two vectors on the triangle to get the normal
    normal_vec = np.cross(Point_v[1], Point_v[2])
    return normal_vec / np.linalg.norm(normal_vec)  # Normalize the normal vector


# Function to calculate the Euclidean distance between two points
def distance(P1, P2):
    return np.linalg.norm(P1 - P2)


# Function to compute the area of a triangle
def triangle_area(Point_v):
    v1 = Point_v[1] - Point_v[0]
    v2 = Point_v[2] - Point_v[0]
    return 0.5 * np.linalg.norm(np.cross(v1, v2))


# Function to normalize a vector
def vector_norm(V_vec):
    norm = np.linalg.norm(V_vec)
    return V_vec / norm if norm != 0 else V_vec


# Function to compute capacitance matrix element via distance integral
def F_K_1_N(Point_v_IN, Point_t_IN, N):
    N = (N + 1) // 2  # Adjust for the number of integration steps
    Point_v = np.zeros((3, 3))
    R = np.zeros(3)  # Distances between target and triangle vertices
    L = np.zeros(3)  # Lengths of triangle edges
    S_M = np.zeros(3)
    T_0 = np.zeros(3)
    I_0 = np.zeros((3, N + 1))

    BELTA = np.zeros(3)
    K_1_N = np.zeros(N + 2)
    BUFF = np.zeros(N + 1)

    # Translate coordinates of triangle and target point relative to vertex 0
    Point_v[1] = Point_v_IN[1] - Point_v_IN[0]
    Point_v[2] = Point_v_IN[2] - Point_v_IN[0]
    Point_t = Point_t_IN - Point_v_IN[0]

    # Calculate normal vector of the triangle
    Normal_S_vec = surface_normal_vector(Point_v)

    # Calculate distances and edge lengths
    for i in range(3):
        R[i] = distance(Point_v[i], Point_t)
        L[i] = distance(Point_v[(i + 2) % 3], Point_v[(i + 1) % 3])

    # Compute vectors for integration
    U_vec = Point_v[1] / L[2]
    V_vec = vector_norm(np.cross(np.cross(U_vec, Point_v[2]), U_vec))

    W0 = np.dot(Point_t - Point_v[0], Normal_S_vec)
    U0 = np.dot(Point_t - Point_v[0], U_vec)
    V0 = np.dot(Point_t - Point_v[0], V_vec)

    U3 = np.dot(Point_v[2] - Point_v[0], Point_v[1] - Point_v[0]) / L[2]
    V3 = 2.0 * triangle_area(Point_v) / L[2]

    # Perform integration
    for j in range(N + 1):
        for i in range(3):
            S_M[i] = -((L[2] - U0) * (L[2] - U3) + V0 * V3) / L[0]
            T_0[i] = (V0 * (U3 - L[2]) + V3 * (L[2] - U0)) / L[0]

    # Logarithmic distance integral approximation
    for i in range(3):
        I_0[i, 0] = np.log(np.maximum((R[(i + 2) % 3] + S_M[i]), LAMDA_MAX * 5.0e-10) /
                           np.maximum((R[(i + 1) % 3] + S_M[i]), LAMDA_MAX * 5.0e-10))

    # Return the result of the integral
    for j in range(N + 1):
        BUFF[j] = K_1_N[j + 1]

    return BUFF


# Function to compute centroid of a triangle
def calculate_centroid(points):
    return np.mean(points, axis=0)


# Function to compute capacitance matrix element
def calculate_P_matrix_element(point_v, point_t):
    s = F_K_1_N(point_v, point_t, -1)
    return s[0] * C


# Read the mesh file and return centroids, vertices, and areas
def read_mesh(address):
    gmsh.initialize()
    gmsh.open(address)

    elementTypes, elementTags, _ = gmsh.model.mesh.getElements(dim=2)

    centroids, point_v, areas = [], [], []
    for element in elementTags[0]:
        nodeTags = gmsh.model.mesh.getElement(element)[1]
        points = np.array([gmsh.model.mesh.getNode(tag)[0] for tag in nodeTags])
        centroids.append(calculate_centroid(points))
        point_v.append(points)
        areas.append(triangle_area(points))

    gmsh.finalize()
    return np.array(centroids), np.array(point_v), np.array(areas)


# Load mesh data for upper and lower plates
centroids_u, point_v_u, areas_u = read_mesh("mesh_upper.msh")
centroids_l, point_v_l, areas_l = read_mesh("mesh_lower.msh")

# Concatenate data from both plates
centroids = np.concatenate((centroids_u, centroids_l), axis=0)
point_v = np.concatenate((point_v_u, point_v_l), axis=0)
areas = np.concatenate((areas_u, areas_l), axis=0)
num_elements = len(centroids)

# Initialize capacitance matrix
P = np.zeros((num_elements, num_elements))


# Function to compute capacitance matrix row in parallel
def compute_upper_triangle(i):
    P_row = np.zeros(num_elements)
    for j in range(num_elements):
        P_row[j] = calculate_P_matrix_element(point_v[j], centroids[i])
    return i, P_row


# Compute capacitance matrix using parallel processing
results = Parallel(n_jobs=-1)(delayed(compute_upper_triangle)(i) for i in tqdm(range(num_elements), desc='Processing elements'))

# Fill the capacitance matrix
for i, P_row in results:
    P[i] = P_row

# Set voltage vector (upper plate: 1V, lower plate: -1V)
V = np.zeros(num_elements)
V[:len(centroids_u)] = 1  # Lower plate
V[len(centroids_u):] = -1  # Upper plate

# Solve for charge distribution
sol = np.linalg.solve(P, V)
sol = sol * areas
print("Capacitance =", sum(sol[V > 0]) / 2)

# Plot the charge distribution
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=sol, cmap='seismic', s=25)
plt.colorbar(sc, label='Charge Distribution')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Charge Distribution on Plates')
plt.show()
