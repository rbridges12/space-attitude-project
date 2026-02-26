import numpy as np
import matplotlib.pyplot as plt

def plot_rect_prism(com, R, a, b, c):
  # Define the vertices of the rectangular prism
  vertices = np.array([[-a/2, -b/2, -c/2],
                       [a/2, -b/2, -c/2],
                       [a/2, b/2, -c/2],
                       [-a/2, b/2, -c/2],
                       [-a/2, -b/2, c/2],
                       [a/2, -b/2, c/2],
                       [a/2, b/2, c/2],
                       [-a/2, b/2, c/2]])
  
  # Rotate and translate the vertices
  rotated_vertices = (R @ vertices.T).T + com
  
  # Define the edges of the rectangular prism
  edges = [[0, 1], [1, 2], [2, 3], [3, 0],
           [4, 5], [5, 6], [6, 7], [7, 4],
           [0, 4], [1, 5], [2, 6], [3, 7]]
  
  # Plot the edges of the rectangular prism
  for edge in edges:
    plt.plot(rotated_vertices[edge][:, 0], rotated_vertices[edge][:, 1], rotated_vertices[edge][:, 2], 'k-')

def plot_axes(p, R):
  # Rotate and translate the axes
  rotated_axes = R + p
  
  # Plot the axes
  plt.quiver(p[0], p[1], p[2], R[0, 0], R[1, 0], R[2, 0], color='r', label='X-axis')
  plt.quiver(p[0], p[1], p[2], R[0, 1], R[1, 1], R[2, 1], color='g', label='Y-axis')
  plt.quiver(p[0], p[1], p[2], R[0, 2], R[1, 2], R[2, 2], color='b', label='Z-axis')

def plot_inertia_tensor(J, p, R):
  # Eigenvalue decomposition of the inertia tensor
  eigenvalues, eigenvectors = np.linalg.eig(J)
  eigenvalues = np.real(eigenvalues) / 1000
  
  # plot the scaled eigenvectors as axes
  for i in range(3):
    plt.quiver(p[0], p[1], p[2], eigenvectors[0, i] * eigenvalues[i], eigenvectors[1, i] * eigenvalues[i], eigenvectors[2, i] * eigenvalues[i], color='m', label=f'Inertia Axis {i+1}')

def plot_surface_normals(centroids, normals, areas):
  scales = areas / np.max(areas) * 0.5  # Scale normals by area for better visualization
  for i in range(len(centroids)):
    plt.quiver(centroids[i][0], centroids[i][1], centroids[i][2], normals[i][0] * scales[i], normals[i][1] * scales[i], normals[i][2] * scales[i], color='c', label='Surface Normal')