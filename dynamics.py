import numpy as np
import matplotlib.pyplot as plt

def inertia_tensor_rect(a, b, c, m):
  return m/12 * np.array([[b**2 + c**2, 0, 0],
                          [0, a**2 + c**2, 0],
                          [0, 0, a**2 + b**2]])

# satellite body
a_body = 2.7
b_body = 4.3
c_body = 0.25
m_body = 750
J_body = inertia_tensor_rect(a_body, b_body, c_body, m_body)
r_body = np.array([0, 0, 0])

# solar panel
a_panel = 12.8
b_panel = 4.1
c_panel = 0.02
m_panel = 10
J_panel = inertia_tensor_rect(a_panel, b_panel, c_panel, m_panel)
r_panel1 = np.array([-8.85, 0, -0.56])
r_panel2 = np.array([8.85, 0, -0.56])

# connecting beam
a_beam = 30.5
b_beam = 1
c_beam = 0.1
m_beam = 30
J_beam = inertia_tensor_rect(a_beam, b_beam, c_beam, m_beam)
r_beam = np.array([0, 0, -0.5])

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

def plot_axes(com, R):
  # Rotate and translate the axes
  rotated_axes = R + com
  
  # Plot the axes
  plt.quiver(com[0], com[1], com[2], R[0, 0], R[1, 0], R[2, 0], color='r', label='X-axis')
  plt.quiver(com[0], com[1], com[2], R[0, 1], R[1, 1], R[2, 1], color='g', label='Y-axis')
  plt.quiver(com[0], com[1], com[2], R[0, 2], R[1, 2], R[2, 2], color='b', label='Z-axis')

ax = plt.figure().add_subplot(111, projection='3d')
plot_rect_prism(r_body, np.eye(3), a_body, b_body, c_body)
plot_rect_prism(r_panel1, np.eye(3), a_panel, b_panel, c_panel)
plot_rect_prism(r_panel2, np.eye(3), a_panel, b_panel, c_panel)
plot_rect_prism(r_beam, np.eye(3), a_beam, b_beam, c_beam)
plot_axes(r_body, np.eye(3))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.axis('equal')
ax.set_title('Satellite Components')
plt.show()