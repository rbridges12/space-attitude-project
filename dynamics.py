from plotting import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def inertia_tensor_rect(a, b, c, m):
  return m/12 * np.array([[b**2 + c**2, 0, 0],
                          [0, a**2 + c**2, 0],
                          [0, 0, a**2 + b**2]])

def inertia_in_parallel_axis(J, r, m):
  return J + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

def generate_surfaces(bodies):
  centroids = []
  normals = []
  areas = []
  for body in bodies:
    a, b, c, m, r = body
    # Define the vertices of the rectangular prism
    vertices = np.array([[0, 0, 0],
                         [a, 0, 0],
                         [a, b, 0],
                         [0, b, 0],
                         [0, 0, c],
                         [a, 0, c],
                         [a, b, c],
                         [0, b, c]]) - np.array([a/2, b/2, c/2]) + r
    
    # Define the faces of the rectangular prism
    faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]]
    
    for face in faces:
      v1 = vertices[face[1]] - vertices[face[0]]
      v2 = vertices[face[3]] - vertices[face[0]]
      normal = np.cross(v1, v2)
      area = np.linalg.norm(normal) / 2
      normal /= np.linalg.norm(normal)
      centroid = np.mean(vertices[face], axis=0)

      if np.dot(normal, centroid - r) < 0:
        normal = -normal

      # # plot face for debugging
      # ax.add_collection3d(Poly3DCollection([vertices[face]], facecolors='y', linewidths=1, edgecolors='r', alpha=0.5))
      # # plot normal for debugging
      # ax.quiver(centroid[0], centroid[1], centroid[2], normal[0] * area, normal[1] * area, normal[2] * area, color='c', label='Surface Normal')
      # plt.axis('equal')
      # # plt.show()
      
      centroids.append(centroid)
      normals.append(normal)
      areas.append(area)
  return np.array(centroids), np.array(normals), np.array(areas)

# satellite body
a_body = 2.7
b_body = 4.3
c_body = 0.25
m_body = 750
J_body = inertia_tensor_rect(a_body, b_body, c_body, m_body)
r_body = np.array([0, 0, 0])
body = (a_body, b_body, c_body, m_body, r_body)

# solar panel
a_panel = 12.8
b_panel = 4.1
c_panel = 0.02
m_panel = 10
J_panel = inertia_tensor_rect(a_panel, b_panel, c_panel, m_panel)
r_panel1 = np.array([-8.85, 0, -0.56])
r_panel2 = np.array([8.85, 0, -0.56])
panel1 = (a_panel, b_panel, c_panel, m_panel, r_panel1)
panel2 = (a_panel, b_panel, c_panel, m_panel, r_panel2)

# connecting beam
a_beam = 30.5
b_beam = 1
c_beam = 0.1
m_beam = 30
J_beam = inertia_tensor_rect(a_beam, b_beam, c_beam, m_beam)
r_beam = np.array([0, 0, -0.5])
beam = (a_beam, b_beam, c_beam, m_beam, r_beam)

# total inertia tensor
r_com = (m_body * r_body + m_panel * r_panel1 + m_panel * r_panel2 + m_beam * r_beam) / (m_body + 2 * m_panel + m_beam)

r_body_com = r_body - r_com
r_panel1_com = r_panel1 - r_com
r_panel2_com = r_panel2 - r_com
r_beam_com = r_beam - r_com
J_total = inertia_in_parallel_axis(J_body, r_body_com, m_body) + \
          inertia_in_parallel_axis(J_panel, r_panel1_com, m_panel) + \
          inertia_in_parallel_axis(J_panel, r_panel2_com, m_panel) + \
          inertia_in_parallel_axis(J_beam, r_beam_com, m_beam)
print("Total Inertia Tensor:\n", J_total)

# surfaces 
bodies = [body, panel1, panel2, beam]
centroids, normals, areas = generate_surfaces(bodies)

# orbit parameters
mu = 398600.4418  # Earth's gravitational parameter, km^3/s^2
R_earth = 6371  # Earth's radius, km
altitude = 559  # Altitude of the satellite, km
a_orbit = R_earth + altitude  # Semi-major axis, km
inclination = 43  # degrees
v = np.sqrt(mu / a_orbit)  # Orbital velocity for circular orbit
p0 = np.array([a_orbit, 0, 0])  # Initial position in ECI frame
v0 = np.array([0, v, 0])  # Initial velocity in ECI frame
def R_x(theta):
  c, s = np.cos(theta), np.sin(theta)
  return np.array([[1, 0, 0],
                   [0, c, -s],
                   [0, s, c]])

v0 = R_x(np.radians(inclination)) @ v0  # Rotate velocity vector to achieve desired inclination

def satellite_dynamics(t, x):
  p = x[0:3]
  v = x[3:6]
  r = np.linalg.norm(p)
  
  # Gravitational acceleration
  a = -mu * p / r**3
  return np.concatenate((v, a))

def hat(x):
  return np.array([[0, -x[2], x[1]],
                   [x[2], 0, -x[0]],
                   [-x[1], x[0], 0]])

def attitude_dynamics(t, x):
  q = x[0:4]  # Quaternion
  w = x[4:7]  # Angular velocity
  temp = q[0] * np.eye(3) + hat(q[1:4])
  L = np.zeros((4, 4));
  L[0, :] = [q[0], -q[1], -q[2], -q[3]]
  L[1:,0] = q[1:4]
  L[1:,1:] = temp
  H = np.zeros((4,3))
  H[1:, :] = np.eye(3)
  qdot = 0.5 * L @ H @ w
  tau = 0
  wdot = np.linalg.inv(J_total) @ (tau - hat(w) @ J_total @ w)
  return np.concatenate((qdot, wdot))

def rk4_step(func, t, x, dt):
  k1 = func(t, x)
  k2 = func(t + dt/2, x + dt/2 * k1)
  k3 = func(t + dt/2, x + dt/2 * k2)
  k4 = func(t + dt, x + dt * k3)
  return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_step_attitude(func, t, x, dt):
  xn = rk4_step(func, t, x, dt)
  xn[0:4] /= np.linalg.norm(xn[0:4]) # normalize quaternion
  return xn

def simulate_attitude():
  N = 1000
  xs = np.zeros((N, 7))
  q0 = np.array([1, 0, 0, 0])  # initial quaternion (no rotation)
  rpm = 10
  w0 = 2*np.pi*rpm/60 * np.array([0, 0, 1]) + np.random.normal(0, 0.05, 3)  # initial angular velocity with some noise
  print("Initial Angular Velocity (rad/s):", w0)
  xs[0] = np.concatenate((q0, w0))
  dt = 0.05
  ts = np.arange(0, N*dt, dt)
  for i, t in enumerate(ts[1:], 1):
    xs[i] = rk4_step_attitude(attitude_dynamics, t, xs[i-1], dt)
  
  animate_attitude_body(ts, xs, bodies, r_com, interval=10)

def simulate_orbit():
  N = 1000
  xs = np.zeros((N, 6))
  xs[0] = np.concatenate((p0, v0))
  dt = 10  # time step of 10 seconds
  ts = np.arange(0, N*dt, dt)
  for i, t in enumerate(ts[1:], 1):
    xs[i] = rk4_step(satellite_dynamics, t, xs[i-1], 10)

  # animate_orbit(xs, ts, R_earth)
  animate_orbit_with_velocity(xs, ts, R_earth, history_len=600, vel_scale=100, interval=10)

def plot_body():
  # plotting
  ax = plt.figure().add_subplot(111, projection='3d')
  plot_rect_prism(r_body, np.eye(3), a_body, b_body, c_body)
  plot_rect_prism(r_panel1, np.eye(3), a_panel, b_panel, c_panel)
  plot_rect_prism(r_panel2, np.eye(3), a_panel, b_panel, c_panel)
  plot_rect_prism(r_beam, np.eye(3), a_beam, b_beam, c_beam)
  plot_axes(r_com, np.eye(3))
  plot_axes(r_body, np.eye(3))
  plot_inertia_tensor(J_total, r_com, np.eye(3))
  plot_surface_normals(centroids, normals, areas)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.axis('equal')
  ax.set_title('Satellite Components')
  # ax.legend()
  plt.show()

if __name__ == "__main__":
  simulate_attitude()