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

# function to animate satellite orbital position and velocity trajectory given state vectors, times, and earth radius
def animate_orbit(xs, times, earth_radius):
  from matplotlib.animation import FuncAnimation
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  # Plot Earth as a sphere
  u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
  x = earth_radius * np.cos(u) * np.sin(v)
  y = earth_radius * np.sin(u) * np.sin(v)
  z = earth_radius * np.cos(v)
  ax.plot_surface(x, y, z, color='b', alpha=0.5)
  
  # Plot satellite trajectory
  trajectory, = ax.plot([], [], [], 'r-', label='Satellite Trajectory')
  satellite, = ax.plot([], [], [], 'ro', label='Satellite Position')
  def update(frame):
    trajectory.set_data(xs[:frame, 0], xs[:frame, 1])
    trajectory.set_3d_properties(xs[:frame, 2])
    satellite.set_data(xs[frame, 0], xs[frame, 1])
    satellite.set_3d_properties(xs[frame, 2])
    return trajectory, satellite
  
  ani = FuncAnimation(fig, update, frames=len(times), blit=True)
  plt.legend()
  plt.show()

def animate_orbit_with_velocity(xs, times, earth_radius, history_len=100, vel_scale=None, interval=10):
    """
    Animate satellite orbit.
    xs: (N,6) array of state vectors [x,y,z, vx,vy,vz]
    times: (N,) array of times (same units as velocities/positions)
    history_len: number of prior points to show with fading alpha
    vel_scale: optional scale factor for velocity arrow (auto-derived if None)
    interval: ms between frames (auto-derived from times if None)
    """
    from matplotlib.animation import FuncAnimation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Earth
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    x = earth_radius * np.cos(u) * np.sin(v)
    y = earth_radius * np.sin(u) * np.sin(v)
    z = earth_radius * np.cos(v)
    ax.plot_surface(x, y, z, color='b', alpha=0.25, zorder=0)

    positions = xs[:, :3]
    velocities = xs[:, 3:6]

    # axis limits
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = (maxs - mins).max() * 1.2
    ax.set_xlim(center[0] - span/2, center[0] + span/2)
    ax.set_ylim(center[1] - span/2, center[1] + span/2)
    ax.set_zlim(center[2] - span/2, center[2] + span/2)
    ax.set_box_aspect([1,1,1])

    # draw ECI axes at origin (scaled to scene)
    axis_len = span * 0.4
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', linewidth=1.5)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='g', linewidth=1.5)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='b', linewidth=1.5)
    ax.text(axis_len*1.05, 0, 0, 'ECI X', color='r')
    ax.text(0, axis_len*1.05, 0, 'ECI Y', color='g')
    ax.text(0, 0, axis_len*1.05, 'ECI Z', color='b')

    # autoscale vel arrow if not provided
    if vel_scale is None:
        vel_mag = np.linalg.norm(velocities, axis=1).max() + 1e-12
        vel_scale = span * 0.2 / vel_mag

    # initial artists
    history_artist = [None]     # will hold the history scatter (recreated each frame)
    sat_point = ax.scatter([], [], [], c='r', s=40)
    vel_q = [None]  # mutable container so update() can replace

    def update(frame):
        pos = positions[frame]
        vel = velocities[frame]
        start = max(0, frame - history_len)
        hist = positions[start:frame + 1]
        n = len(hist)

        # remove previous history artist
        if history_artist[0] is not None:
            try:
                history_artist[0].remove()
            except Exception:
                pass
            history_artist[0] = None

        # recreate history scatter with explicit RGBA colors (works reliably in 3D)
        if n:
            colors = np.zeros((n, 4))
            colors[:, 0] = 1.0   # red
            colors[:, 3] = np.linspace(0, 0.9, n)  # fading alpha
            history_artist[0] = ax.scatter(hist[:, 0], hist[:, 1], hist[:, 2],
                                           c=colors, s=3, depthshade=True)
        else:
            history_artist[0] = None

        # update satellite point
        sat_point._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        # update velocity arrow (remove old and draw new)
        if vel_q[0] is not None:
            try:
                vel_q[0].remove()
            except Exception:
                pass
        vel_q[0] = ax.quiver(pos[0], pos[1], pos[2],
                             vel[0] * vel_scale, vel[1] * vel_scale, vel[2] * vel_scale,
                             color='g', linewidth=1.5, length=1.0, normalize=False)

        # display current time in title (assumes times in seconds)
        ax.set_title(f"t = {times[frame]:.1f} s")

        # Return artists that changed (history may be None)
        artists = [a for a in (history_artist[0], sat_point, vel_q[0]) if a is not None]
        return artists

    ani = FuncAnimation(fig, update, frames=len(times), interval=interval, repeat=False)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()
    return ani