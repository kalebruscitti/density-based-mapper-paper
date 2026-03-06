import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

def generate_consistent(n_pts, loc, width):
    # Sample a component centered at `loc`, with width `width`
    x,y=loc
    widthx,widthy=width
    x_axis = widthx * np.random.randn(n_pts,) + x
    y_axis = widthy * np.random.randn(n_pts,) + y
    t_axis = 10*np.random.rand(n_pts,)

    cluster_consistent = np.array([
        x_axis,
        y_axis,
        t_axis,
    ])
    return cluster_consistent.T


# Generate three components with differing number of points
cpt1 = generate_consistent(200, (3,0), (0.3,0.2))
cpt2 = generate_consistent(600, (0,0), (0.3,0.2))
cpt3 = generate_consistent(1800, (-3,0), (0.3,0.2))

# Concatenate the arrays to save as a .npy file
cluster_merge = np.concatenate((cpt1,cpt2,cpt3),axis=0)
cluster_merge = np.squeeze(cluster_merge)
intended_clusters = [0]*200 + [1]*600 + [2]*1800
np.save( "./density_demo_data.npy", cluster_merge)

# Generate Figure 4 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(
    cpt1[:,0],
    cpt1[:,1],
    cpt1[:,2],
    c='black',
    label="Low-density",
    marker="x",
)
ax.scatter(
    cpt2[:,0],
    cpt2[:,1],
    cpt2[:,2],
    c='red',
    label="Medium-density",
    marker='.'
)
ax.scatter(
    cpt3[:,0],
    cpt3[:,1],
    cpt3[:,2],
    c='Blue',
    label="High-density",
    marker="1",
)

ax.set_ylim(-3,3)
ax.set_xlabel("x-axis")
ax.set_xlim(-3,3)
ax.set_ylabel("y-axis")
ax.set_zlim(0,10)
ax.set_zlabel("z-axis")
ax.legend()
plt.show()