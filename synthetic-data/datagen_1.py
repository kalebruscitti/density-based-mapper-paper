import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

# The data will be generated in three pieces
# - The loop 
# - The rest of the component containing the loop
# - The second component

# Generate the loop:
angles = np.random.randn(2000,)
circle = np.array([
    0.6*np.cos(2*np.pi*angles), 1.2*np.sin(2*np.pi*angles)+2
]).T
circle += 0.1*np.random.randn(2000,2) 
left=np.amin(circle[:,0])+0.13
right=np.amax(circle[:,0])-0.13

# Generate the rest of the loop's component
handle_leftx = np.linspace(-3,left,2500)+ 0.1*np.random.randn(2500,)
handle_rightx = np.linspace(right,3,2500)+ 0.1*np.random.randn(2500,)
handle_x = np.hstack((handle_leftx,handle_rightx))
handle = np.array([
    handle_x, 0.3*np.random.randn(5000,)+2
]).T
dense = np.vstack((handle, circle)) # combine the dense component into one array

# Generate the sparse component
sparse_x = np.linspace(-3,3,200)
sparse_x += 0.1*np.random.randn(200,)
sparse = np.array([
   sparse_x, 0.5*np.random.randn(200,)-3
]).T

# Combine arrays and save as .npy
points = np.vstack((dense, sparse))
np.save("./genus1_demo.npy",points)

# Generate Figure
fig, ax = plt.subplots(1)
ax.set_title("Generated Data")
ax.scatter(
    points[:,0], points[:,1],
    marker='.',
    s=1,
)
ax.set_xlabel(r'Lens-Space $f(x)$')
ax.set_ylabel(r'Semantic space ($\mathbb{R}^n)$')
plt.show(fig)