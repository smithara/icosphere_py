"""

Author: Ashley Smith

Creates an icosphere by repeated subdivision of an icosahedron.
I have naively built it from:
    1 - defining vertex locations
    2 - finding their neighbouring vertices from a nearest-neighbour search
    3 - interpolating half way between vertices and their neighbours to identify the new vertices
    4 - repeating
It would be more efficient to define the faces (triangles), and subdivide them

Alternatives?
https://en.wikipedia.org/wiki/Goldberg%E2%80%93Coxeter_construction
https://en.wikipedia.org/wiki/List_of_geodesic_polyhedra_and_Goldberg_polyhedra
http://donhavey.com/blog/tutorials/tutorial-3-the-icosahedron-sphere/

https://github.com/brsr/antitile
http://docs.sympy.org/latest/modules/combinatorics/polyhedron.html
https://www.mathworks.com/matlabcentral/fileexchange/50105-icosphere

"""


import numpy as np
import pandas as pd


def get_nearest_neighbours(p, N, i):
    """Return the nearest N neighbours to a given point, i

    Args:
        p (DataFrame): vertices dataframe
        N (int): integer for number of nearest neighbours to return
        i (int): loc within dataframe p

    Returns:
        a tuple of locs of the nearest neighbours
    """
    # p_new will be the returned dataframe
    p_new = p.copy()
    # calculate distances to other points
    vecs = p_new[["x", "y", "z"]] - p[["x", "y", "z"]].loc[i]
    dists = vecs.x**2 + vecs.y**2 + vecs.z**2
    # merge distances into the p_new
    dists = dists.to_frame(name='dist2')
    p_new = p_new.join(dists)
    p_new.sort_values(by='dist2', inplace=True)
    return p_new.iloc[1:N+1]


def matchxyz(xyz0, xyz1, xyz0arr, xyz1arr):
    """Returns True if vector xyz0->xyz1 occurs in arrays of vectors xyz0arr->xyz1arr
    """
    for xyz0_, xyz1_ in zip(xyz0arr, xyz1arr):
        if np.array_equal(xyz0, xyz0_) and np.array_equal(xyz1, xyz1_):
            return True
    return False


def get_edgevecs(vertices):
    """Given a set of vertices,
    return the set of vectors between vertices which define the edges
    """
    edgevecs = []
    for i in range(vertices.shape[0]):
        # Coordinates of point i
        x0, y0, z0 = vertices.loc[i].x, vertices.loc[i].y, vertices.loc[i].z
        p = get_nearest_neighbours(vertices, 6, i)
#         print(vertices.loc[i].origicos)
        if vertices.loc[i].origicos:
            Nnearestneighbours = 5
        else:
            Nnearestneighbours = 6
#         print('\n',str(Nnearestneighbours),':\n',p)
        for j in range(Nnearestneighbours):
            # Coordinates of each nearest neighbour
            x1, y1, z1 = p.iloc[j].x, p.iloc[j].y, p.iloc[j].z
            # Store the vectors
            edgevecs.append([[x0, x1], [y0, y1], [z0, z1]])
    ev = np.array(edgevecs)
    x0 = ev[:, 0, 0]
    x1 = ev[:, 0, 1]
    y0 = ev[:, 1, 0]
    y1 = ev[:, 1, 1]
    z0 = ev[:, 2, 0]
    z1 = ev[:, 2, 1]
    evd = pd.DataFrame({'x0': x0, 'x1': x1,
                        'y0': y0, 'y1': y1,
                        'z0': z0, 'z1': z1})
    evdnew = evd.copy()
    # Remove duplicate edges
    for i in range(len(evd)):
        # search for when a matching vector exists, allowing for (x,y,z)0 / (x,y,z)1 swaps
        if matchxyz(evdnew[['x1', 'y1', 'z1']].loc[i].values, evdnew[['x0', 'y0', 'z0']].loc[i].values,
                    evdnew[['x0', 'y0', 'z0']].loc[i+1:].values, evdnew[['x1', 'y1', 'z1']].loc[i+1:].values):
            evdnew.drop(i, inplace=True)

    return evdnew


def slerp(p0, p1):
    """Spherical linear interpolation to halfway between between p0 and p1
    https://en.wikipedia.org/wiki/Slerp
    """
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
#     print(np.dot(p0,p1))
    slerphalfway = (np.sin(omega/2)/np.sin(omega))*(p0+p1)
    return slerphalfway


def vertices2dataframe(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    return pd.DataFrame({'x': x, 'y': y, 'z': z})


class Polyhedron(object):

    def __init__(self):
        self.vertices = pd.DataFrame({'x': [], 'y': [], 'z': []})
        self.edgevecs = pd.DataFrame({'x0': [], 'x1': [], 'y0': [],
                                      'y1': [], 'z0': [], 'z1': []})

    def _set_vertices(self, verts):
        self.vertices.x = verts[:, 0]
        self.vertices.y = verts[:, 1]
        self.vertices.z = verts[:, 2]

    def _get_edgevecs(self):
        self.edgevecs = get_edgevecs(self.vertices)

    def subdivide(self):
        """Take the edge vectors and subdivide them
        to get the new set of vertices, and merge with the first set.
        Return a new polyhedron with set vertices and edge vectors"""
        x0 = self.edgevecs.x0.values
        y0 = self.edgevecs.y0.values
        z0 = self.edgevecs.z0.values
        x1 = self.edgevecs.x1.values
        y1 = self.edgevecs.y1.values
        z1 = self.edgevecs.z1.values
        p0arr = np.vstack([x0, y0, z0]).T
        p1arr = np.vstack([x1, y1, z1]).T
        newvertices = []
        for p0, p1 in zip(p0arr, p1arr):
        #     print(slerp(p0,p1))
            newvertices.append(slerp(p0, p1))
        newvertices = vertices2dataframe(np.array(newvertices))
        newvertices['origicos'] = False
        newvertices = pd.concat((newvertices, self.vertices), ignore_index=True)

        newpoly = Polyhedron()
        newpoly.vertices = newvertices

        newpoly._get_edgevecs()
        return newpoly

    def drawedges(self, ax, color, linewidth):
        """Draw its edges on a given 3d axis in a given color
        """
        x0, x1 = self.edgevecs.x0.values, self.edgevecs.x1.values
        y0, y1 = self.edgevecs.y0.values, self.edgevecs.y1.values
        z0, z1 = self.edgevecs.z0.values, self.edgevecs.z1.values
        vecsx = np.array([i for i in zip(x0, x1)])
        vecsy = np.array([i for i in zip(y0, y1)])
        vecsz = np.array([i for i in zip(z0, z1)])
        for v in zip(vecsx, vecsy, vecsz):
            ax.plot(v[0], v[1], v[2], color, linewidth=linewidth)


class RegIcos(Polyhedron):

    def __init__(self, radius):
        """Set up a regular icosahedron of given radius
        """
        super().__init__()
        self.radius = 100
        norm = radius/(2*np.sin(2*np.pi/5))
        r = (1.0 + np.sqrt(5.0)) / 2.0
        vertices = norm*np.array([
                [-1.0,   r, 0.0],
                [ 1.0,   r, 0.0],
                [-1.0,  -r, 0.0],
                [ 1.0,  -r, 0.0],
                [0.0, -1.0,   r],
                [0.0,  1.0,   r],
                [0.0, -1.0,  -r],
                [0.0,  1.0,  -r],
                [  r, 0.0, -1.0],
                [  r, 0.0,  1.0],
                [ -r, 0.0, -1.0],
                [ -r, 0.0,  1.0],
                ], dtype=float)
        super()._set_vertices(vertices)
        self.vertices['origicos'] = True
        super()._get_edgevecs()
