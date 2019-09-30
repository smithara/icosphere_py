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
https://github.com/mbrubake/cryoem-cvpr2015/blob/master/quadrature/icosphere.py

https://github.com/brsr/antitile
http://docs.sympy.org/latest/modules/combinatorics/polyhedron.html
https://www.mathworks.com/matlabcentral/fileexchange/50105-icosphere

"""


import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from scipy.spatial import cKDTree as KDTree


def sph2cart(R, t, p):
    # R,t,p are Radius, theta (colatitude), phi (longitude)
        # 0<t<180, 0<p<360
    # Calculate the sines and cosines
    rad = np.pi/180
    s_p = np.sin(p*rad)
    s_t = np.sin(t*rad)
    c_p = np.cos(p*rad)
    c_t = np.cos(t*rad)
    # Calculate the x,y,z over the whole grid
    X = R*c_p*s_t
    Y = R*s_p*s_t
    Z = R*c_t
    return X, Y, Z


def cart2sph(X, Y, Z):
    """Returns r, t, p with t,p in degrees
    """
    rad = np.pi/180
    theta = 90 - np.arctan2(Z, np.sqrt(X**2 + Y**2))/rad
    phi = np.mod(np.arctan2(Y, X)/rad, 360)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    return R, theta, phi


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


def get_edgevecs(vertices, fudge=False):
    """Given a set of vertices, find the neighbouring 5 or 6 vertices to each,
    return the set of vectors between vertices (which define the edges)
    """
    vertices = vertices.copy()
    try:
        # Remove the previous neighbours as they will be recalculated
        vertices = vertices.drop(['neighbours'], axis=1)
    except:
        pass

    kdt = KDTree(list(zip(vertices.x.values,
                          vertices.y.values,
                          vertices.z.values)))
    # Get 7 nearest neighbours for every vertex (includes itself, i.e. dist 0)
    dists, indices = kdt.query(list(zip(vertices.x.values,
                                        vertices.y.values,
                                        vertices.z.values)), k = 7)

    # Add the neighbour vertices to the vertex dataframe
    # 5 for the original icosahedron vertices
    # 6 for the others
    locs_origicos = vertices[vertices.iteration == 0].index.values
    locs_others = vertices[vertices.iteration != 0].index.values
    neighbs5 = pd.DataFrame({'neighbours':indices[:,1:6].tolist()}).loc[locs_origicos]
    neighbs6 = pd.DataFrame({'neighbours':indices[:,1:7].tolist()}).loc[locs_others]
    neighbs = pd.concat([neighbs5,neighbs6])
    vertices = vertices.join(neighbs)

#    # New dataframe with the previous iteration's vertices as centres of faces
#    faces = vertices[vertices.iteration < vertices.iteration.max()]
#    faces['corners'] = np.empty((faces.shape[0]),dtype=list)
#    faces['corners'][:] = []
    #faces['corners'] =
    # Set up all the edge vectors from each vertex's neighbour sets
    # E = 3V-6  number of edges, E, from number of vertices, V
    if not fudge:
        edgevecs = np.zeros((3*vertices.shape[0]-6, 3, 2))
    else:
        edgevecs = np.zeros((9*vertices.shape[0], 3, 2))
    k = 0 # loop counter through edgevecs
    for i in range(vertices.shape[0]):
        # i runs from 0 to V
        # Coordinates of point i:
        x0,y0,z0 = vertices.loc[i].x, vertices.loc[i].y, vertices.loc[i].z

        for j in vertices.loc[i].neighbours:
            # Coordinates of each  neighbour:
            x1,y1,z1 = vertices.loc[j].x, vertices.loc[j].y, vertices.loc[j].z

#            # Add face corners if we are on a face centre
#            if i in faces.index.values:
#                faces['corners'].loc[i].append([x1,y1,z1])

            # Check if p1->p0 already exists in a previous p0->p1
            # https://stackoverflow.com/a/33218744
            if not (edgevecs == np.array([[x1,x0],[y1,y0],[z1,z0]])).all((1,2)).any():
                # Store the vectors
                edgevecs[k] = np.array([[x0,x1],[y0,y1],[z0,z1]])
                k+=1

    x0 = edgevecs[:,0,0]
    x1 = edgevecs[:,0,1]
    y0 = edgevecs[:,1,0]
    y1 = edgevecs[:,1,1]
    z0 = edgevecs[:,2,0]
    z1 = edgevecs[:,2,1]
    edgevecs = pd.DataFrame({'x0':x0,'x1':x1,'y0':y0,'y1':y1,'z0':z0,'z1':z1})
    if fudge:
        edgevecs = edgevecs.dropna().reset_index().drop(columns="index")

    return edgevecs, vertices


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

    def _get_edgevecs(self, fudge=False):
        self.edgevecs, self.vertices = get_edgevecs(self.vertices, fudge=fudge)

    def rotate(self, delta_phi):
        """Rotate by delta_phi degrees.
        """
        r, t, p = cart2sph(*[self.vertices[i] for i in "xyz"])
        p = (p + delta_phi) % 360
        x, y, z = sph2cart(r, t, p)
        for i, var in zip("xyz", (x, y, z)):
            self.vertices[i] = var
        self._get_edgevecs()
        return self

    def get_faces(self):
        """Construct the triagonal faces.

        There are duplicate faces in what gets returned
        """
        faces = []
        # (p, q, r) are indexes within self.vertices
        for p in self.vertices.index:
            # define all the faces neighbouring point p
            # Loop through the points, q, neighbouring p, and identify
            #   those neighbours, r, of q, which themselves also neighbour p
            for q in self.vertices.loc[p].neighbours:
                # build "face", an array containing points (p, q, r)
                # to define p->q, q->r, r->p
                # [[px, py, pz]
                #  [qx, qy, qz]
                #  [rx, ry, rz]]
                face = np.empty((3, 3))
                face[:] = np.nan
                if q not in self.vertices.index:
                    continue
                face[0] = self.vertices.loc[p][["x", "y", "z"]].values
                face[1] = self.vertices.loc[q][["x", "y", "z"]].values
                for r in self.vertices.loc[q].neighbours:
                    if r not in self.vertices.index:
                        continue
                    if r in self.vertices.loc[p].neighbours:
                        face[2] = self.vertices.loc[r][["x", "y", "z"]].values
                        # break
                faces.append(face)
        return faces

    def get_dualfaces(self, dual=False):
        """CURRENTLY BROKEN

        Construct the hexagonal(+12pentagons) faces from the next subdivision
        Get the sets of vertices that define the corners of the faces
        The faces will be centred on the vertices of the current Polyhedron
        """
        pass

    def _construct_centroid_polygons(self):
        """Constructs pentagons/hexagons around the grid vertices.

        It was meant to be get_dualfaces() but doesn't actually do that...
        """
        newpoly = self.subdivide()
        verts = newpoly.vertices
        facecentres = verts[verts.iteration < verts.iteration.max()]
        # faces = [[] for i in range(facecentres.shape[0])]
        faces = []
        for i in range(facecentres.shape[0]):
            locs_neighbs = facecentres.iloc[i].neighbours
            neighbs = verts.loc[locs_neighbs]
            faces.append(neighbs[['x', 'y', 'z']].values)

        # Reorder the vertices in each face so that they can go into a patch
        # i.e. that they are ordered from one neighbour to the next
        newfaces = []
        for f in faces:
            fnew = np.zeros_like(f)
            fnew[0] = f[0]
            ftemp = np.delete(f, 0, axis=0)
            for i in range(len(ftemp)):
                j = ((ftemp-fnew[i])**2).sum(axis=1).argmin()
                fnew[i+1] = ftemp[j]
                ftemp = np.delete(ftemp, j, axis=0)
            newfaces.append(fnew)
        return newfaces

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
            newvertices.append(slerp(p0, p1))
        newvertices = vertices2dataframe(np.array(newvertices))
        # Set the iteration number on the set of new vertices
        last_iteration = self.vertices["iteration"].max()
        newvertices["iteration"] = last_iteration + 1
        # newvertices['origicos'] = False
        newvertices = pd.concat((newvertices, self.vertices), ignore_index=True)

        newpoly = Polyhedron()
        newpoly.vertices = newvertices

        newpoly._get_edgevecs()
        return newpoly

    def drawedges(self, ax, **kwargs):
        """Draw its edges on a given 3d axis in a given color.
        """
        x0, x1 = self.edgevecs.x0.values, self.edgevecs.x1.values
        y0, y1 = self.edgevecs.y0.values, self.edgevecs.y1.values
        z0, z1 = self.edgevecs.z0.values, self.edgevecs.z1.values
        vecsx = np.array([i for i in zip(x0, x1)])
        vecsy = np.array([i for i in zip(y0, y1)])
        vecsz = np.array([i for i in zip(z0, z1)])
        for v in zip(vecsx, vecsy, vecsz):
            ax.plot(v[0], v[1], v[2], **kwargs)

    def drawverts(self, ax, **kwargs):
        """Draw the vertices as points on a given 3d axis with a given style.
        """
        x = self.vertices.x.values
        y = self.vertices.y.values
        z = self.vertices.z.values
        ax.scatter(x, y, z, depthshade=True, **kwargs)

    def drawfaces(
            self, ax,
            color="r", edgecolor="k", linewidth=0.1, alpha=None,
            **kwargs
            ):
        """Draw the triangular faces.
        """
        for face_coords in self.get_faces():
            face = mplot3d.art3d.Poly3DCollection([face_coords])
            # Bug with setting alpha:
            # https://github.com/matplotlib/matplotlib/issues/10237
            # Must apply alpha first:
            face.set_alpha(alpha)
            face.set_color(color)
            face.set_edgecolor(edgecolor)
            face.set_linewidth(linewidth)
            ax.add_collection3d(face)

    def get_verts_thetaphi(self):
        _, theta, phi = cart2sph(self.vertices.x.values,
                                 self.vertices.y.values,
                                 self.vertices.z.values)
        return theta, phi


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
        self.vertices['iteration'] = 0
        super()._get_edgevecs()
