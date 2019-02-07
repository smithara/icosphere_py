# icosphere_py

![Base icosahedron and three subdivisions](https://raw.githubusercontent.com/smithara/icosphere_py/master/icospheres.png)

Generating a spherical grid (currently only the vertices, which linked would form triangular faces, i.e. spherical triangular tessellation) based on repeated subdivision of a regular icosahedron. It's done in an inefficient way so it's slow (about 10 minutes to do 6 subdivisions).

Pip installation:
```
pip install git+https://github.com/smithara/icosphere_py.git
```

Quick start:
```Python
from icosphere_py.shapes import RegIcos
icos = RegIcos(100)
icos2 = icos.subdivide()
```

See Jupyter notebook, `demo.ipynb`, for more.

If you only want the generated shape coordinates, download `icosphere_data.h5`, and load with:
```Python
import pandas as pd
df = pd.read_hdf('icosphere_data.h5', '40962')
theta = df["theta"]
phi = df["phi"]
```
`theta` and `phi` are the spherical coordinates (in degrees) of each vertex after 6 subdivisions; there are 40962 vertices. Use one of the values of V below to get one of the other levels.

Number of subdivisions, k, gives number of vertices, V: V = 2 + 10*2^(2k) <br>
k: V <br>
0: 12 <br>
1: 42 <br>
2: 162 <br>
3: 642 <br>
4: 2562 <br>
5: 10242 <br>
6: 40962 <br>


