import pymatgen.core as pmg
import random

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
def green(string):
    
    return f"{bcolors.OKGREEN}{string}{bcolors.ENDC}"

# # Integrated symmetry analysis tools from spglib
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# # si = pmg.Element("Si")
# # si.atomic_mass  # 28.0855
# # print(si.melting_point)
# # # 1687.0 K

# comp = pmg.Composition("Fe2O3")
# print(comp.weight)  # 159.6882
# # Note that Composition conveniently allows strings to be treated just like an Element object.
# print(comp["Fe"])  # 2.0
# print(comp.get_atomic_fraction("Fe"))  # 0.4
# lattice = pmg.Lattice.cubic(4.2)
# print(green(lattice))
# structure = pmg.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
# # structure.volume
# # 74.088000000000008
# # structure[0]
# # PeriodicSite: Cs (0.0000, 0.0000, 0.0000) [0.0000, 0.0000, 0.0000]

# # You can create a Structure using spacegroup symmetry as well.
# li2o = pmg.Structure.from_spacegroup(
#     "Fm-3m", pmg.Lattice.cubic(3), ["Li", "O"], [[0.25, 0.25, 0.25], [0, 0, 0]]
# )
# print(f"{green('li2o')} -- {li2o}")
# finder = SpacegroupAnalyzer(structure)
# print(f"{green('finder.get_space_group_symbol()')} -- {finder.get_space_group_symbol()}")
# # "Pm-3m"

# # Convenient IO to various formats. You can specify various formats.
# # Without a filename, a string is returned. Otherwise,
# # the output is written to the file. If only the filename is provided,
# # the format is intelligently determined from a file.
# structure.to(fmt="poscar")
# structure.to(filename="POSCAR")
# structure.to(filename="CsCl.cif")

# # Reading a structure is similarly easy.
# structure = pmg.Structure.from_str(open("CsCl.cif").read(), fmt="cif")
# structure = pmg.Structure.from_file("CsCl.cif")
# print(f"{green('structure')} -- {structure}")

# from pymatgen.io.cif import CifParser
# from mayavi import mlab
# import numpy as np

# def visualize_structure(cif_path):
#     # CIF 파일을 파싱하여 구조 객체 생성
#     parser = CifParser(cif_path)
#     structure = parser.get_structures()[0]
    
#     # 구조의 모든 원자 위치를 얻기
#     frac_coords = structure.frac_coords
#     lattice = structure.lattice.matrix
    
#     # 분수 좌표를 실제 좌표로 변환
#     coords = np.dot(frac_coords, lattice)
    
#     # 3D 시각화
#     mlab.figure(bgcolor=(1, 1, 1))
#     for i, site in enumerate(structure):
#         # 각 원자에 대해 구의 형태로 시각화
#         mlab.points3d(coords[i, 0], coords[i, 1], coords[i, 2], scale_factor=0.5, resolution=20,
#                       color=(random.random(), random.random(), random.random()))
#     mlab.options.offscreen = True
#     mlab.savefig('/workspace/24prin/test.jpeg')
#     mlab.show()
#     mlab.options.offscreen = False
    
    
    
from mp_api.client import MPRester
with MPRester(api_key="3Nv7W8UXZGeljybWkrgZRRURopbO2bqX") as mpr:
    data = mpr.materials.search(material_ids=["mp-1072444"])
    data = data[0]
    # print(data)
    structure = data.structure

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.core.structure import Structure

# pymatgen Structure 객체 생성
# structure = Structure.from_dict(data.structure)

# 시각화를 위한 데이터 추출
atom_positions = [site.coords for site in structure.sites]
elements = [site.species_string for site in structure.sites]

# 3D 그래픽 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 원자를 그래픽으로 추가
for pos, element in zip(atom_positions, elements):
    ax.scatter(pos[0], pos[1], pos[2], label=element)

# 축 레이블 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 그래픽 시각화
plt.savefig("testtest.png")


## ---------------------------------------------------------
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# SpacegroupAnalyzer를 사용하여 분자의 대칭성을 분석합니다.
analyzer = SpacegroupAnalyzer(structure)
symmetrized_structure = analyzer.get_symmetrized_structure()

# 대칭성을 고려하여 분자의 3D 좌표를 가져옵니다.
symmetrized_positions = symmetrized_structure.cart_coords
import py3Dmol

# 분자의 좌표를 가져옵니다.
positions = symmetrized_positions.tolist()

# 3Dmol의 Viewer 객체를 생성합니다.
viewer = py3Dmol.view(width=800, height=400)

# 분자를 추가합니다.
for idx, pos in enumerate(positions):
    x, y, z = pos
    viewer.addAtom({'x': x, 'y': y, 'z': z, 'elem': 'C'}, idx)

# 시각화를 렌더링합니다.
viewer.setStyle({'sphere': {'radius': 0.5}})  # 구 형태로 변경
viewer.setBackgroundColor('white')
viewer.zoomTo()

viewer.show()
# 이미지로 저장합니다.
png = viewer.png()
with open("molecule.png", "wb") as f:
    f.write(viewer.png())
    
# # 분자를 추가합니다.
# for idx, pos in enumerate(positions):
#     x, y, z = pos
#     viewer.addAtom({'x': x, 'y': y, 'z': z, 'elem': 'C'}, idx)

# # 연결된 원자들을 추가합니다.
# for site in symmetrized_structure:
#     neighbors = symmetrized_structure.get_neighbors(site, 2.5)
#     for neighbor in neighbors:
#         idx1 = site.species_string
#         idx2 = neighbor.specie.symbol
#         viewer.addBond(int(site.species_string), int(neighbor.specie.symbol))

# # 시각화를 렌더링합니다.
# # viewer.setStyle({'stick': {}})
# viewer.setStyle({'sphere': {'radius': 0.5}})
# viewer.setBackgroundColor('white')
# viewer.zoomTo()

# # 이미지로 저장합니다.
# img_data = viewer.png()

# # 이미지를 파일로 저장합니다.
# with open("molecule.png", "wb") as f:
#     f.write(img_data)
