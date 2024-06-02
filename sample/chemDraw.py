from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image

# # ChemDraw에서 내보낸 파일을 읽어들입니다.
# # mol = Chem.MolFromMolFile('molecule.mol')
# mol1 = Chem.MolFromSmiles('CCO') 

# # 분자의 3D 구조를 생성합니다.
# mol = Chem.AddHs(mol1)  # 수소 원자 추가
# AllChem.EmbedMolecule(mol)
# AllChem.MMFFOptimizeMolecule(mol)

# # 분자 구조를 가시화합니다.
# img = Draw.MolToImage(mol)
# # img.show()
# img.save('test.png')

from rdkit import Chem
from rdkit.Chem import Descriptors

# SMILES로 분자 로드
smiles = "CCO"  # 예시 SMILES
mol = Chem.MolFromSmiles(smiles)

# 분자량 계산
mw = Descriptors.MolWt(mol)

# 로지P 계산
logp = Descriptors.MolLogP(mol)

# LogS 계산
# logs = Descriptors.MolLogS(mol)

# TPSA 계산
tpsa = Descriptors.TPSA(mol)

# 수소 수 계산
hbd = Descriptors.NumHDonors(mol)
hba = Descriptors.NumHAcceptors(mol)

print("SMILES:", smiles)
print("Molecular Weight:", mw)
print("LogP:", logp)
# print("LogS:", logs)
print("TPSA:", tpsa)
print("Number of Hydrogen Bond Donors:", hbd)
print("Number of Hydrogen Bond Acceptors:", hba)

