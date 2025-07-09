import yaml
import json

conda_list = []
pip_list = []

file_object = {}

with open("req.yaml", "r") as f:
	file_object = yaml.safe_load(f.read())

for dep in file_object["dependencies"]:
	if type(dep) == dict:
		pip_list = dep["pip"]
	else:
		conda_list.append(dep)

conda_prefix = "conda install -c conda-forge "
pip_prefix = "pip install "

conda_prep = " ".join(conda_list)
pip_prep = " ".join(pip_list)

print(conda_prefix + conda_prep)
print("\n\n")
print(pip_prefix + pip_prep)