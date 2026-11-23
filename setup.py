from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "pothole_rover"
model_files = [
    filename for filename in ("best.pt", "new_model.pt") if Path(filename).is_file()
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data={"pothole_rover": ["web/*.html"]},
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml", *model_files]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/web", glob("pothole_rover/web/*.html")),
    ],
    zip_safe=False,
)
