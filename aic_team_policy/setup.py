from setuptools import find_packages, setup

package_name = "aic_team_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="naveen",
    maintainer_email="naveen@example.com",
    description="Team ACT policy for AIC submission",
    license="Apache-2.0",
)
