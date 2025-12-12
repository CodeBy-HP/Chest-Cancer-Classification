import setuptools

__version__ = "0.0.0"

REPO_NAME = "Chest-Cancer-Classification"
AUTHOR_USER_NAME = "CodeBy-HP"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "code.by.hp@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)