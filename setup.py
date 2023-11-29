import os


from pathlib import Path
from setuptools import setup, find_namespace_packages

README_TEXT = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

MAINTAINER = "Roman Medvedev"
MAINTAINER_EMAIL = "medvedev.daff@gmail.com"


setup(
    name = 'telegram_bot',
    version=os.getenv('PACKAGE_VERSION', '0.0.dev0'),
    long_description=README_TEXT,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    package_dir={'': 'src'},
    license="MIT",
    packages=find_namespace_packages('src'),
)
