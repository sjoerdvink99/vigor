from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = ''
LONG_DESCRIPTION = ''

setup(
    name='vigor',
    version=VERSION,
    author='Sjoerd Vink, Remco Chang, and Michael Behrisch',
    author_email='s.a.vink@uu.nl',
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['networkx'],
    keywords=['graphs', 'visualization recommendation'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)