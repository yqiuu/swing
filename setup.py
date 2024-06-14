from setuptools import setup


# Get version
exec(open('swing/version.py', 'r').read())
#
description = """Swarm intelligence optimization."""

install_requires = [
    'numpy>=1.17',
    'tqdm',
]

setup(
    name='swing-opt',
    version=__version__,
    author='Yisheng Qiu',
    description=description,
    url='https://github.com/yqiuu/swing',
    install_requires=install_requires,
    license='MIT',
    packages=['swing'],
)
