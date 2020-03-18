from setuptools import setup


# Get version
exec(open('swing/version.py', 'r').read())
#
setup(
    name='swing',
    version=__version__,
    author='Yisheng Qiu',
    license='MIT',
    packages=['swing'],
)
