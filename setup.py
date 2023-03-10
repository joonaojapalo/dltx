from setuptools import setup

with open("README.md") as fd:
    long_description = fd.read()

setup(name='dltx',
      version='0.1.1',
      description='Direct Linear Transform (DLT)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/joonaojapalo/dltx',
      author='Marcos Duarte',
      author_email='duartexyz@gmail.com',
      maintainer='Joona Ojapalo',
      maintainer_email='joona.ojapalo@iki.fi',
      license='Attribution 4.0 International',
      packages=['dltx'],
      install_requires=[
        'numpy'
      ],
      zip_safe=False)
