from setuptools import setup, find_packages

setup(name='navigation_simulator_2d',
      version='0.0.0',
      author='Kohei Honda',
      author_email='honda.kohei.b0@s.mail.nagoya-u.ac.jp',
      description='Simple Navigation Simulator 2D with LiDAR model',
      python_requires='>=3.8.10',
      
      install_requires=[
            'pymap2d>=0.1.15',
            'scikit-image',
            'matplotlib',
      ],
      dependency_links=[
      ],
      packages=find_packages())