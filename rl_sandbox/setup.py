from setuptools import setup, find_packages

setup(name='rl_sandbox',
      version='2.0.0+lfgp',
      packages=[package for package in find_packages()
                if package.startswith('rl_sandbox')],
      install_requires=['gym==0.21',
                        'numpy==1.23.1',
                        'tensorboard==2.10.0',
                        'torch==1.13',
                        'manipulator_learning @ git+ssh://git@github.com/utiasSTARS/manipulator_learning@master#egg=manipulator_learning']
      )
