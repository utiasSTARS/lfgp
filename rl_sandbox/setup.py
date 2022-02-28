from setuptools import setup, find_packages

setup(name='rl_sandbox',
      version='1.0.1-lfgp',
      packages=[package for package in find_packages()
                if package.startswith('rl_sandbox')],
      install_requires=['gym==0.17.2',
                        'numpy==1.19.0',
                        'tensorboard==2.2.2',
                        'torch==1.5.1',
                        'manipulator_learning @ git+ssh://git@github.com/utiasSTARS/manipulator_learning@master#egg=manipulator_learning']
      )
