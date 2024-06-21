from setuptools import setup, find_packages

setup(name='rl_sandbox',
      version='3.1.0+vpace',
      packages=[package for package in find_packages()
                if package.startswith('rl_sandbox')],
      install_requires=['gym>=0.15.4,<=0.23.0',
                        'numpy>=1.23.4,<2.0',
                        'tensorboard<=2.11',
                        'torch==1.13.*',
                        'manipulator_learning @ git+ssh://git@github.com/utiasSTARS/manipulator-learning@master#egg=manipulator_learning',
                        'ConfigArgParse',
                        'PyYAML']
      )
