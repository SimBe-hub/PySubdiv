from setuptools import setup, find_packages
version = '0.1'

setup(
    name='PySubdiv',
    version=version,
    packages=find_packages(),
    install_requires=['networkx>=2.6.3'
            'numpy>=1.21.4',
            'pyacvd>=0.2.7',
            'pyswarms>=1.3.0',
            'pyvista>=0.32.1',
            'scipy>=1.7.2',
            'pymeshlab',
            'pyvistaqt',
            'PyQt5',
            'meshio'],
    url='',
    license='',
    author='Simon Bernard,  S. Mohammad Moulaeifard ',
    author_email='simon.bernard@rwth-aachen.de, Mohammad.Moulaeifard@cgre.rwth-aachen.de',
    description='An Open-source, Python-based software for fitting subdivision surfaces'
)


