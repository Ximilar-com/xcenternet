import setuptools
from distutils.command.build_ext import build_ext as DistUtilsBuildExt
from setuptools.extension import Extension


# from https://github.com/fizyr/tf-retinanet/blob/master/setup.py
class BuildExtension(setuptools.Command):
    description = DistUtilsBuildExt.description
    user_options = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__["_command"] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy

        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [
    Extension("xcenternet.model.evaluation.overlap", ["xcenternet/model/evaluation/overlap.pyx"]),
]

with open("requirements.txt") as f:
    install_requirements = f.read().splitlines()

setuptools.setup(
    name="xcenternet",
    version="1",
    description="Object detection model for images.",
    long_description="Anchor free, fast object detection model in TensorFlow 2+.",
    author="Libor Vanek (https://github.com/liborvaneksw) and Michal Lukac (https://github.com/Cospel) from Ximilar",
    author_email="tech@ximilar.com",
    license="MIT",
    cmdclass={"build_ext": BuildExtension},
    packages=setuptools.find_packages(),
    ext_modules=extensions,
    setup_requires=["cython>=0.28", "numpy>=1.14.0"],
    install_requires=install_requirements,
    zip_safe=False,
    namespace_packages=["xcenternet"],
)
