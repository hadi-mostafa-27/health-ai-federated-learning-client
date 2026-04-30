# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.lib.modulegraph import util as modulegraph_util

block_cipher = None
project_root = Path.cwd()

_original_iterate_instructions = modulegraph_util.iterate_instructions


def _safe_iterate_instructions(code_object):
    try:
        yield from _original_iterate_instructions(code_object)
    except IndexError:
        # Python 3.10.0 can fail while disassembling bytecode from large
        # optional third-party modules during analysis. Hidden imports below
        # keep the application modules explicit while allowing the build to
        # continue on this interpreter.
        return


modulegraph_util.iterate_instructions = _safe_iterate_instructions


def include_dir(name):
    path = project_root / name
    return [(str(path), name)] if path.exists() else []


datas = []
for folder in [
    "assets",
    "config",
    "models",
    "data",
    "sample_data",
    "docs",
]:
    datas += include_dir(folder)


hiddenimports = [
    "PIL.Image",
    "cv2",
    "numpy",
    "pandas",
    "sklearn.metrics",
    "sklearn.model_selection",
    "torch",
    "torchvision",
    "torchvision.models",
    "torchvision.transforms",
]

excludes = [
    "dask",
    "pytest",
    "tensorboard",
    "matplotlib",
    "matplotlib.tests",
    "matplotlib.backends",
    "numpy.f2py.tests",
    "pandas.tests",
    "sklearn.tests",
    "torch.utils.tensorboard",
    "gradio",
]


a = Analysis(
    ["app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HospitalFLSystem",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="HospitalFLSystem",
)
