# HospitalFLSystem Windows Build Guide

This project is packaged with PyInstaller in `--onedir` mode for reliability.

## Step 1: Test in PyCharm / development first

From the project root:

```bat
.venv\Scripts\python.exe app.py
```

Verify login, admin dashboard, hospital dashboard, project requests, approval, FL project runner, results page, Docker export after project completion, and database writes.

## Step 2: Rebuild the Windows executable

```bat
build_exe.bat
```

The executable will be created at:

```text
dist\HospitalFLSystem\HospitalFLSystem.exe
```

Equivalent PyInstaller command:

```bat
pyinstaller --noconfirm --windowed --onedir --name HospitalFLSystem app.py
```

This repository uses `HospitalFLSystem.spec` so hidden imports and data folders are bundled consistently.

## Step 3: Test the rebuilt executable

```bat
dist\HospitalFLSystem\HospitalFLSystem.exe
```

Run through the same workflows again:

- login works
- admin role works
- hospital role works
- request creation works
- request approval works
- approved request appears in FL runner
- selected hospitals and FL settings are preserved
- inactive hospitals appear unavailable and do not participate
- completed projects enable Docker package export
- results page opens
- database read/write works after closing and reopening
- no missing asset/icon/UI file errors
- no terminal window appears

## Step 4: Open Inno Setup Compiler

Install Inno Setup 6 if it is not installed:

```text
https://jrsoftware.org/isdl.php
```

Open Inno Setup Compiler from the Start Menu.

## Step 5: Open the installer script

Open:

```text
installer\HospitalFLSystem.iss
```

## Step 6: Click Compile

In Inno Setup Compiler, click **Compile**.

Command-line alternative:

```bat
iscc installer\HospitalFLSystem.iss
```

## Step 7: Final installer output

The final installer should be generated as:

```text
installer\Output\HospitalFLSystem_Setup.exe
```

## What Inno Setup Does

- PyInstaller creates the executable app folder in `dist\HospitalFLSystem`.
- Inno Setup creates the official Windows installer.
- The installer adds a desktop shortcut, a Start Menu shortcut, and installs the app cleanly.

## Runtime data location

When running from source, files stay in the project folder.

When running as an `.exe`, writable files are stored under:

```text
%APPDATA%\HospitalFLSystem
```

This includes:

- `config/app_config.json`
- `database/hospital_client.db`
- `models/`
- `reports/`
- `logs/`
- `exports/docker/`
- prediction and visualization outputs

This prevents user data from being lost or blocked by Windows install permissions.

## Notes

- Use `--onedir` first. Do not switch to `--onefile` until the onedir executable is fully tested.
- If PyInstaller reports missing hidden imports, add them to `HospitalFLSystem.spec`; avoid changing app logic for packaging-only issues.
