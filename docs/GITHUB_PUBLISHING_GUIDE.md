# GitHub Publishing Guide

This guide explains how to publish the Health AI Federated Learning Client repository professionally without uploading private data, local databases, trained model weights, or build artifacts.

## What Should Be Uploaded

Upload source code and documentation:

- `app.py`
- `core/`
- `ui/`
- `server/`
- `assets/`
- `config/`
- `docs/`
- `tests/`
- `installer/HospitalFLSystem.iss`
- `build_exe.bat`
- `HospitalFLSystem.spec`
- `requirements.txt`
- `README.md`
- `README_BUILD.md`

## What Should Not Be Uploaded

Do not upload generated or private files:

- `.venv/`
- `.idea/`
- `data/raw/`
- `data/predictions/`
- `data/visualizations/`
- `database/*.db`
- `models/`
- `server/storage/`
- `reports/`
- `exports/`
- `build/`
- `dist/`
- `installer/Output/`
- trained checkpoints such as `.pt`, `.pth`, `.ckpt`
- installer files such as `.exe`, `.msi`, `.zip`

The `.gitignore` file is configured to exclude these files.

## Recommended Repository Name

Use a clear academic repository name, for example:

```text
health-ai-federated-learning-client
```

## Step 1: Install Git

Download and install Git for Windows:

```text
https://git-scm.com/download/win
```

Restart PyCharm or PowerShell after installing Git.

## Step 2: Initialize Git Locally

Open PowerShell inside the project folder:

```powershell
cd C:\Users\User\PycharmProjects\hospital_fl_client
git init
git branch -M main
```

## Step 3: Check What Will Be Uploaded

Run:

```powershell
git status --short
```

Make sure the following are not staged:

- `.venv/`
- `build/`
- `dist/`
- `data/raw/`
- `database/hospital_client.db`
- `models/`
- `exports/`

## Step 4: Stage Files

Run:

```powershell
git add .
git status --short
```

Review the staged files before committing. This is the most important safety step.

## Step 5: Create The First Commit

Run:

```powershell
git commit -m "Initial academic prototype release"
```

If Git asks for your name/email:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Then run the commit command again.

## Step 6: Create The GitHub Repository

Go to GitHub and create a new repository:

```text
https://github.com/new
```

Recommended settings:

- Repository name: `health-ai-federated-learning-client`
- Visibility: public if you want to showcase it, private if your supervisor requires privacy
- Do not add a README on GitHub because this project already has one
- Do not add `.gitignore` on GitHub because this project already has one
- Do not add a license unless you are sure which license you want

## Step 7: Connect Local Project To GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/health-ai-federated-learning-client.git
git push -u origin main
```

## Step 8: Verify The GitHub Page

After pushing, check the repository page and confirm:

- README displays correctly
- docs are visible
- source folders are present
- no dataset images are uploaded
- no `.db` database file is uploaded
- no `.pt` model checkpoint is uploaded
- no `.exe` installer is uploaded
- no `.venv`, `build`, or `dist` folder is uploaded

## Optional: Add A Release Later

After the source code is safely published, you may create a GitHub Release and attach the demo ZIP:

```text
HospitalFLSystem_Demo_Setup.zip
```

Only attach an installer ZIP if you are comfortable publishing the executable demo. Do not include patient data, local databases, or private trained models inside the release.

## Recommended Demo Installer Workflow

The generated Windows installer is too large and too binary-heavy for the normal source repository. Share it as a GitHub Release asset instead.

1. Build the `.exe` application folder with PyInstaller.
2. Build the setup installer with Inno Setup.
3. Run:

```powershell
.\package_demo_zip.bat
```

4. Upload this file to GitHub Releases:

```text
release/HospitalFLSystem_Demo_Setup.zip
```

5. In `README.md`, replace `YOUR_USERNAME` in the demo download link:

```text
https://github.com/YOUR_USERNAME/health-ai-federated-learning-client/releases/latest/download/HospitalFLSystem_Demo_Setup.zip
```

Use the Releases page for large demo files. Keep the Git repository for source code, documentation, tests, and configuration.

## README Preview Image

The README uses this image at the top:

```text
docs/assets/app_preview.svg
```

If you later capture a real screenshot of the app, replace this file or add a new image in `docs/assets/` and update the README image path.

## Academic Presentation Note

This repository should be described as an academic prototype. The README and documentation should not claim clinical validation, production security, or formal privacy guarantees.
