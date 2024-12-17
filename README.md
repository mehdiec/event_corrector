# NextOn Password Manager

A secure password management system that requires specific environment variables for operation.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone 
 
cd <repository-name>
```


2. Install required dependencies:
```bash
pip install -r requirements.txt
```


## Environment Variables Setup

The application requires two environment variables:
- `nexton_password`: Password to nexton
- `path_public`: Path to isis server public folder

### Windows

```powershell
setx nexton_password "your-master-password"
setx path_public "C:\path\to\public\keys"
```

### Linux/macOS
Add these lines to your `~/.bashrc` (Linux) or `~/.zshrc` (Linux/macOS):
```bash
export nexton_password="your-master-password"
export path_public="C:\path\to\public\keys"
```
Then reload your shell configuration:
```bash
source ~/.bashrc
```
or
```bash
source ~/.zshrc
```

## Security Notice

Never share your `nexton_password` or store it in version control. 

## Usage
Make sure you have the environment variables set.
Make sure you have zarrified your folder usinf pyanimalprocessing.

```bash
python event_corrector/main_outline_zarr.py path_to_zarr
```