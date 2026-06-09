# PPS2 Timing Thermal Simulation Stack

Modular thermal simulations for PPS2 Timing using GMSH for meshing, FEniCSx for solving, and ParaView for visualization.

## Environment Setup (macOS / Linux via Miniforge)

Because FEniCSx requires complex C++ math libraries (MPI, PETSc) to solve finite element equations, standard `pip` cannot install it natively. **Conda** is the recommended tool because it handles both Python packages and system-level C++ binaries.

For maximum compatibility (especially on Intel Macs), we use **Miniforge**, the open-source, community-driven version of Conda.

### 1. Install Miniforge

**On macOS (Intel or Apple Silicon):**
The easiest method is using Homebrew:
```bash
brew install miniforge
```
*Alternatively, download the installer from the [Miniforge GitHub](https://github.com/conda-forge/miniforge).*

**Important First-Time Setup:**
After installing Miniforge for the first time, you must initialize it so your shell recognizes the `activate` command. Run the following command and then **restart your terminal**:
```bash
conda init
```
*(On modern macOS, you may specifically want to run `conda init zsh`).*

### 2. Create the Environment

We use the `environment.yml` file to guarantee all system and Python dependencies are installed correctly. Open your terminal in the repository folder and run:

```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate thermal_env
```

*(Note: Whenever you want to run these simulations in a new terminal window, you must run `conda activate thermal_env` first).*

### 3. CI/CD Environment (GitHub Actions)

The repository includes a GitHub Action that uses the identical `environment.yml` file to automatically test the mesh generation on every push, ensuring your code remains stable.

## Usage

Run the Simple ETROC simulation using the CLI parameters.

**Basic Run (Default parameters):**

```bash
python simulate_simple_etroc.py
```

**Custom Run (Scanning thicknesses, changing power, enabling convection):**

```bash
python simulate_simple_etroc.py --thickness 0.2 0.5 1.0 --power 2.5 --heatsink-temp 25.0 --convection --air-temp 22.0
```

This will output `.xdmf` and `.h5` files into an `output/` directory. You can download [ParaView](https://www.paraview.org/download/) to open the `.xdmf` files natively and visualize the 3D temperature gradients.
