import os
import time
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
from src.QDDPM_torch import DiffusionModel, QDDPM, naturalDistance

rc("text", usetex=False)
rc("axes", linewidth=3)
device = torch.device("cpu")


# srun -N 1 -n 1 -c 32 --gres=gpu:1 --mem=64G -p L40 --pty bash
# ==========================================
# Utility Functions
# ==========================================
def ensure_dir(file_path):
    """Ensure the directory exists"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def cluster0Gen(n, N_train, scale, seed=None):
    """
    generate random quantum states close to |0...0>
    Args:
    n: number of qubits
    N_train: number of data to generate
    scale: the scaling factor on amplitudes except |0...0>
    seed: control the randomness
    """
    np.random.seed(seed)
    # amplitude for basis except |0...0>
    remains = np.random.randn(N_train, 2**n - 1) + 1j * np.random.randn(
        N_train, 2**n - 1
    )
    states = np.hstack((np.ones((N_train, 1)), scale * remains))  # un-normalized
    states /= np.tile(np.linalg.norm(states, axis=1).reshape((1, N_train)), (2**n, 1)).T
    return states.astype(np.complex64)


def Training_t(model, t, inputs_T, params_tot, Ndata, epochs):
    """
    the trianing for the backward PQC at step t
    """
    input_tplus1 = model.prepareInput_t(inputs_T, params_tot, t, Ndata)  # prepare input
    states_diff = model.states_diff
    loss_hist = []  # record of training history

    # initialize parameters
    # np.random.seed() # Remove specific seed reset to allow randomness or set specific seed if needed
    params_t = torch.tensor(
        np.random.normal(size=2 * model.n_tot * model.L),
        device=device,
        requires_grad=True,
    )
    # set optimizer and learning rate decay
    optimizer = torch.optim.Adam([params_t], lr=0.0005)

    t0 = time.time()
    pbar = tqdm(range(epochs), desc=f"Training Step t={t}", leave=False)
    for step in pbar:
        indices = np.random.choice(states_diff.shape[1], size=Ndata, replace=False)
        true_data = states_diff[t, indices]

        output_t = model.backwardOutput_t(input_tplus1, params_t)
        loss = naturalDistance(output_t, true_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.item())  # record the current loss value

        if step % 10 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

    return params_t.detach().cpu(), torch.tensor(loss_hist)


# ==========================================
# Experiment workflow function
# ==========================================
def run_1qubit_experiment():
    print("Running 1-qubit Cluster Experiment...")

    # 1. Generate Diffusion Data
    n = 1
    T = 20
    Ndata_diff = 1000
    diff_hs = torch.from_numpy(np.linspace(1.0, 4.0, T)).to(device)
    model_diff = DiffusionModel(n, T, Ndata_diff, device=device)

    diff_data_path = f"data/cluster/n{n}/cluster0Diff_n{n}T{T}_N{Ndata_diff}.npy"
    if os.path.exists(diff_data_path):
        print("Loading existing diffusion data...")
        Xout = np.load(diff_data_path)
    else:
        X = torch.from_numpy(cluster0Gen(n, Ndata_diff, 0.08, seed=12)).to(device)
        Xout = np.zeros((T + 1, Ndata_diff, 2**n), dtype=np.complex64)
        Xout[0] = X.cpu().numpy()
        for t in range(1, T + 1):
            Xout[t] = (
                model_diff.set_diffusionData_t(t, X, diff_hs[:t], seed=t).cpu().numpy()
            )
        ensure_dir(diff_data_path)
        np.save(diff_data_path, Xout)
        print(f"Diffusion data saved to {diff_data_path}")

    # 2. Training
    na = 1
    L = 4
    Ndata_train = 100
    epochs = 2001

    # Generate random samples at step t=T for training input base
    diffModel = DiffusionModel(n, T, Ndata_train)
    inputs_T = diffModel.HaarSampleGeneration(Ndata_train, seed=22)

    # Load diffusion process for training target
    states_diff = np.load(diff_data_path)
    model = QDDPM(n=n, na=na, T=T, L=L)
    model.set_diffusionSet(states_diff)

    print("Starting Training...")
    for t in range(T - 1, -1, -1):
        print(f"Training step t={t}")
        params_tot = np.zeros((T, 2 * (n + na) * L))
        # Load previously trained parameters for steps > t
        for tt in range(t + 1, T):
            p_path = f"data/cluster/n{n}/QDDPMcluster0params_n{n}na{na}T{T}L{L}_t{tt}_mmd.npy"
            if os.path.exists(p_path):
                params_tot[tt] = np.load(p_path)

        params, loss_hist = Training_t(
            model, t, inputs_T, params_tot, Ndata_train, epochs
        )

        save_p_path = (
            f"data/cluster/n{n}/QDDPMcluster0params_n{n}na{na}T{T}L{L}_t{t}_mmd.npy"
        )
        save_l_path = (
            f"data/cluster/n{n}/QDDPMcluster0losshist_n{n}na{na}T{T}L{L}_t{t}_mmd.npy"
        )
        ensure_dir(save_p_path)
        np.save(save_p_path, params.detach().numpy())
        np.save(save_l_path, loss_hist.detach().numpy())

    # 3. Collect Results & Generate Test Data
    print("Collecting results and generating test data...")
    params_tot_final = np.zeros((T, 2 * (n + na) * L))
    loss_tot_final = np.zeros((T, epochs))

    for t in range(T):
        params_tot_final[t] = np.load(
            f"data/cluster/n{n}/QDDPMcluster0params_n{n}na{na}T{T}L{L}_t{t}_mmd.npy"
        )
        loss_tot_final[t] = np.load(
            f"data/cluster/n{n}/QDDPMcluster0losshist_n{n}na{na}T{T}L{L}_t{t}_mmd.npy"
        )[
            :epochs
        ]  # Ensure size match

    np.save(
        f"data/cluster/n{n}/QDDPMcluster0params_n{n}na{na}T{T}L{L}_mmd.npy",
        params_tot_final,
    )
    np.save(
        f"data/cluster/n{n}/QDDPMcluster0loss_n{n}na{na}T{T}L{L}_mmd.npy",
        loss_tot_final,
    )

    # Generate Data
    inputs_T_tr = diffModel.HaarSampleGeneration(Ndata_train, seed=22)
    inputs_T_te = diffModel.HaarSampleGeneration(Ndata_train, seed=28)

    data_tr = model.backDataGeneration(inputs_T_tr, params_tot_final, Ndata_train)[
        :, :, : 2**n
    ]
    data_te = model.backDataGeneration(inputs_T_te, params_tot_final, Ndata_train)[
        :, :, : 2**n
    ]

    np.save(
        f"data/cluster/n{n}/QDDPMcluster0trainGen_n{n}na{na}T{T}L{L}_mmd.npy",
        data_tr.detach().numpy(),
    )
    np.save(
        f"data/cluster/n{n}/QDDPMcluster0testGen_n{n}na{na}T{T}L{L}_mmd.npy",
        data_te.detach().numpy(),
    )

    print("1-qubit experiment finished.")


def run_2qubit_experiment():
    print("Running 2-qubit Cluster Experiment...")

    # 1. Generate Diffusion Data
    n = 2
    T = 20
    Ndata_diff = 1000
    diff_hs = np.linspace(0.5, 4.0, T)

    model_diff = DiffusionModel(n, T, Ndata_diff)
    X = torch.from_numpy(cluster0Gen(n, Ndata_diff, 0.06, seed=12))

    Xout = np.zeros((T + 1, Ndata_diff, 2**n), dtype=np.complex64)
    Xout[0] = X.numpy()
    for t in range(1, T + 1):
        Xout[t] = model_diff.set_diffusionData_t(
            t, X, torch.from_numpy(diff_hs[:t]), seed=t
        ).numpy()

    diff_data_path = f"data/cluster/n{n}/cluster0Diff_n{n}T{T}_N{Ndata_diff}.npy"
    ensure_dir(diff_data_path)
    np.save(diff_data_path, Xout)
    print(f"Diffusion data saved to {diff_data_path}")

    # 2. Training
    na = 1
    L = 6
    Ndata_train = 100
    epochs = 2001

    diffModel = DiffusionModel(n, T, Ndata_train)
    inputs_T = diffModel.HaarSampleGeneration(Ndata_train, seed=22)

    states_diff = np.load(diff_data_path)
    model = QDDPM(n=n, na=na, T=T, L=L)
    model.set_diffusionSet(states_diff)

    print("Starting Training...")
    for t in range(T - 1, -1, -1):
        print(f"Training step t={t}")
        params_tot = np.zeros((T, 2 * (n + na) * L))
        for tt in range(t + 1, T):
            p_path = f"data/cluster/n{n}/QDDPMcluster0params_n{n}na{na}T{T}L{L}_t{tt}_mmd.npy"
            if os.path.exists(p_path):
                params_tot[tt] = np.load(p_path)

        params, loss_hist = Training_t(
            model, t, inputs_T, params_tot, Ndata_train, epochs
        )

        save_p_path = (
            f"data/cluster/n{n}/QDDPMcluster0params_n{n}na{na}T{T}L{L}_t{t}_mmd.npy"
        )
        save_l_path = (
            f"data/cluster/n{n}/QDDPMcluster0loss_n{n}na{na}T{T}L{L}_t{t}_mmd.npy"
        )
        ensure_dir(save_p_path)
        np.save(save_p_path, params.detach().numpy())
        np.save(save_l_path, loss_hist.detach().numpy())

    # 3. Collect Results & Generate Test Data
    print("Collecting results and generating test data...")
    params_tot_final = np.zeros((T, 2 * (n + na) * L))
    loss_tot_final = np.zeros((T, epochs))

    for t in range(T):
        params_tot_final[t] = np.load(
            f"data/cluster/n{n}/QDDPMcluster0params_n{n}na{na}T{T}L{L}_t{t}_mmd.npy"
        )
        loss_tot_final[t] = np.load(
            f"data/cluster/n{n}/QDDPMcluster0loss_n{n}na{na}T{T}L{L}_t{t}_mmd.npy"
        )[:epochs]

    np.save(
        f"data/cluster/n{n}/QDDPMcluster0params_n{n}na{na}T{T}L{L}_mmd.npy",
        params_tot_final,
    )
    np.save(
        f"data/cluster/n{n}/QDDPMcluster0loss_n{n}na{na}T{T}L{L}_mmd.npy",
        loss_tot_final,
    )

    inputs_T_tr = diffModel.HaarSampleGeneration(Ndata_train, seed=22)
    inputs_T_te = diffModel.HaarSampleGeneration(Ndata_train, seed=23)

    data_tr = model.backDataGeneration(inputs_T_tr, params_tot_final, Ndata_train)[
        :, :, : 2**n
    ]
    data_te = model.backDataGeneration(inputs_T_te, params_tot_final, Ndata_train)[
        :, :, : 2**n
    ]

    np.save(
        f"data/cluster/n{n}/QDDPMcluster0trainGen_n{n}na{na}T{T}L{L}_mmd.npy",
        data_tr.detach().numpy(),
    )
    np.save(
        f"data/cluster/n{n}/QDDPMcluster0testGen_n{n}na{na}T{T}L{L}_mmd.npy",
        data_te.detach().numpy(),
    )

    print("2-qubit experiment finished.")


def plot_results(n, T, L, Ndata_diff=1000):
    """
    Optional: Function to generate plots similar to the notebook
    """
    print(f"Plotting results for n={n}...")
    states_diff = np.load(f"data/cluster/n{n}/cluster0Diff_n{n}T{T}_N{Ndata_diff}.npy")
    states_train = np.load(
        f"data/cluster/n{n}/QDDPMcluster0trainGen_n{n}na1T{T}L{L}_mmd.npy"
    )
    states_test = np.load(
        f"data/cluster/n{n}/QDDPMcluster0testGen_n{n}na1T{T}L{L}_mmd.npy"
    )

    F0_train = np.abs(states_train[:, :, 0]) ** 2
    F0_test = np.abs(states_test[:, :, 0]) ** 2
    F0_diff = np.abs(states_diff[:, : states_train.shape[1], 0]) ** 2  # Match size

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        range(T + 1),
        np.mean(F0_diff, axis=1),
        "o--",
        markersize=5,
        mfc="white",
        lw=2,
        c="r",
        zorder=5,
        label=r"diffusion",
    )

    baseline = 0.5 if n == 1 else 0.25
    ax.plot(range(T + 1), baseline * np.ones(T + 1), "--", lw=2, c="orange")

    ax.plot(
        range(T + 1),
        np.mean(F0_train, axis=1),
        "o--",
        markersize=5,
        mfc="white",
        lw=2,
        c="b",
        zorder=5,
        label=r"training",
    )
    ax.fill_between(
        range(T + 1),
        np.mean(F0_train, axis=1) - np.std(F0_train, axis=1),
        np.mean(F0_train, axis=1) + np.std(F0_train, axis=1),
        color="b",
        alpha=0.1,
    )

    ax.plot(
        range(T + 1),
        np.mean(F0_test, axis=1),
        "o--",
        markersize=5,
        mfc="white",
        lw=2,
        c="forestgreen",
        zorder=5,
        label=r"testing",
    )
    ax.fill_between(
        range(T + 1),
        np.mean(F0_test, axis=1) - np.std(F0_test, axis=1),
        np.mean(F0_test, axis=1) + np.std(F0_test, axis=1),
        color="forestgreen",
        alpha=0.1,
    )

    ax.legend(fontsize=20, framealpha=0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, T + 1, 5))
    ax.set_xlabel(r"Diffusion steps $t$", fontsize=30)
    ax.set_ylabel(r"$\overline{F_0}$", fontsize=30)
    ax.tick_params(
        direction="in", length=10, width=3, top="on", right="on", labelsize=30
    )

    plot_path = f"data/cluster/n{n}/fidelity_plot_n{n}.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    # run_1qubit_experiment()
    plot_results(n=1, T=20, L=4)
