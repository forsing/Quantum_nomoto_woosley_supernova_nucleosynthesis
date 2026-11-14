# Quantum_nomoto_woosley_supernova_nucleosynthesis



import random
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_2981.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
BURN_WINDOW = 5
SHOCK_ENERGY_GRID = (0.35, 0.75, 1.25)
ALPHA_CAPTURE_GRID = (0.20, 0.50)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < BURN_WINDOW + 35:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def build_nuclear_reaction_network(
    draws_train: np.ndarray,
    alpha_capture_rate: float,
) -> np.ndarray:
    h_sub = np.zeros((MAX_NUM, MAX_NUM), dtype=np.complex128)

    for k in range(MAX_NUM):
        shell_mass = float(k + 1) / float(MAX_NUM)
        h_sub[k, k] = -1.2 * ((shell_mass - 0.55) ** 2)

        if k + 4 < MAX_NUM:
            h_sub[k, k + 4] += alpha_capture_rate
            h_sub[k + 4, k] += alpha_capture_rate

        if k + 2 < MAX_NUM:
            h_sub[k, k + 2] += 0.5 * alpha_capture_rate
            h_sub[k + 2, k] += 0.5 * alpha_capture_rate

    trans_counts = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
    n_tr = len(draws_train)
    decay_w = np.exp(np.linspace(-1.4, 0.0, num=n_tr - 1, dtype=np.float64))

    for t in range(n_tr - 1):
        w_t = float(decay_w.__getitem__(t))
        src_zero = draws_train.__getitem__(t) - 1
        dst_zero = draws_train.__getitem__(t + 1) - 1
        for u in src_zero:
            u_int = int(u)
            for v in dst_zero:
                v_int = int(v)
                if u_int != v_int:
                    trans_counts[u_int, v_int] += w_t

    sym_trans = 0.5 * (trans_counts + trans_counts.T)
    max_t = float(np.max(sym_trans))
    if max_t > 1e-12:
        h_sub += 0.85 * (sym_trans / max_t)

    h_sub = 0.5 * (h_sub + np.conj(h_sub.T))
    mean_diag = np.trace(h_sub) / float(MAX_NUM)
    h_sub = h_sub - mean_diag * np.eye(MAX_NUM, dtype=np.complex128)
    spec_norm = float(np.linalg.norm(h_sub, ord=2))
    if spec_norm > 1e-12:
        h_sub = h_sub / spec_norm

    h_net = np.zeros((STATE_DIM, STATE_DIM), dtype=np.complex128)
    h_net[:MAX_NUM, :MAX_NUM] = h_sub
    for k in range(MAX_NUM, STATE_DIM):
        h_net[k, k] = 2.0 + 0.12 * float(k - MAX_NUM)
    return h_net


def build_supernova_shock_unitary(
    h_net: np.ndarray,
    shock_energy: float,
) -> np.ndarray:
    u_raw = expm(-1j * shock_energy * h_net)
    u_svd, _, vh_svd = np.linalg.svd(u_raw)
    return u_svd @ vh_svd


def build_shell_initial_states(draws: np.ndarray) -> np.ndarray:
    n_total = len(draws)
    n_all_win = n_total - BURN_WINDOW + 1

    shell_weights = np.exp(np.linspace(-1.1, 0.0, num=BURN_WINDOW, dtype=np.float64))
    shell_weights = shell_weights / np.sum(shell_weights)
    shell_phases = np.exp(1j * np.arange(BURN_WINDOW, dtype=np.float64) * np.pi / float(BURN_WINDOW))

    init_states = np.zeros((n_all_win, STATE_DIM), dtype=np.complex128)
    for idx in range(n_all_win):
        win = draws[idx : idx + BURN_WINDOW] - 1
        psi = np.zeros(STATE_DIM, dtype=np.complex128)
        for s in range(BURN_WINDOW):
            amp = np.sqrt(float(shell_weights.__getitem__(s))) * shell_phases.__getitem__(s)
            psi[win.__getitem__(s)] += amp
        norm_p = float(np.linalg.norm(psi))
        init_states[idx] = psi / max(norm_p, 1e-12)

    return init_states


def simulate_onion_shell_nucleosynthesis(
    init_states_hist: np.ndarray,
    u_shock: np.ndarray,
) -> np.ndarray:
    evolved_states = np.einsum("ij,nj->ni", u_shock, init_states_hist)
    abundances = np.abs(evolved_states[:, :MAX_NUM]) ** 2
    return abundances / np.maximum(np.sum(abundances, axis=1, keepdims=True), 1e-12)


def extract_latest_supernova_yield_qiskit(
    latest_init_state: np.ndarray,
    u_shock: np.ndarray,
) -> np.ndarray:
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(UnitaryGate(u_shock, label="SNShock"), list(range(NUM_QUBITS)))
    sv_latest = Statevector(latest_init_state).evolve(qc).data
    latest_abundance = np.abs(sv_latest[:MAX_NUM]) ** 2
    return latest_abundance / max(float(np.sum(latest_abundance)), 1e-12)


def evaluate_nucleosynthesis_nll(
    abundances_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    prob_mat = 0.88 * abundances_eval + 0.12 * (1.0 / float(MAX_NUM))
    prob_mat = prob_mat / np.sum(prob_mat, axis=1, keepdims=True)

    n_eval = len(y_eval)
    target_mask = np.zeros((n_eval, MAX_NUM), dtype=bool)
    row_idx = np.arange(n_eval)[:, None]
    target_mask[row_idx, y_eval - 1] = True

    log_p = np.log(np.maximum(prob_mat, 1e-12))
    nll_per_draw = -np.sum(np.where(target_mask, log_p, 0.0), axis=1)
    return float(np.mean(nll_per_draw))


def decode_peak_isotope_yields(abundance_vec: np.ndarray) -> list[int]:
    top_indices = np.argsort(abundance_vec)[::-1][:NUM_POSITIONS]
    combo = sorted([int(idx) + 1 for idx in top_indices])
    if len(set(combo)) != NUM_POSITIONS:
        raise RuntimeError("Neuspesno dekodiranje 7 vrsnih izotopskih prinosa supernove.")
    return combo


def main() -> None:
    print("Metoda: Nomoto-Woosley Core-Collapse Supernova Nucleosynthesis Regressor (1x6 qubita)")
    print("Posebnost: Nuklearna mreza alfa-zahvata i kolapsa jezgra + SVD UnitaryGate na simulatoru")

    for file_path in FILES:
        if not file_path.exists():
            print(f"Fajl nije pronadjen: {file_path}")
            continue

        draws = load_draws(file_path)
        n_total = len(draws)
        n_samples = n_total - BURN_WINDOW

        init_all = build_shell_initial_states(draws)
        init_hist = init_all[:n_samples]
        latest_init = init_all.__getitem__(n_samples)
        y_all = draws[BURN_WINDOW:].copy()

        train_end = BURN_WINDOW + int(0.75 * n_samples)
        val_end = BURN_WINDOW + int(0.88 * n_samples)
        val_slice_start = train_end - BURN_WINDOW
        val_slice_end = val_end - BURN_WINDOW

        best_shock = float(SHOCK_ENERGY_GRID.__getitem__(0))
        best_alpha = float(ALPHA_CAPTURE_GRID.__getitem__(0))
        best_val_nll = float("inf")

        for alpha_c in ALPHA_CAPTURE_GRID:
            h_tr = build_nuclear_reaction_network(draws[:train_end], alpha_capture_rate=alpha_c)
            for shock_e in SHOCK_ENERGY_GRID:
                u_tr = build_supernova_shock_unitary(h_tr, shock_energy=shock_e)
                abund_all = simulate_onion_shell_nucleosynthesis(init_hist, u_tr)

                val_nll = evaluate_nucleosynthesis_nll(
                    abund_all[val_slice_start:val_slice_end],
                    y_all[val_slice_start:val_slice_end],
                )
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_shock = shock_e
                    best_alpha = alpha_c

        h_tr_best = build_nuclear_reaction_network(draws[:train_end], alpha_capture_rate=best_alpha)
        u_tr_best = build_supernova_shock_unitary(h_tr_best, shock_energy=best_shock)
        abund_eval_all = simulate_onion_shell_nucleosynthesis(init_hist, u_tr_best)
        test_nll = evaluate_nucleosynthesis_nll(
            abund_eval_all[val_slice_end:],
            y_all[val_slice_end:],
        )

        h_full = build_nuclear_reaction_network(draws, alpha_capture_rate=best_alpha)
        u_full = build_supernova_shock_unitary(h_full, shock_energy=best_shock)
        next_abundance = extract_latest_supernova_yield_qiskit(latest_init, u_full)
        next_combo = decode_peak_isotope_yields(next_abundance)

        print(f"CSV: {file_path.name}")
        print(f"Validacija NLL (shock={best_shock}, alpha_cap={best_alpha}): {best_val_nll:.4f}")
        print(f"Test NLL: {test_nll:.4f}")
        print(f"NEXT: {next_combo}")


if __name__ == "__main__":
    main()



"""
Metoda: Nomoto-Woosley Core-Collapse Supernova Nucleosynthesis Regressor (1x6 qubita)
Posebnost: Nuklearna mreza alfa-zahvata i kolapsa jezgra + SVD UnitaryGate na simulatoru

CSV: loto7_4718_k90.csv
Validacija NLL (shock=1.25, alpha_cap=0.2): 27.3384
Test NLL: 27.2864
NEXT: [6, 7, 9, 15, 17, 23, 38]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (shock=1.25, alpha_cap=0.5): 27.3181
Test NLL: 27.4576
NEXT: [6, 7, 9, 14, 19, 23, 29]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (shock=1.25, alpha_cap=0.5): 27.4924
Test NLL: 27.3260
NEXT: [3, 6, 13, 17, 21, 26, 37]
"""



"""
2026 winners are: 
Ken'ichi Nomoto and Stanford Woosley in astronomy; 
Anne Dejean, Hugues de Thé and Zhu Chen in life science and medicine; 
and Emmanuel Candès and Camillo De Lellis in mathematical sciences.


Dobitnici Shaw Prize 2026 — svaku radim kao zaseban, potpuno drugačiji matematičko-kvantni mehanizam na 6 qubita.

1. Oblast 1 (Mathematical Sciences — Emmanuel Candès & Camillo De Lellis):
	• Metoda: Compressed Sensing (\(\ell _{1}\) Sparse Basis Pursuit) & Convex Integration Oscillatory Perturbations na 1x6 qubita.
	
		• Svako Loto kolo od 7 brojeva u prostoru 1-39 je 7-sparse signal (ima tačno 7 nenultih komponenti od 39).
		• Kvantno kolo na 6 qubita (QFTGate + nekomutativne fazne rotacije) formira mernu matricu \(\Phi \in \mathbb{C}^{64 \times 39}\) (Restricted Isometry Property — Candès).
		• Rekonstrukcija sledećeg stanja vrši se ortogonalnom iterativnom \(\ell _{1}\) projekcijom (ISTA / Basis Pursuit Denoising) u kombinaciji sa De Lellis-ovim visoko-frekventnim konveksnim oscilacijama na aktivnom talasnom frontu poslednjih kola.
		• NEXT kombinacija dobija se direktnim \(\ell _{1}\)-sparse izborom 7 najjačih amplituda rekonstruisanog retkog vektora \(\hat{x} \in \mathbb{R}^{39}\) (bez ikakve pozicione matrice 7×39, bez pos_bias i bez DP lanca!).

2. Oblast 2 (Astronomy — Ken'ichi Nomoto & Stanford Woosley):
	• Metoda: Core-Collapse Supernova Nucleosynthesis & Stellar Shell Reaction Network na 1x6 qubita.
	
        • Kvantni evolucioni operator nuklearnih reakcija između ljuski (He → C → Ne → O → Si → Fe) u 39 izotopskih kanala i kolaps jezgra sa udarnim talasom (shock-revival).

3. Oblast 3 (Life Science & Medicine — Anne Dejean, Hugues de Thé & Zhu Chen):
	• Metoda: PML-RARA Oncoprotein Degradation & Epigenetic Chromatin Remodeling Dynamical System.
	
        • Bistabilni nelinearni dinamički sistem diferencijacije i ciljane degradacije kompleksa na 6-qubitnom simulatoru.
"""
