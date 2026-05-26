import argparse
import math
import mmap
import numpy as np


def load_digits_mod(path: str, ell: int) -> np.ndarray:
    """
    Load comma-separated residues from a text file.
    Assumes residues are single ASCII digits, which is valid for ell in {2,3,5}.
    Commas/newlines are ignored.
    """
    if ell not in (2, 3, 5):
        raise ValueError("This script is intended for ell in {2,3,5}.")

    lo = ord("0")
    hi = ord("0") + ell - 1

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            buf = np.frombuffer(mm, dtype=np.uint8)
            mask = (buf >= lo) & (buf <= hi)
            digits = (buf[mask] - lo).astype(np.uint8)
        finally:
            try:
                del buf
            except UnboundLocalError:
                pass
            mm.close()
    return digits


def blocks_to_indices(digits: np.ndarray, m: int, ell: int) -> np.ndarray:
    """Encode non-overlapping length-m blocks as base-ell integers."""
    L = (len(digits) // m) * m
    if L == 0:
        raise ValueError(f"The sequence is too short to form blocks of length m={m}.")
    blocks = digits[:L].reshape(-1, m).astype(np.int64, copy=False)
    pows = ell ** np.arange(m - 1, -1, -1, dtype=np.int64)
    return blocks @ pows


def chi2_uniform_test(digits: np.ndarray, ell: int, m: int):
    from scipy.stats import chi2 as chi2_dist
    idx = blocks_to_indices(digits, m, ell)
    M = ell ** m
    counts = np.bincount(idx, minlength=M).astype(np.float64)
    B = counts.sum()
    lam = B / M
    chi2_stat = np.sum((counts - lam) ** 2 / lam)
    df = M - 1
    return {
        "ell": ell, "m": m, "blocks": int(B), "pattern_space": int(M),
        "lambda": float(lam), "chi2": float(chi2_stat), "df": int(df),
        "p_value": float(chi2_dist.sf(chi2_stat, df)),
    }


def missing_word_test(digits: np.ndarray, ell: int, m: int):
    """Missing-word statistic with exact occupancy expectation/variance."""
    idx = blocks_to_indices(digits, m, ell)
    M = ell ** m
    counts = np.bincount(idx, minlength=M)
    B = int(counts.sum())
    obs_zeros = int(np.count_nonzero(counts == 0))
    lam = B / M

    q1 = (1.0 - 1.0 / M) ** B
    q2 = (1.0 - 2.0 / M) ** B if M > 1 else 0.0
    exp_zeros = M * q1
    var_zeros = M * q1 * (1.0 - q1) + M * (M - 1) * (q2 - q1 * q1)
    if var_zeros < 0 and abs(var_zeros) < 1e-7:
        var_zeros = 0.0
    z = (obs_zeros - exp_zeros) / math.sqrt(var_zeros) if var_zeros > 0 else float("nan")

    return {
        "ell": ell, "m": m, "blocks": B, "pattern_space": int(M),
        "lambda": float(lam), "obs_zeros": obs_zeros,
        "exp_zeros": float(exp_zeros), "var_zeros": float(var_zeros),
        "z_score": float(z),
    }


def make_control_sequence(control: str, length: int, ell: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if control == "periodic":
        return (np.arange(length, dtype=np.int64) % ell).astype(np.uint8)
    if control == "biased":
        p0 = 1.0 / ell + 0.01
        prest = (1.0 - p0) / (ell - 1)
        probs = np.array([p0] + [prest] * (ell - 1), dtype=np.float64)
        return rng.choice(ell, size=length, p=probs).astype(np.uint8)
    if control in ("digit_sum", "thue_morse", "thue_morse_type"):
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        out = np.empty(length, dtype=np.uint8)
        chunk = 2_000_000
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            n = np.arange(start, end, dtype=np.uint64)
            b = n.view(np.uint8).reshape(-1, 8)
            pc = lut[b].sum(axis=1, dtype=np.uint16)
            out[start:end] = (pc % ell).astype(np.uint8)
        return out
    raise ValueError("control must be one of: periodic, biased, digit_sum")


def sample_blocks_for_c2st(digits: np.ndarray, m: int, samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = (len(digits) // m) * m
    blocks = digits[:L].reshape(-1, m)
    nblocks = blocks.shape[0]
    samples = min(samples, nblocks)
    idx = rng.choice(nblocks, size=samples, replace=False)
    return blocks[idx].copy()


def run_c2st(digits, ell, m, samples, model_name, epochs, batch_size, lr, seed, control=None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import roc_auc_score, accuracy_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    if control is None:
        X_real = sample_blocks_for_c2st(digits, m, samples, seed=seed)
    else:
        ctrl = make_control_sequence(control, length=len(digits), ell=ell, seed=seed)
        X_real = sample_blocks_for_c2st(ctrl, m, samples, seed=seed)

    X_fake = np.random.randint(0, ell, size=(X_real.shape[0], m), dtype=np.uint8)
    y_real = np.ones((X_real.shape[0],), dtype=np.int64)
    y_fake = np.zeros((X_fake.shape[0],), dtype=np.int64)

    X = np.concatenate([X_real, X_fake], axis=0)
    y = np.concatenate([y_real, y_fake], axis=0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    n = len(y)
    n_train = int(n * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    class DilatedCNN(nn.Module):
        def __init__(self, m, vocab, emb=16, channels=64):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb)
            self.convs = nn.ModuleList([
                nn.Conv1d(emb, channels, kernel_size=3, padding=1, dilation=1),
                nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
                nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
                nn.Conv1d(channels, channels, kernel_size=3, padding=8, dilation=8),
                nn.Conv1d(channels, channels, kernel_size=3, padding=16, dilation=16),
            ])
            self.act = nn.ReLU()
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, 2))
        def forward(self, x):
            x = self.emb(x).transpose(1, 2)
            for conv in self.convs:
                x = self.act(conv(x))
            return self.head(x)

    class SmallTransformer(nn.Module):
        def __init__(self, m, vocab, d_model=32, nhead=4, num_layers=2):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.pos = nn.Parameter(torch.zeros(1, m, d_model))
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.cls = nn.Linear(d_model, 2)
        def forward(self, x):
            x = self.emb(x) + self.pos[:, :x.size(1), :]
            x = self.enc(x)
            return self.cls(x.mean(dim=1))

    if model_name == "cnn":
        model = DilatedCNN(m=m, vocab=ell).to(device)
    elif model_name == "transformer":
        model = SmallTransformer(m=m, vocab=ell).to(device)
    else:
        raise ValueError("model must be 'cnn' or 'transformer'")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        print(f"[ell={ell}, m={m}] epoch {ep}/{epochs} train_loss={total / n_train:.6f}")

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p1 = torch.softmax(model(xb), dim=1)[:, 1].detach().cpu().numpy()
            probs.append(p1)
            ys.append(yb.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    pred = (probs >= 0.5).astype(np.int64)

    return {
        "ell": ell, "m": m, "samples_each": int(X_real.shape[0]),
        "model": model_name, "auc": float(roc_auc_score(ys, probs)),
        "accuracy": float(accuracy_score(ys, pred)), "control": control,
    }


def default_chi2_m_list(ell: int):
    return {2: [1, 5, 10, 14, 17], 3: [1, 5, 8, 11], 5: list(range(1, 9))}[ell]


def default_missing_m_list(ell: int):
    return {2: [18, 20, 22, 24, 25], 3: [12, 13, 14, 15, 16], 5: [9, 10, 11]}[ell]


def default_c2st_m_list(ell: int):
    return {2: [50], 3: [32], 5: [22]}[ell]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ell", type=int, required=True, choices=[2, 3, 5], help="modulus ell")
    ap.add_argument("--data", required=True, help="path to pmod{ell}.txt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap1 = sub.add_parser("chi2", help="Dense regime: Pearson chi-square test")
    ap1.add_argument("--m-list", type=int, nargs="+", default=None)

    ap2 = sub.add_parser("missing", help="Intermediate regime: missing-word statistic")
    ap2.add_argument("--m-list", type=int, nargs="+", default=None)

    ap3 = sub.add_parser("c2st", help="Sparse regime: classifier two-sample test")
    ap3.add_argument("--m-list", type=int, nargs="+", default=None)
    ap3.add_argument("--samples", type=int, default=1_000_000, help="samples per class")
    ap3.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    ap3.add_argument("--epochs", type=int, default=2)
    ap3.add_argument("--batch-size", type=int, default=4096)
    ap3.add_argument("--lr", type=float, default=2e-3)
    ap3.add_argument("--seed", type=int, default=0)
    ap3.add_argument("--controls", nargs="+", default=None, choices=["target", "periodic", "biased", "digit_sum"])

    args = ap.parse_args()
    ell = args.ell
    print(f"Loading digits from: {args.data}")
    print(f"ell = {ell}")
    digits = load_digits_mod(args.data, ell)
    print("digits length:", len(digits))

    if args.cmd == "chi2":
        m_list = args.m_list if args.m_list is not None else default_chi2_m_list(ell)
        for m in m_list:
            r = chi2_uniform_test(digits, ell, m)
            print(f"[CHI2] ell={r['ell']} m={r['m']} blocks={r['blocks']:,} M={ell}^{m}={r['pattern_space']:,} "
                  f"lambda={r['lambda']:.6f} chi2={r['chi2']:.6f} df={r['df']:,} p={r['p_value']:.8g}")

    elif args.cmd == "missing":
        m_list = args.m_list if args.m_list is not None else default_missing_m_list(ell)
        for m in m_list:
            r = missing_word_test(digits, ell, m)
            print(f"[MISSING] ell={r['ell']} m={r['m']} blocks={r['blocks']:,} M={ell}^{m}={r['pattern_space']:,} "
                  f"lambda={r['lambda']:.6f} zeros(obs)={r['obs_zeros']:,} zeros(exp)={r['exp_zeros']:.3f} Z={r['z_score']:.6f}")

    elif args.cmd == "c2st":
        m_list = args.m_list if args.m_list is not None else default_c2st_m_list(ell)
        controls = args.controls if args.controls is not None else ["target"]
        for m in m_list:
            for c in controls:
                control_arg = None if c == "target" else c
                r = run_c2st(digits, ell, m, args.samples, args.model, args.epochs, args.batch_size, args.lr, args.seed, control_arg)
                tag = f"control={r['control']}" if r["control"] else f"target=p(n) mod {ell}"
                print(f"[C2ST] ell={r['ell']} m={r['m']} {tag} model={r['model']} samples_each={r['samples_each']:,} "
                      f"AUC={r['auc']:.6f} ACC={r['accuracy']:.6f}")


if __name__ == "__main__":
    main()
