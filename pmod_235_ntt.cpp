#include <bits/stdc++.h>
using namespace std;

using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

static constexpr u32 MOD = 3221225473u;
static constexpr u32 G   = 5u;

static inline u32 addmod(u32 a, u32 b) {
    u64 s = (u64)a + b;
    return (s >= MOD) ? (u32)(s - MOD) : (u32)s;
}
static inline u32 submod(u32 a, u32 b) {
    return (a >= b) ? (a - b) : (u32)((u64)a + MOD - b);
}
static inline u32 mulmod(u32 a, u32 b) {
    return (u32)((u128)a * b % MOD);
}

static u32 powmod(u32 a, u64 e) {
    u32 r = 1;
    while (e) {
        if (e & 1) r = mulmod(r, a);
        a = mulmod(a, a);
        e >>= 1;
    }
    return r;
}
static inline u32 invmod(u32 a) { return powmod(a, (u64)MOD - 2); }

static void ntt(vector<u32>& a, bool invert) {
    int n = (int)a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        u32 wlen = powmod(G, (u64)(MOD - 1) / (u64)len);
        if (invert) wlen = invmod(wlen);

        for (int i = 0; i < n; i += len) {
            u32 w = 1;
            int half = len >> 1;
            for (int j = 0; j < half; j++) {
                u32 u = a[i + j];
                u32 v = mulmod(a[i + j + half], w);
                a[i + j]        = addmod(u, v);
                a[i + j + half] = submod(u, v);
                w = mulmod(w, wlen);
            }
        }
    }

    if (invert) {
        u32 inv_n = invmod((u32)n);
        for (u32 &x : a) x = mulmod(x, inv_n);
    }
}

static vector<uint8_t> convolve_mod_ell(const uint8_t* A, int nA,
                                        const uint8_t* B, int nB,
                                        int need, int ell)
{
    int want = nA + nB - 1;
    int n = 1;
    while (n < want) n <<= 1;

    if (n > (1 << 30)) {
        throw runtime_error("NTT length exceeds 2^30; this modulus cannot handle the requested N.");
    }

    static vector<u32> fa, fb;
    fa.assign(n, 0);
    fb.assign(n, 0);

    for (int i = 0; i < nA; i++) fa[i] = (u32)A[i];
    for (int i = 0; i < nB; i++) fb[i] = (u32)B[i];

    ntt(fa, false);
    ntt(fb, false);
    for (int i = 0; i < n; i++) fa[i] = mulmod(fa[i], fb[i]);
    ntt(fa, true);

    vector<uint8_t> res(need);
    for (int i = 0; i < need; i++) res[i] = (uint8_t)(fa[i] % (u32)ell);
    return res;
}

struct StreamWriter {
    ofstream &out;
    int N;
    int next_idx = 0;
    string buf;

    explicit StreamWriter(ofstream &o, int n) : out(o), N(n) {
        buf.reserve(1 << 20);
    }

    void flush() {
        if (!buf.empty()) {
            out.write(buf.data(), (streamsize)buf.size());
            buf.clear();
        }
    }

    void write_upto(const vector<uint8_t>& b, int m) {
        int upto = min(m - 1, N);
        while (next_idx <= upto) {
            if (next_idx > 0 && (next_idx % 1'000'000 == 0)) {
                cerr << "[progress] computed+written up to n=" << next_idx << "\n";
            }
            out_digit(b[next_idx]);
            if (next_idx != N) buf.push_back(',');
            next_idx++;
            if (buf.size() >= (1 << 20)) flush();
        }
    }

    void out_digit(uint8_t x) {
        // ell is only 2, 3, or 5, so one digit is enough.
        buf.push_back(char('0' + x));
    }

    void finish() { flush(); }
};

static vector<uint8_t> inverse_series_mod_ell_stream(
    const vector<uint8_t>& a, int n, int ell,
    const function<void(const vector<uint8_t>&, int)>& on_stage
) {
    vector<uint8_t> b(1, 1);
    int m = 1;
    on_stage(b, m);

    while (m < n) {
        int m2 = min(2 * m, n);

        auto ab = convolve_mod_ell(a.data(), m2, b.data(), m, m2, ell);

        vector<uint8_t> t(m2, 0);
        for (int i = 0; i < m2; i++) {
            int val = (i == 0) ? 2 : 0;
            int x = val - (int)ab[i];
            x %= ell;
            if (x < 0) x += ell;
            t[i] = (uint8_t)x;
        }

        auto bnew = convolve_mod_ell(b.data(), m, t.data(), m2, m2, ell);
        b.swap(bnew);
        m = m2;

        cerr << "[stage] Newton expanded; final coefficients up to n=" << (m - 1) << "\n";
        on_stage(b, m);
    }
    return b;
}

static void usage(const char* prog) {
    cerr << "Usage: " << prog << " <ell: 2|3|5> [N] [output.txt]\n";
    cerr << "Examples:\n";
    cerr << "  " << prog << " 2 100000000 pmod2.txt\n";
    cerr << "  " << prog << " 3 100000000 pmod3.txt\n";
    cerr << "  " << prog << " 5 100000000 pmod5.txt\n";
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    int ell = stoi(argv[1]);
    if (!(ell == 2 || ell == 3 || ell == 5)) {
        cerr << "Error: this program supports only ell = 2, 3, or 5.\n";
        return 1;
    }

    int N = 100'000'000;
    if (argc >= 3) N = stoi(argv[2]);
    if (N < 0) return 0;

    string outPath;
    if (argc >= 4) {
        outPath = argv[3];
    } else {
        outPath = "pmod" + to_string(ell) + ".txt";
    }

    vector<uint8_t> A(N + 1, 0);
    A[0] = 1;
    for (long long k = 1;; k++) {
        long long g1 = k * (3 * k - 1) / 2;
        long long g2 = k * (3 * k + 1) / 2;
        if (g1 > N) break;
        uint8_t coef = (k & 1) ? (uint8_t)(ell - 1) : (uint8_t)1; // -1 or +1 mod ell
        A[(int)g1] = coef;
        if (g2 <= N) A[(int)g2] = coef;
    }
    cerr << "[info] A(q) built modulo " << ell << " up to N=" << N << "\n";

    ofstream out(outPath, ios::binary);
    if (!out) {
        cerr << "Failed to open output file: " << outPath << "\n";
        return 1;
    }
    StreamWriter writer(out, N);

    try {
        auto P = inverse_series_mod_ell_stream(A, N + 1, ell,
            [&](const vector<uint8_t>& b, int m) {
                writer.write_upto(b, m);
            }
        );

        writer.write_upto(P, (int)P.size());
        writer.finish();
        out.close();
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    cerr << "[done] wrote p(0)..p(" << N << ") modulo " << ell
         << " to " << outPath << "\n";
    return 0;
}
