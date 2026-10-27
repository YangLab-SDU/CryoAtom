"""
Sequence assignment of the traced nucleic-acid fragments. Every fragment is aligned to every RNA/DNA record by
forward-backward over a semi-global affine alignment; the posterior decides the record position and the base of each
residue, and the fragments are coupled through the base pairs, chain links and record occupancy of the model written
before alignment. Fragments without an admissible record are left to the profile HMM of hmm_sequence_align.
"""
from collections import namedtuple

import numpy as np
from scipy.spatial import cKDTree

from CryoAtom2.utils.residue_constants import index_to_restype_3, num_prot, restype_name_to_atomc_names

NEG = -1e30
LETTER_INDEX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
IUPAC = {"R": "AG", "Y": "CU", "K": "GU", "M": "AC", "S": "CG", "W": "AU", "B": "CGU", "D": "AGU", "H": "ACU",
         "V": "ACG", "N": "ACGU", "X": "ACGU", "-": "ACGU"}
COMPLEMENT = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.float64)  # Watson-Crick and G-U
CLASS_OFFSET = {"DNA": num_prot, "RNA": num_prot + 4}  # 28-class index = offset + letter index (A, C, G, U/T)
P_ATOM, O3_ATOM = 1, 8
LOG_BG = np.log(0.25)

WITHIN_CLASS_WEIGHT = 0.5         # weight of the evidence within the purine / pyrimidine class
GAP_OPEN, GAP_EXTEND = -5.0, -1.5  # unassigned residue or skipped record position inside a fragment
END_OPEN, END_EXTEND = -1.0, -0.5  # unassigned residues before the first / after the last matched residue
JUMP_COST = -6.0                  # re-entering the record at a uniformly chosen position (repeated entities)
SPLIT_GAP = 10                    # a chain is cut where its record position steps back or skips more than this
MIN_MATCHED = 4                   # a record is admissible with >= min(MIN_MATCHED, S) matched residues
MIN_POST = 0.5                    # a residue is numbered only if its MAP position holds this much posterior mass
PAIR_PRECISION = 0.85             # P(a detected pair is a true complementary pair)
PAIR_C1 = (9.8, 11.2)             # C1'-C1' distance window of a base pair (A)
PAIR_P = (17.0, 20.5)             # P-P distance window of a base pair (A)
OCCUPANCY_WEIGHT = 0.5            # penalty per extra residue assigned to a record position
LINK_PRECISION = 0.8              # P(a detected link joins sequence-adjacent fragments)
LINK_MAX_DIST = 8.0               # O3'-P distance of a link between fragment ends (A)
LINK_PRIOR = [0.6, 0.3, 0.1]      # over 0, 1, 2 missing nucleotides between linked fragments
ICM_SWEEPS = 3
INS_COEF = (-4.3692, 2.4512)      # logit of P(traced residue has no counterpart in the sequence | local confidence)
INS_BASE_LOGIT = -2.3887

NAAlign = namedtuple("NAAlign", ["P", "C1", "O3", "conf", "chains"])
Record = namedtuple("Record", ["merged_idx", "type", "seq", "W"])
Fragment = namedtuple("Fragment", ["chain_id", "gids", "lp", "ins"])
State = namedtuple("State", ["rec", "pos", "letters", "ident_conf"])


def na_align_input(atoms, aatype, local_confidence, chains):
    atoms = np.asarray(atoms, dtype=np.float64)
    geo = {k: np.full((len(aatype), 3), np.nan) for k in ("P", "C1", "O3")}
    c1_idx = {t: restype_name_to_atomc_names[index_to_restype_3[t]].index("C1'") for t in range(num_prot, num_prot + 8)}
    for i in np.nonzero(np.asarray(aatype) >= num_prot)[0]:
        geo["P"][i] = atoms[i, P_ATOM]
        geo["O3"][i] = atoms[i, O3_ATOM]
        geo["C1"][i] = atoms[i, c1_idx[int(aatype[i])]]
    return NAAlign(geo["P"], geo["C1"], geo["O3"], np.asarray(local_confidence, dtype=np.float64).reshape(-1), chains)


def _log_softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def residue_letter_logprobs(aa_logits):
    # per residue and molecule type: log P(letter | type) (L, 4) and log P(type) (L,), from the 8 NA classes
    lp = _log_softmax(np.asarray(aa_logits, dtype=np.float64)[:, num_prot:num_prot + 8])
    out = {}
    for t, sl in (("DNA", slice(0, 4)), ("RNA", slice(4, 8))):
        lt = np.logaddexp.reduce(lp[:, sl], axis=1)
        lb = lp[:, sl] - lt[:, None]
        lc = np.stack([np.logaddexp(lb[:, 0], lb[:, 2]), np.logaddexp(lb[:, 1], lb[:, 3])], axis=1)[:, [0, 1, 0, 1]]
        out[t] = _log_softmax(lc + WITHIN_CLASS_WEIGHT * (lb - lc))
        out["type_" + t] = lt
    return out


def record_matrix(seq):
    # (S, 4) membership of the record letters; ambiguity codes spread uniformly over the compatible letters
    W = np.zeros((len(seq), 4))
    for p, ch in enumerate(seq.upper()):
        comp = [LETTER_INDEX[ch]] if ch in LETTER_INDEX else [LETTER_INDEX[c] for c in IUPAC[ch]]
        W[p, comp] = 1.0 / len(comp)
    return W


def build_records(prot_sequences, rna_sequences, dna_sequences):
    n_prot = len(prot_sequences or [])
    rna, dna = list(rna_sequences or []), list(dna_sequences or [])
    records = []
    for t, seqs, off in (("RNA", rna, n_prot), ("DNA", dna, n_prot + len(rna))):
        records += [Record(off + k, t, s, record_matrix(s)) for k, s in enumerate(seqs)]
    return records


def emission_matrix(lp, rec):
    # (L, S) log-odds of matching residue i to record position p, plus the molecule-type evidence
    e = np.log(np.maximum(np.exp(lp[rec.type]) @ rec.W.T, 1e-300)) - LOG_BG
    return e + lp["type_" + rec.type][:, None]


def posterior_align(E, ins=None):
    """
    Forward-backward over the semi-global alignment of L residues (rows of E) to S record positions (columns).
    States: M (residue matched to a position), I (unassigned residue between matches), D (skipped position),
    B / T (unassigned residues before the first / after the last match); affine costs; the start and end in the
    record are free; a jump re-enters the record at any position. ins (L,) is an emission of the unassigned states.
    Returns logZ, postM (L, S) and postI (L,), the posterior of being unassigned; postM.sum(1) + postI == 1.
    """
    L, S = E.shape
    ins = np.zeros(L) if ins is None else ins
    fM = np.full((L + 1, S + 1), NEG)
    fI = np.full((L + 1, S + 1), NEG)
    fD = np.full((L + 1, S + 1), NEG)
    fB = np.full((L + 1, S + 1), NEG)
    fT = np.full((L + 1, S + 1), NEG)
    fM[0, :] = 0.0
    ar = np.arange(S + 1)
    neg_row = np.full(S + 1, NEG)
    for i in range(1, L + 1):
        prev = np.logaddexp(np.logaddexp(fM[i - 1], fI[i - 1]), np.logaddexp(fD[i - 1], fB[i - 1]))
        fM[i, 1:] = E[i - 1] + prev[:-1]
        if i > 1:
            fM[i, 1:] = np.logaddexp(fM[i, 1:], E[i - 1] + np.logaddexp.reduce(prev) + JUMP_COST - np.log(S))
        real_m = fM[i - 1] if i > 1 else neg_row
        fI[i] = np.logaddexp(real_m + GAP_OPEN, fI[i - 1] + GAP_EXTEND) + ins[i - 1]
        fT[i] = np.logaddexp(real_m + END_OPEN, fT[i - 1] + END_EXTEND) + ins[i - 1]
        fB[i] = (END_OPEN if i == 1 else fB[i - 1] + END_EXTEND) + ins[i - 1]
        A = fM[i] - ar * GAP_EXTEND
        run = np.logaddexp.accumulate(A)
        fD[i, 1:] = run[:-1] + GAP_OPEN + (ar[1:] - 1) * GAP_EXTEND
    logZ = float(np.logaddexp.reduce(np.concatenate([fM[L], fT[L], fB[L]])))
    bM = np.full((L + 1, S + 1), NEG)
    bI = np.full((L + 1, S + 1), NEG)
    bD = np.full((L + 1, S + 1), NEG)
    bB = np.full((L + 1, S + 1), NEG)
    bT = np.full((L + 1, S + 1), NEG)
    bM[L, :] = 0.0
    bT[L, :] = 0.0
    bB[L, :] = 0.0
    for i in range(L - 1, 0, -1):
        nextM = np.full(S + 1, NEG)
        nextM[:-1] = E[i] + bM[i + 1, 1:]
        jump = float(np.logaddexp.reduce(E[i] + bM[i + 1, 1:])) + JUMP_COST - np.log(S)
        bB[i] = np.logaddexp(np.logaddexp(nextM, END_EXTEND + ins[i] + bB[i + 1]), jump)
        bI[i] = np.logaddexp(np.logaddexp(nextM, GAP_EXTEND + ins[i] + bI[i + 1]), jump)
        bT[i] = END_EXTEND + ins[i] + bT[i + 1]
        X = np.logaddexp(nextM, jump) + ar * GAP_EXTEND
        suf = np.logaddexp.accumulate(X[::-1])[::-1]
        bD[i] = suf - ar * GAP_EXTEND
        toD = np.full(S + 1, NEG)
        toD[:-1] = GAP_OPEN + bD[i, 1:]
        bM[i] = np.logaddexp(np.logaddexp(np.logaddexp(nextM, GAP_OPEN + ins[i] + bI[i + 1]),
                                          np.logaddexp(END_OPEN + ins[i] + bT[i + 1], toD)), jump)
    postM = np.exp(fM[1:, 1:] + bM[1:, 1:] - logZ)
    unassigned = np.logaddexp(np.logaddexp(fI[1:, :] + bI[1:, :], fB[1:, :] + bB[1:, :]), fT[1:, :] + bT[1:, :])
    postI = np.exp(np.logaddexp.reduce(unassigned, axis=1) - logZ)
    return logZ, postM, postI


def detect_pairs(na, chain_ids):
    # candidate base pairs among the fragments' residues: C1'-C1' and P-P windows plus an antiparallel
    # neighbour pair; one partner per residue, the closest to 10.5 A first. Keys and values are residue indices.
    gids = np.concatenate([np.asarray(na.chains[k]) for k in chain_ids])
    frag = np.concatenate([np.full(len(na.chains[k]), k) for k in chain_ids])
    pos = np.concatenate([np.arange(len(na.chains[k])) for k in chain_ids])
    C1, P = na.C1[gids], na.P[gids]
    ok = np.isfinite(C1).all(1) & np.isfinite(P).all(1)
    loc = {(int(f), int(q)): n for n, (f, q) in enumerate(zip(frag, pos)) if ok[n]}
    idx = np.nonzero(ok)[0]
    if len(idx) < 2:
        return {}

    def in_window(a, b):
        d = np.linalg.norm(C1[a] - C1[b])
        return PAIR_C1[0] <= d <= PAIR_C1[1]

    cands = []
    for ia, ib in cKDTree(C1[idx]).query_pairs(PAIR_C1[1]):
        a, b = idx[ia], idx[ib]
        d = np.linalg.norm(C1[a] - C1[b])
        if d < PAIR_C1[0]:
            continue
        if frag[a] == frag[b] and abs(int(pos[a]) - int(pos[b])) < 4:
            continue
        dp = np.linalg.norm(P[a] - P[b])
        if not (PAIR_P[0] <= dp <= PAIR_P[1]):
            continue
        fa, pa, fb, pb = int(frag[a]), int(pos[a]), int(frag[b]), int(pos[b])
        n1 = (loc.get((fa, pa + 1)), loc.get((fb, pb - 1)))
        n2 = (loc.get((fa, pa - 1)), loc.get((fb, pb + 1)))
        if not ((None not in n1 and in_window(*n1)) or (None not in n2 and in_window(*n2))):
            continue
        cands.append((abs(d - 10.5), a, b))
    partner = {}
    for _, a, b in sorted(cands):
        ga, gb = int(gids[a]), int(gids[b])
        if ga in partner or gb in partner:
            continue
        partner[ga] = gb
        partner[gb] = ga
    return partner


def detect_links(na, chain_ids):
    # O3' of the last residue of g within LINK_MAX_DIST of P of the first residue of f: {f: g} and {g: f}
    ends, starts = {}, {}
    for k in chain_ids:
        c = np.asarray(na.chains[k])
        o3, p = na.O3[c[-1]], na.P[c[0]]
        if np.isfinite(o3).all():
            ends[k] = o3
        if np.isfinite(p).all():
            starts[k] = p
    cands = []
    for g, o3 in ends.items():
        for f, p in starts.items():
            if f == g:
                continue
            d = float(np.linalg.norm(o3 - p))
            if d <= LINK_MAX_DIST:
                cands.append((d, g, f))
    pred, succ = {}, {}
    for _, g, f in sorted(cands):
        if g in succ or f in pred:
            continue
        succ[g] = f
        pred[f] = g
    return pred, succ


def pair_log_odds(letter, p):
    # log-odds of the letters of a residue whose partner reads `letter`, for a pair genuine with probability p
    v = COMPLEMENT[letter]
    return np.log(p * (4 / v.sum()) * v + (1.0 - p))


def link_log_odds(gap, S):
    pg = LINK_PRIOR[gap] / sum(LINK_PRIOR)
    return float(np.log(LINK_PRECISION * S * pg + (1.0 - LINK_PRECISION)))


def _emissions(frag, rec, r, mods):
    lp = frag.lp
    if mods is not None and mods["pair"] is not None:
        lp = dict(lp)
        lp[rec.type] = lp[rec.type] + mods["pair"]
    E = emission_matrix(lp, rec)
    if mods is not None:
        if r in mods["occ"]:
            E = E - OCCUPANCY_WEIGHT * mods["occ"][r][None, :]
        for (i, p), bonus in mods["link"].get(r, {}).items():
            E[i, p] += bonus
    return E


def enforce_monotone(pos):
    # the MAP positions are per-residue decisions: unmatch a residue whose position is not ahead of the previous
    # matched one, unless the step back is a jump to another copy (split_chains_at_jumps cuts there)
    pos = pos.copy()
    last = None
    for i in range(len(pos)):
        if pos[i] < 0:
            continue
        if last is not None and pos[i] <= last and last - pos[i] < SPLIT_GAP:
            pos[i] = -1
        else:
            last = pos[i]
    return pos


def align_fragment(frag, records, mods=None):
    # records weighted by their evidence; per residue the MAP (record, position) and the posterior base
    L = len(frag.gids)
    E, logZ, postM, postI = [], [], [], []
    for r, rec in enumerate(records):
        E.append(_emissions(frag, rec, r, mods))
        z, pm, pi = posterior_align(E[r])
        logZ.append(z)
        postM.append(pm)
        postI.append(pi)
    logZ = np.array(logZ)
    w = np.exp(logZ - logZ.max())
    w /= w.sum()
    r_star = int(np.argmax(w))
    pair = None if mods is None else mods["pair"]
    base = np.zeros((L, 4))
    for r, rec in enumerate(records):
        if w[r] < 1e-12 or rec.type != records[r_star].type:
            continue
        lp = frag.lp[rec.type]
        matched_base = postM[r] @ rec.W
        matched_lp = lp if pair is None else lp + pair
        for p in np.flatnonzero((rec.W > 0).sum(axis=1) > 1):  # ambiguity codes, conditioned on the residue's evidence
            allowed = rec.W[p] > 0
            probs = np.zeros((L, 4))
            probs[:, allowed] = np.exp(_log_softmax(matched_lp[:, allowed] + np.log(rec.W[p, allowed])))
            matched_base += postM[r][:, p, None] * (probs - rec.W[p])
        base += w[r] * matched_base
        base += w[r] * postI[r][:, None] * np.exp(lp)
    best_r, best_p, best_v = np.full(L, -1), np.full(L, -1), np.zeros(L)
    for r in range(len(records)):
        if w[r] < 1e-12:
            continue
        v = w[r] * postM[r]
        p = v.argmax(axis=1)
        vv = v[np.arange(L), p]
        upd = vv > best_v
        best_r[upd], best_p[upd], best_v[upd] = r, p[upd], vv[upd]
    # the insertion prior decides only which residues are numbered: the MAP position must keep MIN_POST of the mass
    # under the prior-augmented posterior of the chosen record as well
    gate = posterior_align(E[r_star], frag.ins)[1].max(axis=1)
    pos = np.where((best_r == r_star) & (best_v >= MIN_POST) & (gate >= MIN_POST), best_p, -1)
    pos = enforce_monotone(pos)
    if (pos >= 0).sum() < min(MIN_MATCHED, len(records[r_star].seq)):
        return None
    return State(r_star, pos, base.argmax(axis=1), base.max(axis=1) / np.maximum(base.sum(axis=1), 1e-12))


def modifiers(i, frags, states, records, partner, where, pred, succ, chain_to_n, copies):
    # position-specific additions to fragment i's emissions from the other fragments' current assignments
    frag = frags[i]
    pair = None
    if partner:
        pair = np.zeros((len(frag.gids), 4))
        any_pair = False
        for j, gid in enumerate(frag.gids):
            pj = partner.get(int(gid))
            if pj is None:
                continue
            fi2, j2 = where[pj]
            st = states[fi2]
            if fi2 == i or st is None or st.pos[j2] < 0:
                continue
            pair[j] = pair_log_odds(int(st.letters[j2]), PAIR_PRECISION * float(st.ident_conf[j2]))
            any_pair = True
        if not any_pair:
            pair = None
    occ = {}
    for fi2, st in enumerate(states):
        if fi2 == i or st is None:
            continue
        m = st.pos >= 0
        if m.any():
            occ.setdefault(st.rec, np.zeros(len(records[st.rec].seq)))
            np.add.at(occ[st.rec], st.pos[m], 1.0)
    for r in occ:  # the first `copies` occupants of a position are expected (repeated entities)
        occ[r] = np.maximum(occ[r] - (copies - 1), 0.0)
    link = {}
    g = pred.get(frag.chain_id)
    if g is not None and states[chain_to_n[g]] is not None:
        st = states[chain_to_n[g]]
        if st.pos[-1] >= 0:
            S = len(records[st.rec].seq)
            for gap in range(len(LINK_PRIOR)):
                p = int(st.pos[-1]) + 1 + gap
                if p < S:
                    link.setdefault(st.rec, {})[(0, p)] = link_log_odds(gap, S)
    h = succ.get(frag.chain_id)
    if h is not None and states[chain_to_n[h]] is not None:
        st = states[chain_to_n[h]]
        if st.pos[0] >= 0:
            S = len(records[st.rec].seq)
            for gap in range(len(LINK_PRIOR)):
                p = int(st.pos[0]) - 1 - gap
                if p >= 0:
                    link.setdefault(st.rec, {})[(len(frag.gids) - 1, p)] = link_log_odds(gap, S)
    return {"pair": pair, "occ": occ, "link": link}


def same(a, b):
    if a is None or b is None:
        return a is b
    return (a.rec == b.rec and np.array_equal(a.pos, b.pos) and np.array_equal(a.letters, b.letters)
            and np.array_equal(a.ident_conf, b.ident_conf))


def to_hmm_alignment(st, records):
    from CryoAtom2.utils.hmm_sequence_align import HMMAlignment

    rec = records[st.rec]
    res_idx = np.where(st.pos >= 0, st.pos + 1, -1)
    matched = res_idx[res_idx > 0]
    return HMMAlignment(
        sequence=CLASS_OFFSET[rec.type] + st.letters,
        seq_idx=rec.merged_idx,
        res_idx=res_idx,
        key_start_match=int(matched.min()),
        key_end_match=int(matched.max()),
        match_score=len(matched) / len(st.pos),
        hmm_output_match_sequence="".join(rec.seq[p].upper() if p >= 0 else "-" for p in st.pos),
        exists_in_sequence_mask=(st.pos >= 0).astype(int),
    )


def align_na_fragments(chain_aa_logits, chain_prot_mask, prot_sequences, rna_sequences, dna_sequences, na):
    # {chain_id: HMMAlignment} for the nucleic-acid chains of >= 3 residues; a chain without an admissible record
    # is absent (the caller falls back to the profile HMM)
    records = build_records(prot_sequences, rna_sequences, dna_sequences)
    chain_ids = [k for k, (lg, pm) in enumerate(zip(chain_aa_logits, chain_prot_mask)) if len(lg) >= 3 and np.all(~pm)]
    if not records or not chain_ids:
        return {}
    frags = []
    for k in chain_ids:
        gids = np.asarray(na.chains[k])
        frags.append(Fragment(k, gids, residue_letter_logprobs(chain_aa_logits[k]),
                              INS_COEF[0] + INS_COEF[1] * na.conf[gids] - INS_BASE_LOGIT))
    states = [align_fragment(f, records) for f in frags]
    partner = detect_pairs(na, chain_ids)
    pred, succ = detect_links(na, chain_ids)
    where = {int(g): (n, j) for n, f in enumerate(frags) for j, g in enumerate(f.gids)}
    chain_to_n = {f.chain_id: n for n, f in enumerate(frags)}
    copies = max(1, int(round(sum(len(f.gids) for f in frags) / max(sum(len(r.seq) for r in records), 1))))
    order = sorted(range(len(frags)), key=lambda n: -len(frags[n].gids))  # iterated conditional modes, longest first
    for _ in range(ICM_SWEEPS):
        changed = 0
        for n in order:
            new = align_fragment(frags[n], records,
                                 modifiers(n, frags, states, records, partner, where, pred, succ, chain_to_n, copies))
            changed += not same(new, states[n])
            states[n] = new
        if changed == 0:
            break
    return {f.chain_id: to_hmm_alignment(st, records) for f, st in zip(frags, states) if st is not None}


def split_chains_at_jumps(chains, match):
    # cut the nucleic-acid chains where the record position steps back or skips more than SPLIT_GAP positions
    # (another copy of a repeated entity, or a traced chain that jumps); unmatched residues stay with the segment before
    out_chains = []
    fields = {k: [] for k in ("new_sequences", "residue_idxs", "sequence_idxs", "key_start_matches", "key_end_matches",
                              "match_scores", "hmm_output_match_sequences", "exists_in_sequence_mask", "is_nucleotide")}
    for ci, chain in enumerate(chains):
        chain = np.asarray(chain)
        ridx = np.asarray(match.residue_idxs[ci])
        cuts = [0]
        if match.is_nucleotide[ci] and len(ridx) > 1:
            last = None
            for j, r in enumerate(ridx):
                if r < 0:
                    continue
                if last is not None and (r <= last or r - last > SPLIT_GAP + 1):
                    cuts.append(j)
                last = r
        cuts.append(len(ridx))
        if len(cuts) == 2:
            out_chains.append(chain)
            for key in fields:
                fields[key].append(getattr(match, key)[ci])
            continue
        for a, b in zip(cuts[:-1], cuts[1:]):
            out_chains.append(chain[a:b])
            seg = ridx[a:b]
            matched = seg[seg > 0]
            fields["new_sequences"].append(np.asarray(match.new_sequences[ci])[a:b])
            fields["residue_idxs"].append(seg)
            fields["sequence_idxs"].append(match.sequence_idxs[ci])
            fields["key_start_matches"].append(int(matched.min()) if len(matched) else int(seg.min()))
            fields["key_end_matches"].append(int(matched.max()) if len(matched) else int(seg.max()))
            fields["match_scores"].append(float(len(matched) / max(b - a, 1)))
            fields["hmm_output_match_sequences"].append(match.hmm_output_match_sequences[ci][a:b])
            fields["exists_in_sequence_mask"].append(np.asarray(match.exists_in_sequence_mask[ci])[a:b])
            fields["is_nucleotide"].append(match.is_nucleotide[ci])
    match.set_vals(**fields)
    return out_chains
