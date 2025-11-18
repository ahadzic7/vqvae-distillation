"""
beam_search.py
This module implements various beam search algorithms for autoregressive generative models,
with support for class-conditional and non-conditional generation, as well as stochastic and
randomized variants. The code is designed to work with models such as PixelCNNs over VQ-VAE
latents, and supports efficient batched and vectorized search.

Main Functions:
---------------
- beam_search_wc:               Batched, vectorized beam search for class-conditional models.
- beam_search_nc:               Batched, vectorized beam search for non-class-conditional models.
- beam_search_vs:               Stochastic (sampling-based) beam search for class-conditional models.
- beam_search_c:                Beam search over multiple classes, returning top samples per class.
- beam_search_c_unique:         Beam search ensuring unique samples per class.
- beam_search_cs:               Stochastic beam search over multiple classes.
- beam_search_cs_unique:        Stochastic beam search ensuring unique samples per class.
- beam_search_random_start:     Beam search with random starting token(s).
- beam_search_random_start_c:   Random start beam search over multiple classes.

Old Implementations:
--------------------
- beam_search_slow:      Non-batched, slow reference implementation of beam search.
- beam_search_s_slow:    Non-batched, slow stochastic beam search.

Helper Functions:
-----------------
- _logits_wc, _logits_nc:    Compute logits for class-conditional and non-conditional models.
- beam_search_v_init_wc, beam_search_v_init_nc: Core vectorized beam search steps.
- _trajectories:             Helper for random start beam search.

Notes:
------
- All functions assume a compatible PixelCNN-like model (`pcnn`) with a VQ-VAE latent space.
- The code supports both deterministic and stochastic search, as well as uniqueness constraints.
- Some functions print the fraction of unique samples found for debugging/analysis.
"""
import torch
import math
from tqdm import tqdm
from src.utilities import get_iterator


# vanilla bs
def beam_search_wc(pcnn, B, verbose=False):
    _, H, W = pcnn.vqvae.latent_shape
    beam_z = torch.zeros((1, H, W), dtype=torch.int32, device=pcnn.device)
    beam_h = torch.zeros((1, len(pcnn.classes)), device=pcnn.device)  
    return beam_search_v_init_wc(pcnn, B, beam_z, beam_h, verbose=verbose)

@torch.no_grad()
def _logits_wc(pcnn, beam_z, K, H, W, max_chunk):
    C = len(pcnn.classes)
    B_cur = beam_z.shape[0]
    z_chs = beam_z.repeat_interleave(C, dim=0).split(max_chunk, dim=0)
    classes_chs = pcnn.classes.tile(B_cur).split(max_chunk, dim=0)
    flat_logits = torch.empty(B_cur*C, K, H, W, device=pcnn.device)
      
    idx = 0
    for z_chunk, cls_chunk in zip(z_chs, classes_chs):
        out = pcnn.forward(z_chunk.to(pcnn.device), cls_chunk.to(pcnn.device)).log_softmax(dim=1) # (chunk_size, K, H, W)
        flat_logits[idx : idx + out.shape[0]].copy_(out)
        idx += out.shape[0]
    logits = flat_logits.view(B_cur, C, K, H, W)
    return logits

@torch.no_grad()
def beam_search_v_init_wc(pcnn, B, beam_z, beam_h, start_depth=0, max_chunk=2**13, verbose=False):
    K = pcnn.vqvae.codebook_size
    _, H, W = pcnn.vqvae.latent_shape
    C = len(pcnn.classes)

    for depth in get_iterator(range(start_depth, H * W), verbose, name="Trajectories"):
        logits = _logits_wc(pcnn, beam_z, K, H, W, max_chunk) # [B_cur, C, K, H, W]

        # update per‐class scores
        r, c = divmod(depth, W)
        new_h = beam_h.unsqueeze(2) + logits[..., r, c] # [B_cur, C, K]

        # marginalize over classes for each candidate, get ln p(z_i|z_<i)
        mar_lls = torch.logsumexp(new_h - math.log(C), dim=1).view(beam_z.shape[0] * K)

        _, top_idx = torch.topk(mar_lls, k=min(mar_lls.shape[0], B), dim=0)
        beam_idx = top_idx // K
        code_idx = top_idx % K

        beam_h = new_h[beam_idx, :, code_idx]
        beam_z = beam_z[beam_idx].clone()
        beam_z[:, r, c] = code_idx

    mar_lls = torch.logsumexp(beam_h - math.log(C), dim=1)  # [B]
    return beam_z, mar_lls


# beam search for non class conditioned 
def beam_search_nc(pcnn, B, verbose=False):
    _, H, W = pcnn.vqvae.latent_shape
    beam_z = torch.zeros((1, H, W), dtype=torch.int32, device=pcnn.device)
    beam_h = torch.zeros((1), device=pcnn.device)  
    return beam_search_v_init_nc(pcnn, B, beam_z, beam_h, verbose=verbose)

def _logits_nc(pcnn, beam_z, K, H, W, max_chunk):
    B_cur = beam_z.shape[0]
    flat_logits = torch.empty(B_cur, K, H, W, device=pcnn.device)
    with torch.no_grad():        
        idx = 0
        for z_chunk in beam_z.split(max_chunk, dim=0):
            out = pcnn.forward(z_chunk).log_softmax(dim=1) # (chunk_size, K, H, W)
            flat_logits[idx : idx + out.shape[0]].copy_(out)
            idx += out.shape[0]
        return flat_logits


def beam_search_v_init_nc(pcnn, B, beam_z, beam_h, start_depth=0, max_chunk=2**13, verbose=False):
    K = pcnn.vqvae.codebook_size
    _, H, W = pcnn.vqvae.latent_shape

    for depth in get_iterator(range(start_depth, H * W), verbose, name="Trajectories"):
        logits = _logits_nc(pcnn, beam_z, K, H, W, max_chunk) # [B_cur, K, H, W]
    
        # update per‐class scores
        r, c = divmod(depth, W)
        new_h = beam_h.unsqueeze(1) + logits[..., r, c] # [B_cur, K]
        lls = new_h.view(-1) # [B_cur * K]
        _, top_idx = torch.topk(lls, k=min(lls.shape[0], B), dim=0)
        beam_idx = top_idx // K
        code_idx = top_idx % K

        beam_h = new_h[beam_idx, code_idx]
        beam_z = beam_z[beam_idx].clone()
        beam_z[:, r, c] = code_idx

    return beam_z, beam_h





# socthastic bs
@torch.no_grad()
def beam_search_vs(pcnn, classes, B, D, max_chunk = 2**13, replacement=False, verbose=False):
    K = pcnn.vqvae.codebook_size
    _, H, W = pcnn.vqvae.latent_shape
    C = len(classes)
    device = pcnn.device

    beam_z = torch.zeros((1, H, W), dtype=torch.int32, device=device)
    beam_h = torch.zeros((1, C), device=device)
    
    for depth in get_iterator(range(H * W), verbose, name="Stochastic trajectories"):
        logits = _logits_wc(pcnn, beam_z, K, H, W, max_chunk)

        r, c = divmod(depth, W)
        new_h = beam_h.unsqueeze(2) + logits[..., r, c]    # [B_cur, C, K]
        mar_lls = torch.logsumexp(new_h - math.log(C), dim=1)  # [B_cur, K]
        
        if D < K: # perform sampling of D candidates
            children_z = beam_z.repeat_interleave(D, dim=0).clone()
            samples = torch.multinomial(mar_lls.softmax(dim=1), D, replacement=replacement) 
            children_z[:, r, c] = samples.view(-1)

            flat_m = mar_lls.gather(1, samples).view(-1) # [B_cur*D]
            h_sel = new_h.gather(2, samples.unsqueeze(1).expand(-1, C, -1)) # [B_cur,C,D]
            new_h_flat = h_sel.permute(0, 2, 1).reshape(-1, C) # [B_cur*D,C]
        else: # no sampling needed take everybody
            children_z = beam_z.repeat_interleave(K, dim=0).clone()
            children_z[:, r, c] = torch.arange(K, device=device).repeat(beam_z.size(0))

            flat_m = mar_lls.view(-1) # [B_cur*K]
            new_h_flat = new_h.permute(0, 2, 1).reshape(-1, C)  # [B_cur*K,C]

        if B < flat_m.shape[0]:
            samp_idx = torch.multinomial(flat_m.softmax(dim=0), B, replacement=replacement)
            ordered = samp_idx[torch.argsort(flat_m[samp_idx], descending=True)]
        else:
            ordered = torch.argsort(flat_m, descending=True)[:B]

        beam_z = children_z[ordered] # [B, H, W]
        beam_h = new_h_flat[ordered] # [B, C]
        
    mar_lls = torch.logsumexp(beam_h - math.log(C), dim=1)
    return beam_z, mar_lls

# class cond bs
def beam_search_c(pcnn, classes, B):
    zss, h = [], []
    for c in classes.unbind(dim=0):
        if B == 0:
            continue
        zs, _ = beam_search_v(pcnn, classes=classes[c].unsqueeze(0), B=B)
        zss.append(zs)
        h.append(pcnn.log_prob(zs, classes[c].unsqueeze(0)))
    zs = torch.cat(zss)
    print(zs.unique(dim=0).shape[0] / zs.shape[0])
    h = torch.cat(h)
    h, h_idx = torch.sort(h, descending=True)
    zs = zs[h_idx]
    return zs, h
    
def beam_search_c_unique(pcnn, classes, B, increase_pct=0.25):
    seen = set()
    all_zs, all_h = [], []

    for c in classes.unbind(dim=0):
        if B == 0:
            continue
        current_B = B
        c = c.unsqueeze(0)

        while True:
            zs, _ = beam_search_v(pcnn, classes=c, B=current_B)
            zs_flat = [tuple(z.cpu().flatten().tolist()) for z in zs]
            new_idx = [i for i, zf in enumerate(zs_flat) if zf not in seen]

            if len(new_idx) >= B:
                chosen = new_idx[:B]
                zs_chosen = zs[chosen]
                h_chosen = pcnn.log_prob(zs, c)[chosen]
                for i in chosen:
                    seen.add(zs_flat[i])
                all_zs.append(zs_chosen)
                all_h.append(h_chosen)
                break
            else:
                current_B = int(math.ceil(current_B * (1 + increase_pct)))
    zs = torch.cat(all_zs, dim=0)
    h  = torch.cat(all_h,  dim=0)
    print(zs.unique(dim=0).shape[0] / zs.shape[0])
    return zs, h

# class cond stochastic bs
@torch.no_grad()
def beam_search_cs(pcnn, B):
    zss, h = [], []
    for c in tqdm(pcnn.classes.unbind(dim=0), desc="Classes"):
        cls = pcnn.classes[c].unsqueeze(0).to(pcnn.device)
        zs, _ = beam_search_vs(pcnn, classes=cls, B=B//9, D=B)
        zss.append(zs)
        h.append(pcnn.log_prob(zs, cls))
    zs = torch.cat(zss)
    print(zs.unique(dim=0).shape[0] / zs.shape[0])
    print(zs.unique(dim=0).shape[0])
    print(zs.shape[0])
    h, h_idx = torch.sort(torch.cat(h), descending=True)
    zs = zs[h_idx]
    return zs[:B], h[:B]

@torch.no_grad()
def beam_search_cs_unique(pcnn, B, increase_pct=0.25):
    seen = set()
    ccc = pcnn.classes.to(pcnn.device)
    all_zs, all_h = [], []
    B_per_class = math.ceil(B / ccc.shape[0])
    print(B_per_class)

    for c in tqdm(ccc.unbind(dim=0), desc="Classes"):
        if B == 0:
            continue

        current_B = B_per_class
        c = c.unsqueeze(0)
        
        cls = ccc[c].unsqueeze(0).to(pcnn.device)

        while True:
            zs, _ = beam_search_vs(pcnn, classes=cls, B=current_B, D=2*current_B)
            chosen = []
            seen_set = set(seen)

            print(f"I have seen {len(seen_set)}")

            for i, z in enumerate(zs):
                zf = tuple(z.cpu().flatten().tolist())

                if zf not in seen_set:
                    chosen.append(i)
                    seen_set.add(zf)
                    if len(chosen) == B_per_class:
                        break
            print(f"I have chosen {len(chosen)}")

            if len(chosen) == B_per_class:
                print("Selected enough unique latents")
                zs_chosen = zs[chosen]
                h_chosen = pcnn.log_prob(zs_chosen, c)
                for i in chosen:
                    tup = tuple(zs[i].unsqueeze(0).cpu().flatten().tolist())
                    seen.add(tup)
                all_zs.append(zs_chosen)
                all_h.append(h_chosen)
                break
            else:
                print("Not enough unique latents")
                current_B = int(math.ceil(current_B * (1 + increase_pct)))
                print(f"Beam is now {current_B}")

    zs = torch.cat(all_zs, dim=0)
    h  = torch.cat(all_h,  dim=0)
    print(zs.unique(dim=0).shape[0] / zs.shape[0])
    return zs, h


# beam search with random starting token
def _trajectories(pcnn, beam_z, logits, inits, B, n_components, verbose):
    z_outs, lls_outs = [], []
    for init in get_iterator(inits, verbose, name="Inits"):
        beam_z[:, 0, 0] = init
        beam_h = logits[:, :, init]
        zs, mar_lls = beam_search_v_init_wc(pcnn, B, beam_z, beam_h, start_depth=1)
        z_outs.append(zs)
        lls_outs.append(mar_lls)
    zs = torch.cat(z_outs, dim=0)
    lls = torch.cat(lls_outs, dim=0)

    if lls.shape[0] > n_components:
        lls, top_idx = torch.topk(lls, k=n_components, dim=0)
        zs = zs[top_idx]
    return zs, lls

def beam_search_random_start(pcnn, classes, n_components, S, verbose=False):
    K = pcnn.vqvae.codebook_size
    _, H, W = pcnn.vqvae.latent_shape
    C = classes.shape[0]
    beam_z = torch.zeros((1, H, W), dtype=torch.int32, device=pcnn.device)
    logits = _logits_wc(pcnn, beam_z, K, H, W, max_chunk=2**13)[..., 0, 0]
    
    mar_lls = torch.logsumexp(logits - math.log(C), dim=1)[0]

    if S > K:
        S = K

    if n_components < S:
        inits = torch.multinomial(mar_lls.exp(), n_components, replacement=False)
        B = 1
    else:
        inits = torch.multinomial(mar_lls.exp(), S, replacement=False)
        B = -(-n_components // S)
    
    return _trajectories(pcnn, beam_z, logits, inits, B, n_components, verbose)

def beam_search_random_start_c(pcnn, n_components, S, verbose=False):
    classes = pcnn.classes.to(pcnn.device)
    zss, h = [], []
    B_per_class = math.ceil(n_components / classes.shape[0])
    for c in tqdm(classes.unbind(dim=0)):
        c = c.unsqueeze(0)
        zs, _ = beam_search_random_start(pcnn, classes=c, n_components=n_components, S=S, verbose=verbose)
        hc = pcnn.log_prob(zs, c)
        hc, hc_idx = torch.sort(hc, descending=True)
        zs = zs[hc_idx]
        zss.append(zs[:B_per_class])
        h.append(hc[:B_per_class])
    zs = torch.cat(zss)
    print(zs.unique(dim=0).shape[0] / zs.shape[0])
    h = torch.cat(h)
    # h, h_idx = torch.sort(h, descending=True)
    # zs = zs[h_idx]
    return zs[:n_components], h[:n_components]


# non-vectorised, non-batched versions
def beam_search_slow(pcnn, classes, B):
    K = pcnn.vqvae.codebook_size
    _, H, W = pcnn.vqvae.latent_shape
    C = len(classes)
    lnC = math.log(C)

    init_z = torch.zeros((H, W), dtype=torch.int, device=pcnn.device)
    init_scores = torch.zeros(C, device=pcnn.device)  # log p(z_<0>|c)=0
    beam = [(init_z, init_scores, 0)]

    for depth in tqdm(range(H * W), desc="BS"): 
        new_beam = []
        r, c = divmod(depth, W)

        for parent_z, parent_h, _ in beam:
            parent_z = parent_z.unsqueeze(0) 
            with torch.no_grad():
                step_logits = pcnn.forward(parent_z, classes).log_softmax(dim=1)
            step_logits = step_logits[:, :, r, c]

            # Update per-class log-likelihoods
            new_h = parent_h.unsqueeze(1) + step_logits
            # true marginal log-likelihoods for each candidate
            marginals = torch.logsumexp(new_h - lnC, dim=0)

            children_z = parent_z.repeat(K, 1, 1)  # (K, H, W)
            children_z[:, r, c] = torch.arange(K, device=pcnn.device, dtype=children_z.dtype)

            new = zip(children_z.unbind(), new_h.unbind(dim=1), marginals.unbind())
            new_beam.extend(new)

        beam = sorted(new_beam, key=lambda t: t[2].item(), reverse=True)[:B]

    final_beam = []
    for z, scores, _ in beam:
        marginal = torch.logsumexp(scores - lnC, dim=0)
        final_beam.append((z, marginal))

    return final_beam


def beam_search_s_slow(pcnn, classes, B, D):
    K = pcnn.vqvae.codebook_size
    _, H, W = pcnn.vqvae.latent_shape
    C = len(classes)
    lnC = math.log(C)

    init_z = torch.zeros((H, W), dtype=torch.int, device=pcnn.device)
    init_scores = torch.zeros(C, device=pcnn.device)  # log p(z_<0>|c)=0
    beam = [(init_z, init_scores, 0)]

    for depth in tqdm(range(H * W), desc="BS"): 
        new_beam = []
        r, c = divmod(depth, W)

        for parent_z, parent_h, _ in beam:
            parent_z = parent_z.unsqueeze(0) 
            with torch.no_grad():
                step_logits = pcnn.forward(parent_z, classes).log_softmax(dim=1)
            step_logits = step_logits[:, :, r, c]

            # Update per-class log-likelihoods
            new_h = parent_h.unsqueeze(1) + step_logits
            # true marginal log-likelihoods for each candidate
            marginals = torch.logsumexp(new_h - lnC, dim=0)

            if D < K:
                probs=marginals.softmax(dim=0)
                unique_samples = torch.multinomial(probs, D, replacement=False)
                children_z = parent_z.repeat(D, 1, 1)  # (K, H, W)
                children_z[:, r, c] = unique_samples
                marginals = marginals[unique_samples]
                new_h = new_h[:, unique_samples]
            else:
                children_z = parent_z.repeat(K, 1, 1)  # (K, H, W)
                children_z[:, r, c] = torch.arange(K, device=pcnn.device, dtype=children_z.dtype)

            new = zip(children_z.unbind(), new_h.unbind(dim=1), marginals.unbind())
            new_beam.extend(new)

        if len(new_beam) > B:
            probs = torch.stack([m for _, _, m in new_beam]).softmax(dim=0)
            unique_samples = torch.multinomial(probs, B, replacement=False)
            new_beam = [new_beam[i] for i in unique_samples]
        beam = sorted(new_beam, key=lambda t: t[2].item(), reverse=True)

    final_beam = []
    for z, scores, _ in beam:
        marginal = torch.logsumexp(scores - lnC, dim=0)
        final_beam.append((z, marginal))

    return final_beam

