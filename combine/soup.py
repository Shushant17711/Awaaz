"""Linear weight interpolation of two same-lineage checkpoints: out = (1-a)*A + a*B."""
import sys, torch
A, B, a, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
ca = torch.load(A, map_location="cpu", weights_only=False); cb = torch.load(B, map_location="cpu", weights_only=False)
sa, sb = ca["model_state_dict"], cb["model_state_dict"]; assert sa.keys() == sb.keys()
cb["model_state_dict"] = {k: ((1 - a) * sa[k].float() + a * sb[k].float()).to(sa[k].dtype) if sa[k].is_floating_point() else sb[k] for k in sa}
cb["soup"] = {"a": A, "b": B, "alpha_b": a}
torch.save(cb, out)
