"""Explanation backends: `qwen` (Qwen2.5-Instruct via transformers, greedy decoding) and `template` (deterministic
rule-based text; CPU/test fallback only; it is faithful by construction, so FaithScore from it is a plumbing check)."""
import re

from hcbm.explain import prompts


class TemplateExplainer:
    name = "template"

    def __call__(self, facts):
        lines = facts.splitlines()
        emo = next(l for l in lines if l.startswith("Predicted emotion")).split(": ", 1)[1]
        if "Level 2" not in facts:
            return f"The network's most likely class was {emo} No intermediate evidence is available for this model."
        l1 = [l[2:] for l in lines[1:lines.index("Level 2, affective state:")] if l.startswith("- ")]
        val = next(l for l in lines if l.startswith("- valence:")).split(": ", 1)[1].split(" (")[0]
        aro = next(l for l in lines if l.startswith("- arousal:")).split(": ", 1)[1].split(" (")[0]
        app = [l.split(": ", 1)[1].split(" (")[0] for l in lines if l.startswith(("- novelty", "- goal", "- coping"))]
        return (f"EEG evidence: {'; '.join(l1[:3])}. This pattern was read as {val} valence with {aro} arousal. "
                f"The situation was appraised as {', '.join(app)}, which is most consistent with {emo}")


class QwenExplainer:
    name = "qwen"

    def __init__(self, model_name, max_new_tokens=128, device="cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
        self.device, self.max_new_tokens, self.model_name = device, max_new_tokens, model_name

    def __call__(self, facts):
        import torch
        msgs = [{"role": "system", "content": prompts.SYSTEM}, {"role": "user", "content": prompts.user_prompt(facts)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def polarity_terms(text):
    """+1 if only positive-valence terms are used, -1 if only negative, 0 if neither/both ("unpleasant" not "pleasant")."""
    t = text.lower()
    neg = bool(re.search(r"\b(negative|unpleasant)\b", t))
    pos = bool(re.search(r"\b(positive|pleasant)\b", t))
    return int(pos and not neg) - int(neg and not pos)
