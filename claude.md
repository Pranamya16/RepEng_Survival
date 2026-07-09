# Cloud Rules — Google Colab + Google Drive + GitHub

Rules every notebook in this project must follow. Read once, apply everywhere.

---

## 1. Repo layout (GitHub)

Same structure as `project_plan.md` Stage 0:

```
/data       /models      /src      /notebooks      /results      /paper
```

- All code (`.py` in `/src`, `.ipynb` in `/notebooks`) lives in the GitHub repo.
- Colab notebooks `git clone` the repo at the start of each session, and `git pull` before running.
- Never edit code directly in Colab's UI long-term — edit in repo, push, then pull into Colab. (Quick inline fixes during a run are fine, just push them back after.)

---

## 2. Every notebook must have these 3 setup cells, in this order

**Cell 1 — Mount Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

RESULTS_DIR = '/content/drive/MyDrive/RepEng_Survival/results'
import os
os.makedirs(RESULTS_DIR, exist_ok=True)
```

**Cell 2 — Clone/pull repo**
```python
import os
if not os.path.exists('/content/RepEng_Survival'):
    !git clone https://github.com/<your-username>/RepEng_Survival.git /content/RepEng_Survival
else:
    !cd /content/RepEng_Survival && git pull
%cd /content/RepEng_Survival
```

**Cell 3 — Anti-disconnect (prevents Colab's idle-timeout from killing the session)**
```python
from IPython.display import Javascript
import IPython

display(Javascript('''
function KeepAlive(){
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(KeepAlive, 60000)
'''))
```
This clicks Colab's own "connect" button every 60 seconds, which resets the idle timer even with no cursor movement. It only stops the *idle* disconnect — it does not stop the **12-hour hard session limit** on free tier. For long jobs, checkpoint (see rule 3) so a run can resume after a forced disconnect.

---

## 3. One model at a time — never load multiple models together

Free-tier T4 has only 16GB VRAM. Loading more than one model at once (even in 4-bit) risks running out of memory once activations, probes, and steering hooks are added on top.

- Every notebook loops over models **sequentially**: load one model → run the stage's work → save to Drive → unload → move to next.
- Unload fully before loading the next model:
```python
del model
import gc, torch
gc.collect()
torch.cuda.empty_cache()
```
- Never keep two models in GPU memory at the same time, even briefly for comparison — compare using saved results from Drive instead.

---

## 4. Save results as you go, not at the end

- Never hold results only in a Python variable until the notebook finishes. Colab can disconnect anytime.
- After **every** unit of work (one layer's probe, one model's extraction, one recovery run), write the result straight to `RESULTS_DIR` on Drive.
- Use a checkpoint pattern: before starting a unit of work, check if its result file already exists on Drive — if yes, skip it. This makes every stage resumable after a disconnect.

```python
import json

def save_result(name, data):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, 'w') as f:
        json.dump(data, f)

def load_result_if_exists(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
```

- Activation tensors: save with `torch.save()` to Drive, not just to `/content` (local Colab disk is wiped on disconnect).

---

## 5. End-of-notebook: push results back to GitHub

After a stage finishes (or Drive already has the final result files), commit the result summaries (tables, small JSON/CSV — not huge tensor files) back to the repo's `/results` folder:

```python
!cp /content/drive/MyDrive/RepEng_Survival/results/*.json /content/RepEng_Survival/results/
!cd /content/RepEng_Survival && git add results/ && git commit -m "Add results: <stage name>" && git push
```

Large files (raw activation tensors, model weights) stay on Drive only — do not push those to GitHub.

---

## 6. Quick checklist before running any notebook

- [ ] Drive mounted, `RESULTS_DIR` set
- [ ] Repo cloned/pulled
- [ ] Anti-disconnect cell run
- [ ] Only one model loaded on GPU at a time, unloaded before switching
- [ ] Checkpoint check added before any expensive step
- [ ] Save-to-Drive call added after every expensive step
