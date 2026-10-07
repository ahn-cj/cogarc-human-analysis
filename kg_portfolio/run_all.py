#!/usr/bin/env python
"""Rebuild everything end to end: graph, analyses, site.

    python run_all.py

Each stage is independent and re-runnable. Total runtime is a couple of minutes,
dominated by the per-row inserts in stage 1.
"""
import subprocess, sys, time, os

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    ("build the graph in Kuzu",        "etl/build_layer1.py"),
    ("add the model describers",       "etl/add_agents.py"),
    ("community + projection analyses", "etl/graph_algos.py"),
    ("link prediction (PyTorch Geometric)", "etl/link_prediction.py"),
    ("export the site data bundle",    "etl/export_site.py"),
]

def main():
    for i, (label, script) in enumerate(STAGES, 1):
        print(f"\n\033[1m[{i}/{len(STAGES)}] {label}\033[0m  ({script})")
        t = time.time()
        r = subprocess.run([sys.executable, script], cwd=HERE)
        if r.returncode:
            print(f"\nFAILED at stage {i} ({script})"); sys.exit(r.returncode)
        print(f"  done in {time.time()-t:.1f}s")
    site = os.environ.get("COGARC_SITE_REPO",
                          os.path.expanduser("~/Documents/GitHub/cogarc-kg-site"))
    print(f"\nAll stages complete. Now rebuild the site:\n"
          f"  python {os.path.join(site, 'src', 'build.py')}")

if __name__ == "__main__":
    main()
