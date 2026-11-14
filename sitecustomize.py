"""Inkling 2 on one GB300: patch the image's Inkling 2 model as soon as it is imported (every process; no-op unless
INK2_SPLIT=1)."""
import importlib.abc
import importlib.util
import os
import sys

if os.environ.get("INK2_SPLIT", "1") == "1":
    class _Finder(importlib.abc.MetaPathFinder):
        target = "vllm.models.inkling2.model"

        def find_spec(self, name, path, target=None):
            if name != self.target:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                from ink2.split import install
                install()
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())

# INK2_GRACE: vLLM's own threads (every process) stay on cores 0 .. INK2_GRACE_CORE0 - 1; the Grace band's pool uses the rest
if os.environ.get("INK2_GRACE") in ("native", "int4"):
    try:
        os.sched_setaffinity(0, set(range(int(os.environ.get("INK2_GRACE_CORE0", "16")))))
    except OSError:
        pass
