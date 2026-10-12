#!/usr/bin/env python3
"""Run formal LIBERO-Dynamic eval with the reviewed old preview physics.

This entrypoint keeps the normal LIBERO-style policy, latency, video, and
success-check loop. The dynamic scene logic is always the reviewed preview
physical-platform route for every dynamic target.
"""

from __future__ import annotations

from eval_dynamic_libero_style import main


if __name__ == "__main__":
    raise SystemExit(main(reviewed_old_physics=True))
