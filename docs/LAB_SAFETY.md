# Laboratory execution rules

- Write only inside the project root. Keep environments, package caches, framework caches, temporary files, assets, weights, and outputs in their project-local directories.
- Keep `sources/` unchanged. Put compatibility code and diagnostics in this project's own source tree.
- Do not use sudo, change system CUDA/drivers/libraries, modify shell startup files, or alter shared environments.
- Do not stop other users' processes or reclaim their GPU allocations. An idle reading is not a reservation: follow laboratory scheduling rules.
- Use at most one assigned GPU for model inference. Run simulation with OSMesa/llvmpipe on CPU.
- Bind model services only to `127.0.0.1`. Check ports before startup. Release your own service after evaluation.
- Do not publish credentials, auth caches, environment variables containing secrets, weights, downloaded assets, datasets, or environments. Check the exact staged file list before committing.
- Preserve prior experiment outputs; use distinct timestamped run directories. Do not overwrite existing files during setup.
- Downloads must have an identified source and destination. Check disk capacity and integrity; avoid unknown unbounded writes. Current launch scripts do not download files.
- These rules are operational boundaries, not a replacement for the laboratory's allocation and data policies.
