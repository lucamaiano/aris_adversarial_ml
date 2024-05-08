# Instructions
1. Install the requirements from `requirements.txt`. I recommend doing it in a Docker container.
2. Select an Attack from the `conf/config.yaml` file by specifying the attack's name. It can be any name in `auto`, `cw_l0`, `cw_l2`, `cw_linf`, `fgsm`, `pgd_l2`, `pgd_linf`, `pixel` or `zoo`.
3. Run `python src/main.py`.
4. All metrics are logged in a `logs` directory which is created at runtime.
