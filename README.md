# Instructions
1. Install the requirements from `requirements.txt`. I recommend doing it in a Docker container.
2. Select a model from the `conf/config.yaml` file by specifying a model's name between `pretrained` and `base_cnn.` `pretrained` can be used for inference on pre-trained models. 
3. Select an Attack from the `conf/config.yaml` file by specifying the attack's name. It can be any name in `auto`, `cw_l0`, `cw_l2`, `cw_linf`, `fgsm`, `pgd_l2`, `pgd_linf`, `pixel` or `zoo`.
4. Run `python src/main.py`.
5. All metrics are logged in a `logs` directory which is created at runtime.
