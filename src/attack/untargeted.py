import numpy as np
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    PixelAttack,
)


class UntargetedAttacks:
    def __init__(self, attack_name: str, classifier, x, batch_size=32) -> None:
        if attack_name == "PIXELATTACK":
            self.attack = PixelAttack(
                classifier=classifier, th=10, max_iter=10, verbose=True
            )
        elif attack_name == "FGSM_UNTARGETED":
            self.attack = FastGradientMethod(
                estimator=classifier, eps=0.2, batch_size=batch_size
            )
        elif attack_name == "PGD_L2_UNTARGETED":
            self.attack = ProjectedGradientDescent(
                estimator=classifier,
                norm=2,
                eps=0.5,
                eps_step=0.2,
                batch_size=batch_size,
                verbose=True,
            )
        elif attack_name == "PGD_LINF_UNTARGETED":
            self.attack = ProjectedGradientDescent(
                estimator=classifier,
                norm=np.inf,
                eps=0.5,
                eps_step=0.2,
                batch_size=batch_size,
                verbose=True,
            )
        else:
            raise NotImplementedError
        # Generate adversarial test examples
        self.x_adv = self.attack.generate(x=x)
