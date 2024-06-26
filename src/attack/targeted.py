import numpy as np
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    CarliniL0Method,
    CarliniL2Method,
    CarliniLInfMethod,
    ZooAttack,
    AutoAttack,
)


class TargetedAttacks:
    def __init__(
        self, attack_name: str, classifier, x, y_target, batch_size=32
    ) -> None:
        # Select a targeted attack
        self.y_target = y_target

        if attack_name.upper() == "FGSM":
            self.attack = FastGradientMethod(
                estimator=classifier, eps=0.2, targeted=True, batch_size=batch_size
            )
        elif attack_name.upper() == "PGD_L2":
            self.attack = ProjectedGradientDescent(
                estimator=classifier,
                norm=2,
                eps=0.5,
                eps_step=0.2,
                targeted=True,
                batch_size=batch_size,
                verbose=True,
            )
        elif attack_name.upper() == "PGD_LINF":
            self.attack = ProjectedGradientDescent(
                estimator=classifier,
                norm=np.inf,
                eps=0.5,
                eps_step=0.2,
                targeted=True,
                batch_size=batch_size,
                verbose=True,
            )
        elif attack_name.upper() == "CW_L0":
            self.attack = CarliniL0Method(
                classifier=classifier,
                targeted=True,
                batch_size=batch_size,
                binary_search_steps=5,
                verbose=True,
            )
        elif attack_name.upper() == "CW_L2":
            self.attack = CarliniL2Method(
                classifier=classifier,
                targeted=True,
                batch_size=batch_size,
                binary_search_steps=5,
                verbose=True,
            )
        elif attack_name.upper() == "CW_LINF":
            self.attack = CarliniLInfMethod(
                classifier=classifier,
                targeted=True,
                batch_size=batch_size,
                verbose=True,
            )
        elif attack_name.upper() == "ZOO":
            self.attack = ZooAttack(
                classifier=classifier,
                targeted=True,
                batch_size=batch_size,
                verbose=True,
            )
        elif attack_name.upper() == "AUTOATTACK":
            self.attack = AutoAttack(estimator=classifier, batch_size=batch_size)
        else:
            raise NotImplementedError

        # Generate adversarial test examples
        self.x_adv = self.attack.generate(x=x, y=self.y_target)
