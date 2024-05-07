from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL0Method, CarliniL2Method, CarliniLInfMethod, ZooAttack, AutoAttack


class TargetedAttacks:
    def __init__(self, attack_name: str, classifier, x, y_target) -> None:
        # Select a targeted attack
        self.y_target = y_target

        if attack_name.upper() == "FGSM":
            self.attack = FastGradientMethod(estimator=classifier, eps=0.2, targeted=True)
        elif attack_name.upper() == "PGD_L2":
            self.attack = ProjectedGradientDescent(estimator=classifier, norm=2, eps=0.5, eps_step=0.2, targeted=True, verbose=True)
        elif attack_name.upper() == "PGD_Linf":
            self.attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=0.5, eps_step=0.2, targeted=True, verbose=True)
        elif attack_name.upper() == "CW_L0":
            self.attack = CarliniL0Method(classifier=classifier, targeted=True, verbose=True)
        elif attack_name.upper() == "CW_L2":
            self.attack = CarliniL2Method(classifier=classifier, targeted=True, verbose=True)
        elif attack_name.upper() == "CW_Linf":
            self.attack = CarliniLInfMethod(classifier=classifier, targeted=True, verbose=True)
        elif attack_name.upper() == "ZOO":
            self.attack = ZooAttack(classifier=classifier, targeted=True, verbose=True)
        elif attack_name.upper() == "AUTOATTACK":
            self.attack = AutoAttack(estimator=classifier, targeted=True, verbose=True)

        # Generate adversarial test examples
        self.x_adv = self.attack.generate(x=x, y=self.y_target)