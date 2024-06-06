import numpy as np
import shutil
from pathlib import Path
from PIL import Image, ImageFilter

from scipy.special import softmax
from skimage.metrics import structural_similarity as SSIM


class RobustnessMetrics:
    def __init__(
        self,
        classifier,
        predictions,
        x_test,
        x_test_adv,
        y_test,
        y_target,
        targeted,
        dataset,
        save_dir,
    ) -> None:
        self.classifier = classifier
        self.softmax = softmax(predictions)
        self.x_test = x_test
        self.x_test_adv = x_test_adv
        self.y_test = y_test
        self.y_target = y_target
        self.target_flag = targeted
        self.dataset = dataset
        self.save_dir = Path(save_dir)

    ### Misclassification
    # https://github.com/ryderling/DEEPSEC/blob/master/Evaluations/AttackEvaluations.py#L315
    # 1 MR:Misclassification Rate
    def misclassification_rate(self):
        cnt = 0
        for i in range(len(self.x_test_adv)):
            if self.successful(
                adv_softmax_preds=self.softmax[i],
                nature_true_preds=self.y_test[i],
                targeted_preds=self.y_target[i],
                target_flag=False,
            ):
                cnt += 1
        mr = cnt / len(self.x_test_adv)
        # print('MR:\t\t{:.1f}%'.format(mr * 100))
        return mr

    # 2 ACAC: average confidence of adversarial class
    def avg_confidence_adv_class(self):
        cnt = 0
        conf = 0
        for i in range(len(self.x_test_adv)):
            if self.successful(
                adv_softmax_preds=self.softmax[i],
                nature_true_preds=self.y_test[i],
                targeted_preds=self.y_target[i],
                target_flag=False,
            ):
                cnt += 1
                conf += np.max(self.softmax[i])

        # print('ACAC:\t{:.3f}'.format(conf / cnt))
        return conf / cnt

    # 3 ACTC: average confidence of true class
    def avg_confidence_true_class(self):
        true_labels = np.argmax(self.y_test, axis=1)
        cnt = 0
        true_conf = 0
        for i in range(len(self.x_test_adv)):
            if self.successful(
                adv_softmax_preds=self.softmax[i],
                nature_true_preds=self.y_test[i],
                targeted_preds=self.y_target[i],
                target_flag=False,
            ):
                cnt += 1
                true_conf += self.softmax[i, true_labels[i]]
        # print('ACTC:\t{:.3f}'.format(true_conf / cnt))
        return true_conf / cnt

    ### Imperceptibility
    # 4 ALP: Average L_p Distortion
    def avg_lp_distortion(self):

        ori_r = np.round(self.x_test * 255)
        adv_r = np.round(self.x_test_adv * 255)

        NUM_PIXEL = int(np.prod(self.x_test.shape[1:]))

        pert = adv_r - ori_r

        dist_l0 = 0
        dist_l2 = 0
        dist_li = 0

        cnt = 0

        for i in range(len(self.x_test_adv)):
            if self.successful(
                adv_softmax_preds=self.softmax[i],
                nature_true_preds=self.y_test[i],
                targeted_preds=self.y_target[i],
                target_flag=False,
            ):
                cnt += 1
                dist_l0 += np.linalg.norm(np.reshape(pert[i], -1), ord=0) / NUM_PIXEL
                dist_l2 += np.linalg.norm(
                    np.reshape(self.x_test[i] - self.x_test_adv[i], -1), ord=2
                )
                dist_li += np.linalg.norm(
                    np.reshape(self.x_test[i] - self.x_test_adv[i], -1), ord=np.inf
                )

        adv_l0 = dist_l0 / cnt
        adv_l2 = dist_l2 / cnt
        adv_li = dist_li / cnt

        # print('**ALP:**\n\tL0:\t{:.3f}\n\tL2:\t{:.3f}\n\tLi:\t{:.3f}'.format(adv_l0, adv_l2, adv_li))
        return adv_l0, adv_l2, adv_li

    # 5 ASS: Average Structural Similarity
    def avg_SSIM(self):
        ori_r_channel = np.transpose(np.round(self.x_test * 255), (0, 2, 3, 1)).astype(
            dtype=np.float32
        )
        adv_r_channel = np.transpose(
            np.round(self.x_test_adv * 255), (0, 2, 3, 1)
        ).astype(dtype=np.float32)

        totalSSIM = 0
        cnt = 0

        """
        For SSIM function in skimage: http://scikit-image.org/docs/dev/api/skimage.measure.html

        multichannel : bool, optional If True, treat the last dimension of the array as channels. Similarity calculations are done
        independently for each channel then averaged.
        """
        for i in range(len(self.x_test_adv)):
            if self.successful(
                adv_softmax_preds=self.softmax[i],
                nature_true_preds=self.y_test[i],
                targeted_preds=self.y_target[i],
                target_flag=False,
            ):
                cnt += 1
                totalSSIM += SSIM(
                    im1=ori_r_channel[i],
                    im2=adv_r_channel[i],
                    data_range=255,
                    channel_axis=2,
                )

        # print('ASS:\t{:.3f}'.format(totalSSIM / cnt))
        return totalSSIM / cnt

    ### Robustness
    # 6 NTE: Noise Tolerance Estimation
    def avg_noise_tolerance_estimation(self):
        nte = 0
        cnt = 0
        for i in range(len(self.x_test_adv)):
            if self.successful(
                adv_softmax_preds=self.softmax[i],
                nature_true_preds=self.y_test[i],
                targeted_preds=self.y_target[i],
                target_flag=False,
            ):
                cnt += 1
                sort_preds = np.sort(self.softmax[i])
                nte += sort_preds[-1] - sort_preds[-2]

        # print('NTE:\t{:.3f}'.format(nte / cnt))
        return nte / cnt

    # 7 RGB: Robustness to Gaussian Blur
    def robust_gaussian_blur(self, radius=0.5):
        total = 0
        num_gb = 0

        if self.target_flag is True:
            for i in range(len(self.x_test_adv)):
                if np.argmax(self.softmax[i]) == np.argmax(self.y_target[i]):
                    total += 1
                    adv_sample = self.x_test_adv[i]
                    gb_sample = self.gaussian_blur_transform(
                        AdvSample=adv_sample, radius=radius, oriDataset=self.dataset
                    )
                    gb_pred = self.classifier.predict(np.array([gb_sample]))
                    if np.argmax(gb_pred) == np.argmax(self.y_target[i]):
                        num_gb += 1

        else:
            for i in range(len(self.x_test_adv)):
                if np.argmax(self.softmax[i]) != np.argmax(self.y_test[i]):
                    total += 1
                    adv_sample = self.x_test_adv[i]
                    gb_sample = self.gaussian_blur_transform(
                        AdvSample=adv_sample, radius=radius, oriDataset=self.dataset
                    )
                    gb_pred = self.classifier.predict(np.array([gb_sample]))
                    if np.argmax(gb_pred) != np.argmax(self.y_test[i]):
                        num_gb += 1

        # print('RGB:\t{:.3f}'.format(num_gb / total))
        return num_gb / total

    # 8 RIC: Robustness to Image Compression
    def robust_image_compression(self, quality=0.1):
        total = 0
        num_ic = 0

        # prepare the save dir for the generated image(png or jpg)
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.target_flag is True:
            for i in range(len(self.x_test_adv)):
                if np.argmax(self.softmax[i]) == np.argmax(self.y_target[i]):
                    total += 1
                    adv_sample = self.x_test_adv[i]

                    ic_sample = self.image_compress_transform(
                        IndexAdv=i,
                        AdvSample=adv_sample,
                        dir_name=self.save_dir,
                        quality=quality,
                        oriDataset=self.dataset,
                    )

                    ic_pred = self.classifier.predict(np.array([ic_sample]))
                    if np.argmax(ic_pred) == np.argmax(self.y_target[i]):
                        num_ic += 1

        else:
            for i in range(len(self.x_test_adv)):
                if np.argmax(self.softmax[i]) != np.argmax(self.y_test[i]):
                    total += 1
                    adv_sample = self.x_test_adv[i]

                    ic_sample = self.image_compress_transform(
                        IndexAdv=i,
                        AdvSample=adv_sample,
                        dir_name=self.save_dir,
                        quality=quality,
                        oriDataset=self.dataset,
                    )

                    ic_pred = classifier.predict(np.array([ic_sample]))
                    if np.argmax(ic_pred) != np.argmax(self.y_test[i]):
                        num_ic += 1
        # print('RIC:\t{:.3f}'.format(num_ic / total))
        return num_ic / total

    # help function
    def successful(
        self, adv_softmax_preds, nature_true_preds, targeted_preds, target_flag=False
    ):
        """

        :param adv_softmax_preds: the softmax prediction for the adversarial example
        :param nature_true_preds: for the un-targeted attack, it should be the true label for the nature example
        :param targeted_preds: for the targeted attack, it should be the specified targets label that selected
        :param target_flag: True if it is a targeted attack, False if it is a un-targeted attack
        :return:
        """
        if target_flag:
            if np.argmax(adv_softmax_preds) == np.argmax(targeted_preds):
                return True
            else:
                return False
        else:
            if np.argmax(adv_softmax_preds) != np.argmax(nature_true_preds):
                return True
            else:
                return False

    # help function for the Gaussian Blur transformation of images
    def gaussian_blur_transform(self, AdvSample, radius, oriDataset):
        if oriDataset.upper() == "CIFAR10":
            assert AdvSample.shape == (3, 32, 32)
            sample = np.transpose(np.round(AdvSample * 255), (1, 2, 0))

            image = Image.fromarray(np.uint8(sample))
            gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            gb_image = (
                np.transpose(np.array(gb_image), (2, 0, 1)).astype("float32") / 255.0
            )
            return gb_image

        if oriDataset.upper() == "MNIST":
            assert AdvSample.shape == (1, 28, 28)
            sample = np.transpose(np.round(AdvSample * 255), (1, 2, 0))
            # for MNIST, there is no RGB
            sample = np.squeeze(sample, axis=2)

            image = Image.fromarray(np.uint8(sample))
            gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))

            gb_image = (
                np.expand_dims(np.array(gb_image).astype("float32"), axis=0) / 255.0
            )
            return gb_image

    # help function for the image compression transformation of images
    def image_compress_transform(
        self, IndexAdv, AdvSample, dir_name, quality, oriDataset
    ):
        if oriDataset.upper() == "CIFAR10":
            assert AdvSample.shape == (3, 32, 32)
            sample = np.transpose(np.round(AdvSample * 255), (1, 2, 0))
            image = Image.fromarray(np.uint8(sample))

            saved_adv_image_path = Path(dir_name, "{}th-adv-cifar.png".format(IndexAdv))
            image.save(saved_adv_image_path, quality=quality, subsampling=0)

            IC_image = Image.open(saved_adv_image_path).convert("RGB")
            IC_image = (
                np.transpose(np.array(IC_image), (2, 0, 1)).astype("float32") / 255.0
            )
            return IC_image

        if oriDataset.upper() == "MNIST":
            assert AdvSample.shape == (1, 28, 28)
            sample = np.transpose(np.round(AdvSample * 255), (1, 2, 0))
            sample = np.squeeze(sample, axis=2)  # for MNIST, there is no RGB
            image = Image.fromarray(np.uint8(sample), mode="L")

            saved_adv_image_path = Path(dir_name, "{}th-adv-mnist.png".format(IndexAdv))
            image.save(saved_adv_image_path, quality=quality, subsampling=0)

            IC_image = Image.open(saved_adv_image_path).convert("L")
            IC_image = (
                np.expand_dims(np.array(IC_image).astype("float32"), axis=0) / 255.0
            )
            return IC_image
