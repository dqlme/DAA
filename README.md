# DAA
A novel two-stage black-box attack framework, termed double adversarial attack (DAA), combining generative perturbation via diffusion models with frequency-domain input transformation to enhance transferability and stealth.
# Abstract
Adversarial attacks have exposed the susceptibility of deep neural networks (DNNs) to malicious perturbations, particularly in black-box scenarios where model internals remain inaccessible. Though effective in transferability, existing restricted black-box attacks are often limited in diversity due to strict perturbation norms. In contrast, unrestricted attacks facilitated by generative models offer higher diversity but suffer from poor transferability and high computational costs.
	This paper proposes a novel two-stage black-box attack framework, termed double adversarial attack (DAA), combining generative perturbation via diffusion models with frequency-domain input transformation to enhance transferability and stealth. Specifically, we first leverage a pretrained diffusion model to generate high-fidelity adversarial examples by perturbing latent variables during denoising. To reduce computational overhead, we approximate gradients using the skip gradient method on the final timestep. We perform a frequency-domain transformation on the generated adversarial examples in the second stage using the discrete wavelet transform. We add noise to the low-frequency components to disrupt high-level semantic features and enhance cross-model transferability. Extensive experiments and ablation studies demonstrate that our approach significantly outperforms state-of-the-art methods in both attack success rate and robustness against advanced defenses.
		
![这是图片](https://github.com/dqlme/DAA/blob/main/fig1.png "DAA framework") 

# Attack Evaluation
* The adversarial examples generated using InceptionV3 as the surrogate model can be found at the following link: [DAA_INCV3](https://www.dropbox.com/scl/fi/n8km9t9j1e1p2ihicg69x/DAA_INCV3.zip?rlkey=v6k1y9lehsxp4uc7755ol5kpx&st=hfrsyya3&dl=0)
* You can reproduce the results of Table 1 by downloading the dataset and running the following code.
  、python attack_eval.py --img_path DAA_INCV3`
