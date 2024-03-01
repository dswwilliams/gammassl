import torch
import random
from functools import partial
import math
import numpy as np

class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
    

def random_masking_generator(n_tokens, p=0.5):
    # generate a mask of size n_tokens with each pixel having a probability of p to be masked, i.e. mask.mean() = p
    return torch.BoolTensor(np.random.choice([0, 1], size=(n_tokens,), p=[1-p, p]))
    

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None, masking_type="block", random_mask_prob=0.5):
    """

    - sample_list: [batch_el_0, batch_el_1, ... batch_el_bs]
    - batch_el_i: tuple(labelled_dict, raw_dict)
    - labelled_dict = {'img', 'label', 'box_A'}


    - collated_masks: (B, N), i.e. batch_size x num_tokens (16*16 for 224, patch_size=14)

    
    """

    collated_labelled_dict = {}
    for key in samples_list[0][0].keys():
        collated_labelled_dict[key] = torch.stack([s[0][key] for s in samples_list])
    collated_raw_dict = {}
    for key in samples_list[0][1].keys():
        collated_raw_dict[key] = torch.stack([s[1][key] for s in samples_list])


    if masking_type == "block":
        B = len(samples_list)
        N = n_tokens
        n_samples_masked = int(B * mask_probability)
        probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
        upperbound = 0
        masks_list = []
        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)
        for i in range(n_samples_masked, B):
            masks_list.append(torch.BoolTensor(mask_generator(0)))
        mask_indices_list = collated_masks.flatten().nonzero().flatten()
        masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    else:
        B = len(samples_list)
        N = n_tokens
        masks_list = []
        for i in range(0, B):
            masks_list.append(random_masking_generator(N, random_mask_prob))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)


    # add collated_masks to collated_raw_dict
    collated_raw_dict['mask'] = collated_masks

    return (collated_labelled_dict, collated_raw_dict)

def val_collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None, masking_type="block", random_mask_prob=0.5):
    """

    - sample_list: [batch_el_0, batch_el_1, ... batch_el_bs]
    - batch_el_i: val_dict
    - val_dict = {'img', 'label', 'name'}

    - collated_masks: (B, N), i.e. batch_size x num_tokens (16*16 for 224, patch_size=14)

    """

    collated_val_dict = {}
    for key in samples_list[0].keys():      # get keys of first batch element
        if isinstance(samples_list[0][key], str):
            collated_val_dict[key] = [s[key] for s in samples_list]
        else:
            collated_val_dict[key] = torch.stack([s[key] for s in samples_list])


    if masking_type == "block":
        B = len(samples_list)
        N = n_tokens
        n_samples_masked = int(B * mask_probability)
        probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
        upperbound = 0
        masks_list = []
        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)
        for i in range(n_samples_masked, B):
            masks_list.append(torch.BoolTensor(mask_generator(0)))
    else:
        B = len(samples_list)
        N = n_tokens
        masks_list = []
        for i in range(0, B):
            masks_list.append(random_masking_generator(N, random_mask_prob))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)

    # add collated_masks to collated_raw_dict
    collated_val_dict['mask'] = collated_masks

    return collated_val_dict



def get_collate_fn(img_size, patch_size, dtype=torch.float32, random_mask_prob=None):
    if random_mask_prob is not None:
        masking_type = "random"
    else:
        masking_type = "block"
    mask_generator = MaskingGenerator(
                            input_size=(img_size // patch_size, img_size // patch_size),
                            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
                            )
    
    return partial(
        collate_data_and_cast,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        n_tokens=(img_size // patch_size) ** 2,
        mask_generator=mask_generator,
        dtype=dtype,
        masking_type=masking_type,
        random_mask_prob=random_mask_prob,
        )

def get_val_collate_fn(img_size, patch_size, dtype=torch.float32, random_mask_prob=None):
    if random_mask_prob is not None:
        masking_type = "random"
    else:
        masking_type = "block"
    if not (isinstance(img_size, tuple) or isinstance(img_size, list)):
        img_size = (img_size, img_size)

    input_size = (img_size[0]//patch_size, img_size[1]//patch_size)
    mask_generator = MaskingGenerator(
                            input_size=input_size,
                            max_num_patches=0.5 * img_size[0] // patch_size * img_size[1] // patch_size,
                            )

    return partial(
        val_collate_data_and_cast,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        n_tokens=img_size[0] // patch_size * img_size[1] // patch_size,
        mask_generator=mask_generator,
        dtype=dtype,
        masking_type=masking_type,
        random_mask_prob=random_mask_prob,
        )
