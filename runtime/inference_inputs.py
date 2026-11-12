"""Use the unchanged native inference renderer with explicit user regions."""
import random
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import ImageFont
from common import VENDOR
sys.path.insert(0, str(VENDOR / 'eval'))
from eval.anytext_inference import T3DataSetWarp
from eval import t3_dataset
from font_support import BASE, font_path, install_fallback

install_fallback(t3_dataset)


class RequestDataset(T3DataSetWarp):
    def __init__(self, request):
        # Construct the native input dictionary directly. No benchmark download,
        # caption filtering, random region selection, or text truncation.
        regions = request['regions']
        for region in regions:
            font_path(region['text'])
        self.data_list = [dict(img_path=request['image'], caption=request['prompt'],
            polygons=[np.array(r['polygon'], dtype=np.int32) for r in regions],
            invalid_polygons=[], texts=[r['text'] for r in regions],
            language=[r['language'] for r in regions], pos=[-1]*len(regions), valid_img=True)]
        self.condition_size = self.target_size = 512
        self.condition_type, self.glyph_scale = 'word_fill', 1
        self.drop_text_prob = self.drop_image_prob = 0
        self.return_pil_image = self.debug = self.for_show = self.using_dlc = self.use_filter = False
        self.max_lines = 8
        self.max_chars = max(len(r['text']) for r in regions)
        self.place_holder = '*'
        self.caption_pos_porb, self.mask_pos_prob, self.mask_img_prob = 0., 1., 0.
        self.to_tensor = T.ToTensor()
        self.font = ImageFont.truetype(str(BASE), 60)


def prepare_input(request):
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(request['seed'])
        np.random.seed(request['seed'] % 2**32)
        data = RequestDataset(request)[0]
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    if data['texts'][:data['n_lines']] != [r['text'] for r in request['regions']]:
        raise RuntimeError('Native renderer changed the requested text')
    if tuple(data['image'].shape) != (3, 512, 512):
        raise RuntimeError('Unexpected native image tensor shape')
    return data
