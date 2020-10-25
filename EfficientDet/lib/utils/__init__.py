from lib.utils.utils import invert_affine, aspectaware_resize_padding, preprocess, preprocess_video, postprocess, display, replace_w_sync_bn
from lib.utils.utils import CustomDataParallel
from lib.utils.utils import get_last_weights, init_weights, variance_scaling_, STANDARD_COLORS, from_colorname_to_bgr, standard_to_bgr, get_index_label, plot_one_box, color_list
from .sync_batchnorm import *