import io, math, struct
import torch
from pathlib import Path
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def get_state_dict(ckpt_path, need="g"):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    
    if need == "g":
        ckpt = ckpt["net_g"]
        consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
        return ckpt
    else:
        consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
        return ckpt


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_ushorts(fd, n, fmt=">{:d}H"):
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_to_file(value, file_path):
    with Path(file_path).open("wb") as f:
        f.write(value)
        
        
def read_from_file(file_path):
    with Path(file_path).open("rb") as f:
        return f.read()
    

def encode_i(pic_height, pic_width, bit_stream_y, bit_stream_z, 
             bit_stream_caption, caption_length):
    buffer = io.BytesIO()
    stream_length = len(bit_stream_y)
    write_uints(buffer, (pic_height, pic_width))
    write_uints(buffer, (stream_length,))
    write_uints(buffer, (caption_length,))
    write_bytes(buffer, bit_stream_y)
    write_bytes(buffer, bit_stream_z)
    write_bytes(buffer, bit_stream_caption)
    return buffer.getvalue()
    

def decode_i(data, index_unit_length, ds):
    buffer = io.BytesIO(data)
    height, width = read_uints(buffer, 2)
    (stream_length,) = read_uints(buffer, 1)
    (caption_length,) = read_uints(buffer, 1)
    
    # processing padding
    padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, p=ds)
    pad_H = height + padding_t + padding_b
    pad_W = width + padding_l + padding_r
    stream_length_z = math.ceil((pad_H // ds) * (pad_W // ds) * index_unit_length / 8.)
    
    bit_stream_y = read_bytes(buffer, stream_length)
    bit_stream_z = read_bytes(buffer, stream_length_z)
    bit_stream_caption = read_bytes(buffer, caption_length)
    return {
        "height": height,
        "width": width,
        "pad_height": pad_H,
        "pad_width": pad_W,
        "pad_tuple": (padding_l, padding_r, padding_t, padding_b),
        "bit_stream_y": bit_stream_y,
        "bit_stream_z": bit_stream_z,
        "bit_stream_caption": bit_stream_caption
    }
    
    
def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom
