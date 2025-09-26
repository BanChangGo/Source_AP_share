import numpy as np
import os
from tensorflow.keras.datasets import fashion_mnist
import time
import json # 메타데이터 저장을 위해 json 라이브러리 사용

# ---------------------------
# (기존 헬퍼 함수들은 이전과 동일하게 유지)
# ---------------------------
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_integer(x, zero_point):
    return np.maximum(zero_point, x)

def maxpool2d_integer(x, pool_size=(2, 2), stride=2):
    H, W, C = x.shape; ph, pw = pool_size; sh, sw = stride, stride
    out_h, out_w = (H - ph) // sh + 1, (W - pw) // sw + 1
    out = np.zeros((out_h, out_w, C), dtype=x.dtype)
    for oh in range(out_h):
        for ow in range(out_w):
            region = x[oh*sh:oh*sh+ph, ow*sw:ow*sw+pw, :]
            out[oh, ow, :] = np.max(region, axis=(0, 1))
    return out

def load_params(param_dir):
    params = {}
    files = [f"layer_{i}.txt" for i in range(10)]
    param_names = ['W0', 'b0', 'W2', 'b2', 'W4', 'b4', 'W6', 'b6', 'W8', 'b8']
    for i, name in enumerate(param_names):
        path = os.path.join(param_dir, files[i])
        if 'W' in name:
            if i < 6:
                with open(path, "r") as f: fh, fw, cin, cout = map(int, f.readline().split())
                params[name] = np.loadtxt(path, skiprows=1).reshape(fh, fw, cin, cout)
            else:
                with open(path, "r") as f: din, dout = map(int, f.readline().split())
                params[name] = np.loadtxt(path, skiprows=1).reshape(din, dout)
        else:
            params[name] = np.loadtxt(path, skiprows=1)
    return params

def conv2d_float(x, W, b, stride=1, padding='valid'):
    H, W_in, Cin = x.shape; fh, fw, _, Cout = W.shape
    if padding == 'same':
        pad_h, pad_w = (fh - 1) // 2, (fw - 1) // 2
        x_padded = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')
    else: x_padded = x
    H_pad, W_pad, _ = x_padded.shape; out_h, out_w = (H_pad - fh) // stride + 1, (W_pad - fw) // stride + 1
    out, W_reshaped = np.zeros((out_h, out_w, Cout)), W.reshape(-1, Cout)
    for oh in range(out_h):
        for ow in range(out_w):
            region_flat = x_padded[oh*stride:oh*stride+fh, ow*stride:ow*stride+fw, :].flatten()
            out[oh, ow, :] = np.dot(region_flat, W_reshaped) + b
    return relu(out)
    
def dense_float(x, W, b, act):
    output = np.dot(x, W) + b
    if act == "relu": return relu(output)
    if act == "softmax": return softmax(output)
    return output

def lenet5_forward_float(x, params, intermediates):
    x = conv2d_float(x, params['W0'], params['b0'], padding='same'); intermediates['c1'] = x
    x = maxpool2d_integer(x); intermediates['s2'] = x
    x = conv2d_float(x, params['W2'], params['b2'], padding='valid'); intermediates['c3'] = x
    x = maxpool2d_integer(x); intermediates['s4'] = x
    x = conv2d_float(x, params['W4'], params['b4'], padding='valid'); intermediates['c5'] = x
    x = x.flatten(); intermediates['flatten'] = x
    x = dense_float(x, params['W6'], params['b6'], act="relu"); intermediates['f6'] = x
    x = dense_float(x, params['W8'], params['b8'], act="softmax"); intermediates['output'] = x
    return x
    
def quantize_tensor_symmetric(tensor, n_bits=8):
    f_min, f_max = np.min(tensor), np.max(tensor)
    abs_max = max(abs(f_min), abs(f_max))
    q_max = 2**(n_bits - 1) - 1
    scale = abs_max / q_max if q_max != 0 else 1e-9
    if scale == 0: scale = 1e-9
    zero_point = 0
    quantized_tensor = np.round(tensor / scale)
    quantized_tensor = np.clip(quantized_tensor, -q_max-1, q_max)
    return quantized_tensor.astype(np.int32), float(scale), int(zero_point)

def dequantize_tensor(q_tensor, scale, zero_point):
    return (q_tensor.astype(np.float32) - zero_point) * scale

def get_M_and_n(M_float):
    if M_float == 0: return 0, 0
    sign = np.sign(M_float)
    M_float = abs(M_float)
    n = int(np.floor(np.log2(M_float)))
    M_q = np.round(M_float * (2** (31 - n))).astype(np.int64)
    n_shift = 31 - n
    return int(sign * M_q), int(n_shift)

def conv2d_static_integer(q_x, q_W, q_b, Zx, Zw, Zb, Zy, M, n_shift, padding='valid', stride=1):
    H, W_in, Cin = q_x.shape; fh, fw, _, Cout = q_W.shape
    if padding == 'same':
        pad_h, pad_w = (fh - 1) // 2, (fw - 1) // 2
        q_x_padded = np.pad(q_x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant', constant_values=Zx)
    else: q_x_padded = q_x
    H_pad, W_pad, _ = q_x_padded.shape; out_h, out_w = (H_pad - fh) // stride + 1, (W_pad - fw) // stride + 1
    q_out = np.zeros((out_h, out_w, Cout), dtype=np.int8); W_reshaped = q_W.reshape(-1, Cout)
    for oh in range(out_h):
        for ow in range(out_w):
            region = q_x_padded[oh*stride : oh*stride+fh, ow*stride : ow*stride+fw, :]
            region_flat = (region.astype(np.int32) - Zx).flatten()
            w_flat = (W_reshaped.astype(np.int32) - Zw)
            accumulator = np.dot(region_flat, w_flat) + (q_b - Zb)
            scaled_acc = (accumulator.astype(np.int64) * M) >> n_shift
            q_out[oh, ow, :] = np.clip(scaled_acc + Zy, -128, 127)
    return q_out

def dense_static_integer(q_x, q_W, q_b, Zx, Zw, Zb, Zy, M, n_shift):
    accumulator = np.dot((q_x.astype(np.int32) - Zx), (q_W.astype(np.int32) - Zw)) + (q_b - Zb)
    scaled_acc = (accumulator.astype(np.int64) * M) >> n_shift
    q_out = np.clip(scaled_acc + Zy, -128, 127)
    return q_out

def lenet5_forward_static_integer(q_x, static_model):
    q_params, act_params, req_params = static_model['quant_params'], static_model['act_params'], static_model['requant_params']
    _, Zx = act_params['input']; q_W0, _, Zw0 = q_params['W0']; q_b0, _, Zb0 = q_params['b0']
    _, Zy_c1 = act_params['c1_out']; M1, n1 = req_params['c1']
    q_x = conv2d_static_integer(q_x, q_W0, q_b0, Zx, Zw0, Zb0, Zy_c1, M1, n1, padding='same'); q_x = relu_integer(q_x, Zy_c1)
    q_x = maxpool2d_integer(q_x)
    q_W2, _, Zw2 = q_params['W2']; q_b2, _, Zb2 = q_params['b2']; _, Zx_c3 = act_params['c1_out']
    _, Zy_c3 = act_params['c3_out']; M3, n3 = req_params['c3']
    q_x = conv2d_static_integer(q_x, q_W2, q_b2, Zx_c3, Zw2, Zb2, Zy_c3, M3, n3, padding='valid'); q_x = relu_integer(q_x, Zy_c3)
    q_x = maxpool2d_integer(q_x)
    q_W4, _, Zw4 = q_params['W4']; q_b4, _, Zb4 = q_params['b4']; _, Zx_c5 = act_params['c3_out']
    _, Zy_c5 = act_params['c5_out']; M5, n5 = req_params['c5']
    q_x = conv2d_static_integer(q_x, q_W4, q_b4, Zx_c5, Zw4, Zb4, Zy_c5, M5, n5, padding='valid'); q_x = relu_integer(q_x, Zy_c5)
    q_x = q_x.flatten()
    q_W6, _, Zw6 = q_params['W6']; q_b6, _, Zb6 = q_params['b6']; _, Zx_f6 = act_params['c5_out']
    _, Zy_f6 = act_params['f6_out']; M6, n6 = req_params['f6']
    q_x = dense_static_integer(q_x, q_W6, q_b6, Zx_f6, Zw6, Zb6, Zy_f6, M6, n6); q_x = relu_integer(q_x, Zy_f6)
    q_W8, _, Zw8 = q_params['W8']; q_b8, _, Zb8 = q_params['b8']; _, Zx_out = act_params['f6_out']
    Sy_out, Zy_out = act_params['output_out']; M_out, n_out = req_params['output']
    q_x = dense_static_integer(q_x, q_W8, q_b8, Zx_out, Zw8, Zb8, Zy_out, M_out, n_out)
    return q_x, (Sy_out, Zy_out)

# ---------------------------
# ✨ 1. 오프라인(Offline) 단계: 파일로 저장
# ---------------------------

def save_quantized_tensor_to_file(filepath, q_tensor, scale, zp):
    """양자화된 텐서를 헤더 정보(shape, scale, zp)와 함께 .txt 파일로 저장합니다."""
    with open(filepath, "w") as f:
        # 헤더 작성
        shape_str = " ".join(map(str, q_tensor.shape))
        f.write(f"{shape_str}\n")
        f.write(f"scale: {scale}\n")
        f.write(f"zero_point: {zp}\n")
        # 데이터 작성
        np.savetxt(f, q_tensor.flatten(), fmt='%d')

def offline_calibration_and_quantization(params_float, calibration_data, output_dir, n_bits=8):
    print(f"--- 1. 오프라인 캘리브레이션 및 양자화 시작 (결과 저장: {output_dir}) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # ... (함수 앞부분의 가중치 양자화 및 캘리브레이션 로직은 이전과 동일) ...
    # ... (생략) ...
    static_model = {'quant_params': {}, 'act_params': {}, 'requant_params': {}}
    
    # 1. 가중치 양자화 및 파일 저장 (이전과 동일)
    param_map = {'W0':0, 'b0':1, 'W2':2, 'b2':3, 'W4':4, 'b4':5, 'W6':6, 'b6':7, 'W8':8, 'b8':9}
    for name, float_tensor in params_float.items():
        q_tensor, scale, zp = quantize_tensor_symmetric(float_tensor, n_bits)
        static_model['quant_params'][name] = (q_tensor, scale, zp)
        filepath = os.path.join(output_dir, f"layer_{param_map[name]}_q.txt")
        save_quantized_tensor_to_file(filepath, q_tensor, scale, zp)
    
    # 2. 입력 및 활성화 스케일 보정 (이전과 동일)
    act_min_max = { 'input': [np.inf, -np.inf] }
    layer_names = ['c1', 'c3', 'c5', 'f6', 'output']
    for name in layer_names: act_min_max[name] = [np.inf, -np.inf]
    for i, x_calib in enumerate(calibration_data):
        act_min_max['input'][0] = min(act_min_max['input'][0], x_calib.min())
        act_min_max['input'][1] = max(act_min_max['input'][1], x_calib.max())
        intermediates = {}
        lenet5_forward_float(x_calib, params_float, intermediates)
        for name in layer_names:
            tensor = intermediates[name]
            act_min_max[name][0] = min(act_min_max[name][0], tensor.min())
            act_min_max[name][1] = max(act_min_max[name][1], tensor.max())
    _, s_x, z_x = quantize_tensor_symmetric(np.array(act_min_max['input']), n_bits)
    static_model['act_params']['input'] = (s_x, z_x)
    for name in layer_names:
        _, scale, zp = quantize_tensor_symmetric(np.array(act_min_max[name]), n_bits)
        static_model['act_params'][name + '_out'] = (scale, zp)
        
    # 3. HW 친화적 재양자화 배율(M, n) 미리 계산 (이전과 동일)
    layer_info = [('c1', 'W0', 'input', 'c1_out'), ('c3', 'W2', 'c1_out', 'c3_out'), ('c5', 'W4', 'c3_out', 'c5_out'), ('f6', 'W6', 'c5_out', 'f6_out'), ('output', 'W8', 'f6_out', 'output_out')]
    for name, w_name, in_act, out_act in layer_info:
        s_in, _ = static_model['act_params'][in_act]
        _, s_w, _ = static_model['quant_params'][w_name]
        s_out, _ = static_model['act_params'][out_act]
        M_float = (s_in * s_w) / s_out
        M, n_shift = get_M_and_n(M_float)
        static_model['requant_params'][name] = (M, n_shift)

    # ✨✨✨ 변경된 부분 시작 ✨✨✨
    # 4. 전역 파라미터를 quant_info.txt 파일로 저장
    meta_path = os.path.join(output_dir, "quant_info.txt")
    with open(meta_path, 'w') as f:
        # act_params 저장
        for key, (scale, zp) in static_model['act_params'].items():
            f.write(f"act_param {key} {scale} {zp}\n")
            
        # requant_params 저장
        for key, (M, n) in static_model['requant_params'].items():
            f.write(f"requant_param {key} {M} {n}\n")
    # ✨✨✨ 변경된 부분 끝 ✨✨✨
        
    print("--- 오프라인 준비 및 파일 저장 완료 ---")

# ✨ 새로 추가할 함수
def save_images_to_txt(images, labels, output_dir, num_to_save):
    """테스트 이미지들을 개별 .txt 파일로 저장하는 함수"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- 테스트 이미지 {num_to_save}개를 .txt 파일로 저장 시작 (저장 경로: {output_dir}) ---")
    for i in range(num_to_save):
        image = images[i]
        label = labels[i]
        # 파일 이름에 인덱스와 정답 레이블을 포함시켜 확인하기 쉽게 만듭니다.
        filepath = os.path.join(output_dir, f"image_{i}_label_{label}.txt")
        with open(filepath, "w") as f:
            # 헤더: Shape 정보
            shape_str = " ".join(map(str, image.shape))
            f.write(f"{shape_str}\n")
            # 데이터 저장
            np.savetxt(f, image.flatten(), fmt='%.8f')
    print("--- 이미지 저장 완료 ---")

# ---------------------------
# ✨ 2. 온라인(Online) 단계: 파일에서 로드 ---> C++에서 구현
# ---------------------------


if __name__ == "__main__":
    # --- 데이터 및 파라미터 경로 설정 ---
    fp32_param_dir = "./SLeNet-5_Parameter/"
    quantized_param_dir = "./Static_INT8_Params/" # 양자화된 파라미터 저장/로드 경로
    test_images_dir = "./Test_Images_TXT/"       # .txt로 변환된 테스트 이미지 저장 경로

    # --- 데이터 로드 ---
    (_, _), (x_test, y_test) = fashion_mnist.load_data()
    x_test_float = np.expand_dims(x_test, -1).astype(np.float32) / 255.0

    if not os.path.exists(fp32_param_dir):
        print(f"Error: FP32 파라미터 디렉토리 '{fp32_param_dir}'를 찾을 수 없습니다.")
    else:
        # --- 1. 오프라인(Offline) 단계 실행: 파라미터 양자화 및 저장 ---
        params_float = load_params(fp32_param_dir)
        calibration_data = x_test_float[:100]
        offline_calibration_and_quantization(params_float, calibration_data, quantized_param_dir)

        # --- 2. 테스트 데이터셋을 .txt 파일로 저장 ---
        # C++에서 사용할 수 있도록 테스트 이미지 100개를 미리 변환해 둡니다.
        save_images_to_txt(x_test_float, y_test, test_images_dir, num_to_save=10000)
        
        
        

        print(f"\n이제 C++ 엔진을 사용하여 '{test_images_dir}'에 저장된 이미지로 추론을 실행할 수 있습니다.")