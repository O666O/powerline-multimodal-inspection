import os
import torch
import faiss
import numpy as np
from PIL import Image
import json
import requests
import cv2
import gradio as gr
import sqlite3
import hashlib
import csv
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ====================== 【必填配置】 ======================
# 1. 豆包API配置（请确认API_KEY和ENDPOINT_ID完全正确）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ULTRALYTICS_CONFIG_DIR = os.path.join(PROJECT_ROOT, "runs", ".ultralytics")
os.makedirs(ULTRALYTICS_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", ULTRALYTICS_CONFIG_DIR)

# Never store a real key in source code. Set DOUBAO_API_KEY in the environment.
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "").strip()
DOUBAO_ENDPOINT_ID = os.getenv("DOUBAO_ENDPOINT_ID", "doubao-seed-2-0-mini-260215")
API_BASE_URL = os.getenv(
    "DOUBAO_API_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
)

# 2. 微调模型路径
FINETUNED_MODEL_PATH = os.getenv(
    "CHINESE_CLIP_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "runs", "chinese_clip", "best"),
)

# 3. 你的数据集路径
TRAIN_IMG_FOLDER = os.path.join(PROJECT_ROOT, "chinese_clip_dataset")
TRAIN_JSON_PATH = os.path.join(TRAIN_IMG_FOLDER, "train_pairs.jsonl")

# 4. 类别ID→输电设备名称映射（和你的JSON里的categories完全对应）
CATEGORY_ID_TO_NAME = {
    1: "轭铁",
    2: "轭铁悬挂",
    3: "间隔棒",
    4: "防振锤",
    5: "悬式绝缘子",
    6: "针式绝缘子",
    7: "复合绝缘子",
    8: "避雷器",
    9: "盘形悬式绝缘子",
    10: "输电杆塔",
    11: "输电导线",
    12: "接地装置",
    13: "电力变压器",
    14: "隔离开关",
    15: "高压断路器"
}

# 5. 索引文件保存路径
INDEX_SAVE_DIR = os.path.join(PROJECT_ROOT, "runs", "system_index")
IMG_INDEX_PATH = os.path.join(INDEX_SAVE_DIR, "img_faiss_index_insplad.bin")
IMG_ID_LIST_PATH = os.path.join(INDEX_SAVE_DIR, "img_id_list_insplad.txt")
IMG_PATH_MAP_PATH = os.path.join(INDEX_SAVE_DIR, "img_path_map_insplad.json")
TEXT_INDEX_PATH = os.path.join(INDEX_SAVE_DIR, "text_faiss_index_insplad.bin")
TEXT_LIST_PATH = os.path.join(INDEX_SAVE_DIR, "text_list_insplad.txt")

# 6. 强制重建索引开关：第一次运行设为True，后续运行设为False
FORCE_REBUILD_INDEX = False

# 7. RT-DETRDefect Detection模型权重路径
DEFECT_MODEL_PATH = os.getenv(
    "DEFECT_MODEL_PATH",
    os.path.join(
        PROJECT_ROOT,
        "runs",
        "baselines",
        "baseline_yolov8_l",
        "weights",
        "best.pt",
    ),
)
# 缺陷检测默认阈值（前端不再展示滑条）
DEFECT_CONF_THRES = float(os.getenv("DEFECT_CONF_THRES", "0.3"))

# ====================== 【新增：全局请求会话-带自动重试】 ======================
def create_retry_session():
    """创建带指数退避自动重试的requests会话，解决网络波动超时问题"""
    retry_strategy = Retry(
        total=3,  # 最多重试3次
        read=True,  # 读取超时允许重试
        connect=True,  # 连接超时允许重试
        backoff_factor=1,  # 重试间隔：1秒 → 2秒 → 4秒 指数退避
        status_forcelist=(429, 500, 502, 503, 504),  # 仅对这些错误重试
        allowed_methods=["POST"],  # 允许POST请求重试（API调用是POST）
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# 全局初始化重试会话
REQUEST_SESSION = create_retry_session()

# ====================== 【全局变量】 ======================
model = None
processor = None
preprocess = None
tokenize = None
img_index = None
img_id_list = []
img_id2path = {}
text_index = None
text_list = []
defect_detector = None
resources_loaded = False
loaded_use_finetuned = None
display_metadata = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ====================== 【第一部分：SQLite用户功能】 ======================
# 使用与项目同目录下的固定 SQLite 文件，避免不同启动目录导致历史查不到
DB_PATH = os.path.join(os.path.dirname(__file__), "user_db.sqlite")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            create_time TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            search_type TEXT NOT NULL,
            query_content TEXT NOT NULL,
            result_summary TEXT,
            create_time TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def get_user_id(username):
    if not username:
        return None
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user[0] if user else None

def register_user(username, password, confirm_password):
    if not username or not password:
        return "用户名和密码不能为空", False
    if len(username) < 3:
        return "❌ 用户名长度至少3位", False
    if len(password) < 6:
        return "❌ 密码长度至少6位", False
    if password != confirm_password:
        return "❌ 两次输入的密码不一致", False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return "❌ 该用户名已被注册，请换一个", False
        
        cursor.execute(
            "INSERT INTO users (username, password_hash, create_time) VALUES (?, ?, ?)",
            (username, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        return f"注册成功！用户名：{username}，现在可以登录了", True
    except Exception as e:
        return f"❌ 注册失败：{str(e)}", False

def login_user(username, password):
    if not username or not password:
        return "❌ 用户名和密码不能为空", False, ""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM users WHERE username = ? AND password_hash = ?",
            (username, hash_password(password))
        )
        user = cursor.fetchone()
        conn.close()
        if user:
            return f"✅ 登录成功！欢迎回来，{username}", True, username
        else:
            return "❌ 用户名或密码错误", False, ""
    except Exception as e:
        return f"❌ 登录失败：{str(e)}", False, ""

def save_history(username, search_type, query_content, result_summary):
    try:
        user_id = get_user_id(username)
        if not user_id:
            return
        summary = result_summary[:100] + "..." if len(result_summary) > 100 else result_summary
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO search_history (user_id, search_type, query_content, result_summary, create_time) VALUES (?, ?, ?, ?, ?)",
            (user_id, search_type, query_content, summary, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"历史记录保存失败：{e}")

def get_user_display_history(username, search_type="全部", keyword="", limit=200, days=0):
    try:
        # 优先使用当前会话用户；如果传入空值，直接返回空结果，避免界面闪退/掉线
        if not username:
            return [["请先登录", "请先登录", "请先登录", "请先登录"]]

        user_id = get_user_id(username)
        if not user_id:
            return [["未找到用户", "用户不存在或未登录", "用户不存在或未登录", "用户不存在或未登录"]]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        sql = """
            SELECT search_type, query_content, result_summary, create_time
            FROM search_history
            WHERE user_id = ?
        """
        params = [user_id]

        if search_type and search_type not in ("全部", "All"):
            sql += " AND search_type = ?"
            params.append(search_type)

        days = int(days or 0)
        if days > 0:
            start_dt = (datetime.now() - timedelta(days=days-1)).strftime("%Y-%m-%d 00:00:00")
            sql += " AND create_time >= ?"
            params.append(start_dt)

        keyword = (keyword or "").strip()
        if keyword:
            sql += " AND (query_content LIKE ? OR result_summary LIKE ?)"
            like_kw = f"%{keyword}%"
            params.extend([like_kw, like_kw])

        safe_limit = max(1, min(int(limit), 1000))
        sql += " ORDER BY create_time DESC LIMIT ?"
        params.append(safe_limit)

        cursor.execute(sql, tuple(params))
        history = cursor.fetchall()
        conn.close()

        if not history:
            return [["暂无记录", "暂无记录", "暂无记录", "暂无记录"]]
        return history
    except Exception as e:
        print(f"历史记录Query failed：{e}")
        return [["查询失败", "查询失败", str(e), "查询失败"]]

def delete_latest_history(username):
    try:
        user_id = get_user_id(username)
        if not user_id:
            return "❌ 用户未登录，无法删除", get_user_display_history(username)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM search_history WHERE user_id = ? ORDER BY create_time DESC LIMIT 1",
            (user_id,)
        )
        latest = cursor.fetchone()
        if not latest:
            conn.close()
            return "❌ 没有可删除的历史记录", get_user_display_history(username)
        cursor.execute("DELETE FROM search_history WHERE id = ?", (latest[0],))
        conn.commit()
        conn.close()
        return "✅ 最新一条历史记录已删除", get_user_display_history(username)
    except Exception as e:
        return f"❌ 删除失败：{str(e)}", get_user_display_history(username)

def delete_all_history(username):
    try:
        user_id = get_user_id(username)
        if not user_id:
            return "❌ 用户未登录，无法清空", get_user_display_history(username)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM search_history WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return "✅ 全部历史记录已清空", get_user_display_history(username)
    except Exception as e:
        return f"❌ 清空失败：{str(e)}", get_user_display_history(username)

# ====================== 【第二部分：数据集解析与索引构建】 ======================
def init_environment():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.makedirs(INDEX_SAVE_DIR, exist_ok=True)

def parse_insplad_coco_dataset():
    print("\n" + "="*60)
    print("📥 开始解析COCO格式标注数据集")
    print("="*60)

    if not os.path.exists(TRAIN_JSON_PATH):
        raise FileNotFoundError(f"❌ 找不到标注JSON文件：{TRAIN_JSON_PATH}")
    if not os.path.isdir(TRAIN_IMG_FOLDER):
        raise NotADirectoryError(f"❌ 找不到图片文件夹：{TRAIN_IMG_FOLDER}")
    if not CATEGORY_ID_TO_NAME:
        raise RuntimeError("❌ 类别映射CATEGORY_ID_TO_NAME不能为空！")

    with open(TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    if not isinstance(coco_data, dict) or "images" not in coco_data or "annotations" not in coco_data:
        raise RuntimeError("❌ 不是标准COCO格式JSON！")

    image_id_to_filename = {}
    for img_item in coco_data["images"]:
        img_id = img_item.get("id")
        file_name = img_item.get("file_name", "").strip()
        if img_id is not None and file_name != "":
            image_id_to_filename[img_id] = file_name
    print(f"✅ 解析到图片总数：{len(image_id_to_filename)}")

    image_id_to_category = {}
    for ann in coco_data["annotations"]:
        img_id = ann.get("image_id")
        category_id = ann.get("category_id")
        if img_id is not None and category_id in CATEGORY_ID_TO_NAME and img_id not in image_id_to_category:
            image_id_to_category[img_id] = category_id
    print(f"✅ 解析到带有效标注的图片总数：{len(image_id_to_category)}")

    img_id2path = {}
    text_list = []
    valid_count = 0

    for img_id in sorted(image_id_to_filename.keys()):
        if img_id not in image_id_to_category:
            continue
        file_name = image_id_to_filename[img_id]
        category_id = image_id_to_category[img_id]
        device_name = CATEGORY_ID_TO_NAME[category_id]
        full_img_path = os.path.join(TRAIN_IMG_FOLDER, file_name)
        if not os.path.exists(full_img_path):
            print(f"⚠️  图片不存在，已跳过：{full_img_path}")
            continue
        text_caption = f"高压输电线路{device_name}，电力巡检设备，输电线路部件{device_name}"
        img_id2path[img_id] = full_img_path
        text_list.append(text_caption)
        valid_count += 1
        print(f"✅ 有效数据 {valid_count} | 图片ID:{img_id} | 设备类型:{device_name}")

    if valid_count == 0:
        raise RuntimeError("❌ 解析失败！未找到有效数据")
    if len(img_id2path) != len(text_list):
        raise RuntimeError(f"❌ 数据对齐失败！")

    with open(IMG_PATH_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in img_id2path.items()}, f, ensure_ascii=False)
    with open(TEXT_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(text_list))

    print("\n" + "="*60)
    print(f"🎉 数据集解析完成！有效数据总数：{valid_count}")
    print("="*60 + "\n")

    return img_id2path, text_list

def parse_chinese_clip_jsonl_dataset():
    """Load the workspace-native Chinese-CLIP JSONL pair format."""
    global CATEGORY_ID_TO_NAME
    if not os.path.isfile(TRAIN_JSON_PATH):
        raise FileNotFoundError(f"Image-text pair file not found: {TRAIN_JSON_PATH}")
    if not os.path.isdir(TRAIN_IMG_FOLDER):
        raise NotADirectoryError(f"Dataset directory not found: {TRAIN_IMG_FOLDER}")

    image_paths = {}
    captions = {}
    class_names = {}
    missing = []
    with open(TRAIN_JSON_PATH, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSONL record at line {line_number}: {error}"
                ) from error
            image_id = int(item["image_id"])
            relative_path = str(item["image"]).replace("/", os.sep)
            image_path = os.path.normpath(os.path.join(TRAIN_IMG_FOLDER, relative_path))
            caption = str(item.get("text", "")).strip()
            if not os.path.isfile(image_path):
                missing.append(image_path)
                continue
            if not caption:
                continue
            image_paths[image_id] = image_path
            captions[image_id] = caption
            if "class_id" in item and item.get("class_name"):
                class_names[int(item["class_id"])] = str(item["class_name"])

    ordered_ids = sorted(image_paths)
    if not ordered_ids:
        example = missing[0] if missing else "none"
        raise RuntimeError(
            f"No usable image-text pairs were found. First missing image: {example}"
        )
    if class_names:
        CATEGORY_ID_TO_NAME = dict(sorted(class_names.items()))
    if missing:
        print(f"Skipped {len(missing)} missing images.")
    print(f"Loaded {len(ordered_ids)} aligned image-text pairs.")

    aligned_paths = {image_id: image_paths[image_id] for image_id in ordered_ids}
    aligned_texts = [captions[image_id] for image_id in ordered_ids]
    with open(IMG_PATH_MAP_PATH, "w", encoding="utf-8") as stream:
        json.dump(
            {str(key): value for key, value in aligned_paths.items()},
            stream,
            ensure_ascii=False,
            indent=2,
        )
    with open(TEXT_LIST_PATH, "w", encoding="utf-8") as stream:
        stream.write("\n".join(aligned_texts))
    return aligned_paths, aligned_texts


def _feature_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    for attribute in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"Cannot extract a feature tensor from {type(output)}")


def encode_image_features(images):
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    try:
        features = _feature_tensor(model.get_image_features(pixel_values=pixel_values))
    except (TypeError, AttributeError):
        output = model.vision_model(pixel_values=pixel_values, return_dict=True)
        features = model.visual_projection(output.last_hidden_state[:, 0, :])
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def encode_text_features(texts):
    inputs = processor(
        text=texts,
        padding=True,
        truncation=True,
        max_length=52,
        return_tensors="pt",
    )
    text_inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    try:
        features = _feature_tensor(model.get_text_features(**text_inputs))
    except TypeError as error:
        if "must be Tensor, not NoneType" not in str(error):
            raise
        output = model.text_model(**text_inputs, return_dict=True)
        features = model.text_projection(output.last_hidden_state[:, 0, :])
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def build_image_index():
    from tqdm import tqdm
    print("🏗️  开始构建图片特征FAISS索引...")

    valid_img_ids = []
    img_features = []

    for img_id in tqdm(sorted(img_id2path.keys()), desc="提取图片特征"):
        try:
            img_path = img_id2path[img_id]
            image = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                feat = encode_image_features([image])
                feat_np = feat.cpu().numpy().flatten()
            valid_img_ids.append(img_id)
            img_features.append(feat_np)
        except Exception as e:
            print(f"⚠️  图片ID {img_id} 特征提取失败：{e}")
            continue

    img_features_np = np.vstack(img_features).astype("float32")
    img_index = faiss.IndexFlatIP(img_features_np.shape[1])
    img_index.add(img_features_np)

    faiss.write_index(img_index, IMG_INDEX_PATH)
    with open(IMG_ID_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join([str(x) for x in valid_img_ids]))

    print(f"✅ 图片索引构建完成！总特征数：{img_index.ntotal}")
    return img_index, valid_img_ids

def build_text_index():
    from tqdm import tqdm
    print("🏗️  开始构建文本特征FAISS索引...")

    text_features = []
    for text in tqdm(text_list, desc="提取文本特征"):
        try:
            with torch.no_grad():
                feat = encode_text_features([text])
                feat_np = feat.cpu().numpy().flatten()
            text_features.append(feat_np)
        except Exception as e:
            print(f"⚠️  文本特征提取失败：{text}")
            projection_dim = int(getattr(model.config, "projection_dim", 512))
            text_features.append(np.zeros(projection_dim, dtype="float32"))

    text_features_np = np.vstack(text_features).astype("float32")
    text_index = faiss.IndexFlatIP(text_features_np.shape[1])
    text_index.add(text_features_np)

    faiss.write_index(text_index, TEXT_INDEX_PATH)
    print(f"✅ 文本索引构建完成！总文本数：{text_index.ntotal}")
    return text_index

def load_existing_index():
    global img_id2path, text_list, img_index, img_id_list, text_index
    try:
        required_files = [IMG_INDEX_PATH, IMG_ID_LIST_PATH, IMG_PATH_MAP_PATH, TEXT_INDEX_PATH, TEXT_LIST_PATH]
        for file in required_files:
            if not os.path.exists(file):
                print(f"⚠️  索引文件不存在：{file}")
                return False

        with open(IMG_PATH_MAP_PATH, "r", encoding="utf-8") as f:
            img_id2path = {int(k): v for k, v in json.load(f).items()}
        with open(TEXT_LIST_PATH, "r", encoding="utf-8") as f:
            text_list = [x.strip() for x in f if x.strip()]
        with open(IMG_ID_LIST_PATH, "r", encoding="utf-8") as f:
            img_id_list = [int(x.strip()) for x in f if x.strip()]
        
        img_index = faiss.read_index(IMG_INDEX_PATH)
        text_index = faiss.read_index(TEXT_INDEX_PATH)

        print("✅ 所有索引加载成功！")
        return True
    except Exception as e:
        print(f"⚠️  索引加载失败：{e}")
        return False

def _load_global_resources_legacy(use_finetuned=True):
    global model, preprocess, tokenize, img_id2path, text_list, resources_loaded, img_index, img_id_list, text_index, loaded_use_finetuned

    if (
        resources_loaded
        and model is not None
        and img_index is not None
        and text_index is not None
        and loaded_use_finetuned == use_finetuned
    ):
        return "系统资源已就绪"

    if resources_loaded and loaded_use_finetuned is not None and loaded_use_finetuned != use_finetuned:
        print("🔄 检测到加载模式切换，正在重新初始化资源...")
        model = None
        preprocess = None
        tokenize = None
        img_index = None
        img_id_list = []
        img_id2path = {}
        text_index = None
        text_list = []
        resources_loaded = False

    init_environment()
    print("📥 正在加载系统资源...")

    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name, tokenize as clip_tokenize
    tokenize = clip_tokenize

    model, preprocess = load_from_name(
        name="ViT-B-16",
        device="cpu",
        download_root=os.path.dirname(FINETUNED_MODEL_PATH)
    )

    if use_finetuned:
        print("🔍 正在加载微调权重...")
        finetuned_ckpt = torch.load(FINETUNED_MODEL_PATH, map_location="cpu")
        state_dict = finetuned_ckpt
        if isinstance(finetuned_ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
                if key in finetuned_ckpt and isinstance(finetuned_ckpt[key], dict):
                    state_dict = finetuned_ckpt[key]
                    break
        if isinstance(state_dict, dict) and any(isinstance(v, dict) for v in state_dict.values()) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ 微调权重加载成功！")
        except RuntimeError as e:
            print(f"⚠️ 权重不完全匹配：{e}")
            model.load_state_dict(state_dict, strict=False)
    else:
        print("🔍 跳过微调权重，使用原始开源 Chinese-CLIP 权重")

    model.eval()

    index_load_success = False
    if not FORCE_REBUILD_INDEX:
        index_load_success = load_existing_index()
    
    if FORCE_REBUILD_INDEX or not index_load_success:
        print("🔄 开始重新构建全量索引...")
        img_id2path, text_list = parse_insplad_coco_dataset()
        img_index, img_id_list = build_image_index()
        text_index = build_text_index()

    resources_loaded = True
    loaded_use_finetuned = use_finetuned
    print("\n🎉 全部系统资源加载完成！")
    return "系统初始化成功"

def load_global_resources(use_finetuned=True):
    """Load the Transformers Chinese-CLIP checkpoint used by this workspace."""
    global model, processor, preprocess, tokenize
    global img_id2path, text_list, resources_loaded
    global img_index, img_id_list, text_index, loaded_use_finetuned

    if (
        resources_loaded
        and model is not None
        and processor is not None
        and img_index is not None
        and text_index is not None
        and loaded_use_finetuned == use_finetuned
    ):
        return "System resources are ready."

    init_environment()
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    model_source = (
        FINETUNED_MODEL_PATH
        if use_finetuned
        else os.getenv(
            "CHINESE_CLIP_BASE_MODEL",
            "OFA-Sys/chinese-clip-vit-base-patch16",
        )
    )
    if use_finetuned and not os.path.isdir(model_source):
        raise FileNotFoundError(
            "Fine-tuned Chinese-CLIP directory is missing: "
            f"{model_source}. Copy the complete runs/chinese_clip/best directory "
            "here, including the model, processor and tokenizer files."
        )

    print(f"Loading Chinese-CLIP from {model_source} on {DEVICE} ...")
    processor = ChineseCLIPProcessor.from_pretrained(model_source)
    model = ChineseCLIPModel.from_pretrained(model_source).to(DEVICE)
    model.eval()
    preprocess = processor
    tokenize = processor

    if not load_existing_index():
        print("Existing FAISS indexes were not found; rebuilding them now.")
        img_id2path, text_list = parse_chinese_clip_jsonl_dataset()
        img_index, img_id_list = build_image_index()
        text_index = build_text_index()

    resources_loaded = True
    loaded_use_finetuned = use_finetuned
    return f"System resources loaded on {DEVICE}."


def load_display_metadata():
    """Map each crop ID back to its original inspection image."""
    global display_metadata
    if display_metadata is not None:
        return display_metadata
    display_metadata = {}
    if not os.path.isfile(TRAIN_JSON_PATH):
        return display_metadata
    with open(TRAIN_JSON_PATH, "r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            source_image = str(item.get("source_image", "")).replace("/", os.sep)
            if not source_image:
                continue
            display_metadata[int(item["image_id"])] = {
                "source_image": os.path.normpath(
                    os.path.join(PROJECT_ROOT, source_image)
                )
            }
    return display_metadata


def get_image_by_id(image_id):
    if image_id not in img_id2path:
        return None
    try:
        metadata = load_display_metadata().get(int(image_id), {})
        full_image_path = metadata.get("source_image")
        selected_path = (
            full_image_path
            if full_image_path and os.path.isfile(full_image_path)
            else img_id2path[image_id]
        )
        with Image.open(selected_path) as source:
            image = source.convert("RGB")

        display_long_side = max(640, int(os.getenv("DISPLAY_IMAGE_LONG_SIDE", "1200")))
        width, height = image.size
        if width > 0 and height > 0 and max(width, height) > display_long_side:
            scale = display_long_side / max(width, height)
            resized = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            image = image.resize(resized, Image.Resampling.LANCZOS)
        return image
    except Exception:
        return None

def _fmt_status(level, message, start_time=None):
    tag = {"ok": "成功", "warn": "警告", "error": "错误"}.get(level, "信息")
    if start_time is not None:
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        return f"[{tag}] {message}（耗时：{elapsed_ms} 毫秒）"
    return f"[{tag}] {message}"

# ====================== 【第三部分：图文生成功能-修复超时问题】 ======================
def text_search_auto(query_text, current_user):
    t0 = datetime.now()
    if model is None or tokenize is None or img_index is None:
        return None, _fmt_status("warn", "系统资源尚未完全加载，请稍后重试。", t0), ""

    if query_text is None or not query_text.strip():
        return None, _fmt_status("warn", "请输入有效的查询文本。", t0), ""

    with torch.no_grad():
        text_feat = encode_text_features([query_text])
        text_feat_np = text_feat.cpu().numpy().astype("float32")

    top_k = 5
    scores, ids = img_index.search(text_feat_np, top_k)

    if len(ids[0]) == 0:
        return None, _fmt_status("warn", "未生成匹配结果。", t0), ""
    top1_idx = ids[0][0]
    top1_img_id = img_id_list[top1_idx]

    result_image = get_image_by_id(top1_img_id)
    if not result_image:
        return None, _fmt_status("error", "匹配图像加载失败。", t0), ""

    llm_result = llm_process_auto(top1_img_id, query_text)

    if current_user:
        save_history(current_user, "Text to Image", query_text, llm_result)

    return result_image, _fmt_status("ok", "生成完成。", t0), llm_result

def _call_doubao_text(prompt, temperature=0.2, max_tokens=1024, fallback_fn=None):
    """调用豆包接口，失败时返回兜底结果"""
    if not DOUBAO_API_KEY:
        if fallback_fn is not None:
            return fallback_fn()
        raise RuntimeError("DOUBAO_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DOUBAO_ENDPOINT_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = REQUEST_SESSION.post(
        API_BASE_URL,
        headers=headers,
        json=payload,
        timeout=(10, 180),
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

# 【修复】大模型调用函数-新增重试+长超时+兜底逻辑
def _sanitize_report_text(text):
    if not text:
        return text
    out = str(text)
    for token in ("InsPLAD", "insplad", "数据集", "样本来源", "置信度", "confidence"):
        out = out.replace(token, "")
    return out


def llm_process_auto(image_id, query_text):
    """文搜图结果解释，包含超时兜底处理"""
    prompt = f"""
    你是一名输电设备巡检领域的专业研究人员。请根据下方输入内容，对生成结果给出简洁、规范的中文解释，语言风格应符合科研论文或工程技术报告的表述要求。

    要求：
    1. 判断对应的输电设备类型；
    2. 总结该设备的常见故障类型；
    3. 给出巡检重点和现场核查建议；
    4. 只输出中文，不使用口语化表达；
    5. 不要提及任何外部数据来源、样本信息、分值数值或类似字段。

    [输入内容]：{_sanitize_report_text(query_text)}
    请严格按以下结构输出：
    [设备类型]
    [常见故障类型]
    [巡检重点与建议]
    [语言要求]
    仅使用中文。
    """
    try:
        return _call_doubao_text(prompt, temperature=0.2, max_tokens=1024, fallback_fn=lambda: get_local_fallback_analysis(query_text))
    except Exception as e:
        print(f"API请求失败：{e}，启用本地兜底分析")
        return get_local_fallback_analysis(query_text)



# 【修复】Image to Text大模型调用函数-同样修复超时问题
def image_search_auto(input_image, current_user):
    t0 = datetime.now()
    if model is None or preprocess is None or text_index is None or not text_list:
        return _fmt_status("warn", "系统资源尚未完全加载，请稍后重试。", t0), ""

    if input_image is None or (isinstance(input_image, np.ndarray) and input_image.size == 0):
        return _fmt_status("warn", "请先上传有效的设备图片。", t0), ""

    image = Image.fromarray(input_image).convert("RGB")
    with torch.no_grad():
        img_feat = encode_image_features([image])
        img_feat_np = img_feat.cpu().numpy().astype("float32")

    top_k = min(5, len(text_list))
    scores, ids = text_index.search(img_feat_np, top_k)

    if len(ids[0]) == 0:
        return _fmt_status("warn", "未生成匹配文本。", t0), "无法诊断。"

    candidates = []
    device_terms = set(CATEGORY_ID_TO_NAME.values())
    for idx, raw_score in zip(ids[0].tolist(), scores[0].tolist()):
        if idx < 0 or idx >= len(text_list):
            continue
        cand_text = text_list[idx]
        hit_bonus = 0.0
        for term in device_terms:
            if term in cand_text:
                hit_bonus = 0.015
                break
        rerank_score = float(raw_score) + hit_bonus
        candidates.append((rerank_score, cand_text, float(raw_score)))

    if not candidates:
        return _fmt_status("warn", "未生成匹配文本。", t0), "无法诊断。"

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_text = candidates[0][1]
    best_raw_score = candidates[0][2]

    llm_result = llm_process_image_auto(best_text, best_raw_score)

    if current_user:
        time_tag = datetime.now().strftime("%H:%M:%S")
        save_history(current_user, "图搜文", f"上传图片_{time_tag}", llm_result)

    return _fmt_status("ok", "生成完成。", t0), llm_result

def llm_process_image_auto(top1_text, similarity_score):
    """图搜文结果解释，包含超时兜底处理"""
    prompt = f"""
    你是一名输电设备巡检领域的专业研究人员。请根据下方匹配文本，对生成结果给出简洁、规范的中文解释，语言风格应符合科研论文或工程技术报告的表述要求。

    要求：
    1. 判断匹配文本对应的输电设备类型；
    2. 说明生成内容与匹配文本之间的关系；
    3. 列出2到4个关键巡检关注点；
    4. 给出可执行的巡检和维护建议；
    5. 只输出中文，不使用口语化表达；
    6. 不要提及任何数据集名称、样本来源、置信度数值或类似字段。

    [匹配文本]：{_sanitize_report_text(top1_text)}
    请严格按以下结构输出：
    [设备类型]
    [匹配说明]
    [关键巡检关注点]
    [建议]
    [语言要求]
    仅使用中文。
    """
    try:
        return _call_doubao_text(prompt, temperature=0.1, max_tokens=1024, fallback_fn=lambda: get_local_fallback_diagnosis(top1_text, similarity_score))
    except Exception as e:
        print(f"API请求失败：{e}，启用本地兜底诊断")
        return get_local_fallback_diagnosis(top1_text, similarity_score)



# ====================== 【新增：API失败本地兜底逻辑】 ======================
def get_local_fallback_analysis(query_text):
    """文搜图 API 失败时的本地兜底分析"""
    device_type = "未知输电设备"
    for cat_name in CATEGORY_ID_TO_NAME.values():
        if cat_name in query_text:
            device_type = cat_name
            break

    fault_map = {
        "盘形悬式绝缘子": "1. 瓷体裂纹、破损或缺口；2. 污秽闪络；3. 金具腐蚀或断裂；4. 零值或低值绝缘子失效",
        "防振锤": "1. 防振锤滑移或脱落；2. 锤头腐蚀或断裂；3. 钢股断裂或锈蚀；4. 安装位置偏移",
        "间隔棒": "1. 夹具松动或脱落；2. 橡胶垫老化或损坏；3. 框架腐蚀或断裂；4. 阻尼组件失效",
        "输电杆塔": "1. 塔材腐蚀、弯曲或变形；2. 螺栓松动或缺失；3. 基础沉降或开裂；4. 杆塔倾斜",
        "避雷器": "1. 漏电流过大；2. 瓷套开裂或污秽；3. 接地引线断裂或腐蚀；4. 计数器失效"
    }

    check_map = {
        "盘形悬式绝缘子": "1. 检查瓷体是否存在破损、裂纹或污秽；2. 使用绝缘电阻表测试零值绝缘子；3. 检查金具是否牢固且无腐蚀；4. 复核绝缘子串倾角是否正常。",
        "防振锤": "1. 检查防振锤安装位置是否符合设计要求；2. 检查锤头是否变形、腐蚀或脱落；3. 检查钢丝股是否断裂或松散；4. 检查夹具螺栓是否紧固。",
        "间隔棒": "1. 检查夹具与导线的连接是否牢固；2. 检查框架是否变形、开裂或腐蚀；3. 检查阻尼组件是否完好且无老化；4. 检查安装间距是否符合要求。",
        "输电杆塔": "1. 检查塔身构件是否弯曲、变形或腐蚀；2. 检查螺栓是否松动、缺失或腐蚀；3. 检查基础是否沉降、开裂或积水；4. 检查杆塔倾斜与挠度是否超限。",
        "避雷器": "1. 检查本体是否存在损伤、污秽或泄漏；2. 检查接地引线是否牢固且无腐蚀；3. 检查动作计数器是否正常；4. 定期测量漏电流和绝缘电阻。"
    }

    faults = fault_map.get(device_type, "1. Exterior damage or deformation; 2. corrosion or fracture of metal parts; 3. loose or detached connections; 4. degraded insulation performance")
    checks = check_map.get(device_type, "1. Check for damage, deformation, and corrosion on the exterior; 2. Verify all connection points are secure; 3. Check for aging or damaged insulating components; 4. Confirm installation position and parameters meet design requirements.")

    return f"""
[设备类型]
{device_type}
[常见故障类型]
{faults}
[巡检重点与说明]
{checks}
[说明]
由于 API 请求超时，当前返回本地兜底分析结果。请检查网络后重试，以获得更准确的结果。
"""

def get_local_fallback_diagnosis(top1_text, similarity_score):
    """图搜文 API 失败时的本地兜底诊断"""
    device_type = "未知输电设备"
    for cat_name in CATEGORY_ID_TO_NAME.values():
        if cat_name in top1_text:
            device_type = cat_name
            break

    return f"""
[设备类型]
{device_type}
[匹配说明]
生成内容与匹配文本高度相关，可用于输电设备巡检场景。
[关键巡检关注点]
1. 检查设备本体是否存在损伤、变形、污秽和腐蚀；
2. 检查连接部位和紧固件是否松动；
3. 检查绝缘与导电部件是否存在放电痕迹和老化迹象；
4. 检查安装位置和姿态是否存在偏移、倾斜或下垂。
[建议]
1. 对关键连接点进行现场复核和紧固；
2. 对疑似老化区域开展专项巡检（红外/绝缘检测）；
3. 建立图像对比基线，便于后续周期性巡检；
4. 若发现异常趋势，及时安排计划检修。
[说明]
由于 API 请求超时，当前返回本地兜底诊断结果。请检查网络后重试，以获得更准确的结果。
"""

# ====================== 【第四部分：统计看板功能】 ======================
def get_statistics_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM search_history")
    total_searches = cursor.fetchone()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM search_history WHERE create_time LIKE ?", (f"{today}%",))
    today_searches = cursor.fetchone()[0]
    total_images = len(img_id2path) if 'img_id2path' in globals() else 0
    
    cursor.execute("SELECT search_type, COUNT(*) FROM search_history GROUP BY search_type")
    search_type_data = cursor.fetchall()
    search_type_dict = {t: c for t, c in search_type_data}
    
    seven_days_ago = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT DATE(create_time) as dt, COUNT(*) 
        FROM search_history 
        WHERE DATE(create_time) >= ? 
        GROUP BY DATE(create_time)
        ORDER BY dt
    """, (seven_days_ago,))
    trend_data = cursor.fetchall()
    date_range = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    trend_dict = {d: 0 for d in date_range}
    for d, c in trend_data:
        trend_dict[d] = c
    
    cursor.execute("SELECT query_content FROM search_history WHERE query_content IS NOT NULL AND query_content != ''")
    all_queries = [q[0] for q in cursor.fetchall()]

    # 词云文本清洗：去掉低价值通用词，避免“上传图片”等词占据视觉
    stop_words = {
        "上传图片", "上传", "图片", "上传设备图片", "开始生成", "开始分析", "上传待检测图片",
        "上传图片_", "query", "none", "null", "txt", "jpg", "png"
    }
    cleaned_queries = []
    for q in all_queries:
        q = str(q)
        for w in stop_words:
            q = q.replace(w, " ")
        cleaned_queries.append(q)

    text_for_wordcloud = " ".join(cleaned_queries) if cleaned_queries else "暂无数据"
    
    category_dist = {}
    if 'img_id2path' in globals() and 'text_list' in globals():
        for text in text_list:
            for cat_name in CATEGORY_ID_TO_NAME.values():
                if cat_name in text:
                    category_dist[cat_name] = category_dist.get(cat_name, 0) + 1
                    break
    if not category_dist:
        category_dist = {v: np.random.randint(50, 200) for v in CATEGORY_ID_TO_NAME.values()}
    
    conn.close()
    return {
        "core_metrics": (total_users, total_searches, today_searches, total_images),
        "search_type": search_type_dict,
        "trend": trend_dict,
        "wordcloud_text": text_for_wordcloud,
        "category_dist": category_dist
    }

def create_core_metrics_cards(metrics):
    total_users, total_searches, today_searches, total_images = metrics

    fig = go.Figure()

    card_specs = [
        ("总用户数", total_users, "#2563eb", "用户"),
        ("总生成次数", total_searches, "#0f766e", "生成次数"),
        ("今日生成", today_searches, "#d97706", "今日"),
        ("数据集图片", total_images, "#7c3aed", "图片"),
    ]

    for i, (title, value, color, suffix) in enumerate(card_specs):
        x0 = i * 0.25
        x1 = x0 + 0.25
        fig.add_shape(
            type="rect",
            x0=x0 + 0.01, y0=0.08, x1=x1 - 0.01, y1=0.92,
            line=dict(color="rgba(219,228,240,0.95)", width=1),
            fillcolor="rgba(255,255,255,0.95)",
            layer="below"
        )
        fig.add_annotation(
            x=x0 + 0.125,
            y=0.72,
            text=f"<span style='font-size: 13px; color: #64748b; letter-spacing: .6px;'>{suffix}</span><br><b>{title}</b>",
            showarrow=False,
            font=dict(size=18, color="#334155"),
            align="center",
            xanchor="center",
            yanchor="middle"
        )
        fig.add_annotation(
            x=x0 + 0.125,
            y=0.46,
            text=f"<span style='font-size: 34px; color: {color}; font-weight: 800;'>{value}</span>",
            showarrow=False,
            align="center",
            xanchor="center",
            yanchor="middle"
        )

    fig.update_layout(
        height=190,
        margin=dict(l=6, r=6, t=6, b=6),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def create_search_type_pie(search_type_dict):
    labels = list(search_type_dict.keys()) if search_type_dict else ["文搜图", "图搜文"]
    values = list(search_type_dict.values()) if search_type_dict else [50, 50]
    fig = px.pie(
        names=labels,
        values=values,
        title="任务类型分布",
        color_discrete_sequence=["#2563eb", "#0f766e", "#d97706", "#7c3aed"],
        hole=0.58
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='white', width=2)))
    fig.update_layout(
        height=400,
        title_x=0.5,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    return fig

def create_trend_line_chart(trend_dict):
    dates = list(trend_dict.keys())
    counts = list(trend_dict.values())
    fig = go.Figure(go.Scatter(
        x=dates,
        y=counts,
        mode='lines+markers',
        name='生成次数',
        line=dict(color="#2563eb", width=3),
        marker=dict(size=9, color="#2563eb", line=dict(color='white', width=1.5))
    ))
    fig.update_layout(
        title="近7天生成趋势",
        xaxis_title="日期",
        yaxis_title="生成次数",
        height=400,
        title_x=0.5,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fbff"
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False)
    return fig

def create_category_bar_chart(category_dist):
    categories = list(category_dist.keys())
    counts = list(category_dist.values())
    sorted_pairs = sorted(zip(counts, categories), reverse=True)
    counts, categories = zip(*sorted_pairs) if sorted_pairs else ([], [])
    fig = go.Figure(go.Bar(
        x=categories,
        y=counts,
        marker_color="#2563eb",
        text=counts,
        textposition='outside',
        hovertemplate="%{x}<br>数量：%{y}<extra></extra>"
    ))
    fig.update_layout(
        title="输电设备类别分布",
        xaxis_title="设备类型",
        yaxis_title="图片数量",
        height=450,
        title_x=0.5,
        xaxis_tickangle=-40,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fbff",
        margin=dict(t=60, l=20, r=20, b=90)
    )
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False)
    return fig

def create_wordcloud_figure(text):
    try:
        font_path = "C:/Windows/Fonts/simhei.ttf"
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/msyh.ttc"
        
        wc = WordCloud(
            font_path=font_path,
            background_color='white',
            width=900,
            height=420,
            max_words=100,
            colormap='viridis'
        ).generate(text)
        fig = go.Figure(go.Image(z=wc))
        fig.update_layout(
            title="高频生成词云",
            height=450,
            title_x=0.5,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=50, b=0)
        )
        return fig
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="词云生成失败", x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(height=450, title="热门生成词云", title_x=0.5)
        return fig

# ====================== 【第五部分：批量处理功能】 ======================
def batch_process_texts(text_input, file_input):
    t0 = datetime.now()
    if model is None or tokenize is None or img_index is None:
        return [], _fmt_status("warn", "系统资源未完成加载，请稍后重试", t0)

    input_lines = []
    
    if file_input is not None:
        try:
            with open(file_input.name, 'r', encoding='utf-8') as f:
                file_content = f.read()
            input_lines.extend(file_content.splitlines())
        except Exception as e:
            return [], _fmt_status("error", f"文件读取失败：{str(e)}", t0)
    
    if text_input is not None and text_input.strip() != "":
        input_lines.extend(text_input.splitlines())
    
    clean_queries = []
    seen = set()
    for line in input_lines:
        line = line.strip()
        if line != "" and line not in seen:
            clean_queries.append(line)
            seen.add(line)
    
    if not clean_queries:
        return [], _fmt_status("warn", "未输入有效的设备描述，请重新输入。", t0)
    
    results = []
    for idx, query in enumerate(clean_queries):
        print(f"\n🔄 批量处理进度：{idx+1}/{len(clean_queries)} | 查询：{query}")
        
        try:
            with torch.no_grad():
                text_feat = encode_text_features([query])
                text_feat_np = text_feat.cpu().numpy().astype("float32")
            
            top_k = 1
            scores, ids = img_index.search(text_feat_np, top_k)
            
            if len(ids[0]) > 0:
                top1_idx = ids[0][0]
                top1_img_id = img_id_list[top1_idx]
                top1_score = scores[0][0]
                
                result_image = get_image_by_id(top1_img_id)
                llm_result = llm_process_auto(top1_img_id, query)
                
                results.append({
                    "query": query,
                    "image": result_image,
                    "analysis": llm_result,
                    "score": top1_score,
                    "success": True
                })
            else:
                results.append({
                    "query": query,
                    "image": None,
                    "analysis": "未生成到匹配结果",
                    "score": 0,
                    "success": False
                })
        except Exception as e:
            print(f"⚠️  处理失败：{query}，错误：{e}")
            results.append({
                "query": query,
                "image": None,
                "analysis": f"❌ 处理失败：{str(e)}",
                "score": 0,
                "success": False
            })
    
    return results, _fmt_status("ok", f"批量处理完成，共处理 {len(results)} 条设备描述。", t0)

# ====================== 【第六部分：输电设备缺陷检测功能（RT-DETR）】 ======================
DEFECT_CLASS_ZH_MAP = {
    "Broken": "断裂",
    "broken": "断裂",
    "Crack": "裂纹",
    "crack": "裂纹",
    "Insulator": "绝缘子",
    "insulator": "绝缘子",
    "Broken_Insulator": "绝缘子破损",
    "Broken_Insulator_Cap": "绝缘子瓷帽破损",
    "Insulator_Cap": "绝缘子瓷帽",
    "Damaged_Cable_Jackets": "电缆护套破损",
    "Frayed_Cable": "导线磨损",
    "Power_Cable": "电力导线",
    "Damper": "防振锤",
    "Spacer": "间隔棒",
    "Tower": "杆塔",
    "bird nest": "鸟巢异物",
    "Bird Nest": "鸟巢异物",
    "vegetation": "树障异物",
    "trees": "树障异物",
    "corrosion": "腐蚀"
}

def to_zh_defect_name(name):
    name = str(name).strip()
    if name in DEFECT_CLASS_ZH_MAP:
        return DEFECT_CLASS_ZH_MAP[name]
    return name.replace("_", " ")

def load_defect_detector():
    global defect_detector
    if defect_detector is not None:
        return defect_detector
    if YOLO is None:
        return None
    if not os.path.exists(DEFECT_MODEL_PATH):
        print(f"缺陷检测模型不存在：{DEFECT_MODEL_PATH}")
        return None
    try:
        defect_detector = YOLO(DEFECT_MODEL_PATH)
        print("缺陷检测模型加载成功")
    except Exception as e:
        print(f"缺陷检测模型加载失败：{e}")
        defect_detector = None
    return defect_detector

def detect_defect_on_image(input_image, current_user=""):
    t0 = datetime.now()
    if input_image is None or (isinstance(input_image, np.ndarray) and input_image.size == 0):
        return None, _fmt_status("warn", "请先上传图片。", t0), ""

    detector = load_defect_detector()
    if detector is None:
        return input_image, _fmt_status("warn", "缺陷检测模型尚未就绪。", t0), f"请先训练并将权重文件放置到：{DEFECT_MODEL_PATH}"

    try:
        conf_value = DEFECT_CONF_THRES
        results = detector.predict(source=input_image, conf=conf_value, verbose=False)
        res = results[0]

        plotted = res.plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        lines = []
        stat_map = {}
        if res.boxes is not None and len(res.boxes) > 0:
            cls_ids = res.boxes.cls.cpu().numpy().astype(int).tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()
            for cid, score in zip(cls_ids, confs):
                raw_name = res.names.get(cid, str(cid)) if isinstance(res.names, dict) else str(cid)
                zh_name = to_zh_defect_name(raw_name)
                lines.append(f"- {zh_name}（{raw_name}）")

                key = f"{zh_name}（{raw_name}）"
                if key not in stat_map:
                    stat_map[key] = {"count": 0, "max_conf": 0.0}
                stat_map[key]["count"] += 1
                stat_map[key]["max_conf"] = max(stat_map[key]["max_conf"], float(score))

        if not lines:
            raw_summary = "未检测到明显缺陷目标"
            status = _fmt_status("ok", "检测完成，未发现明显缺陷目标。", t0)
        else:
            detail_lines = []
            for k, v in sorted(stat_map.items(), key=lambda x: x[1]["max_conf"], reverse=True):
                detail_lines.append(f"- {k}：数量{v['count']}")
            raw_summary = "\n".join(["检测明细："] + lines + ["", "汇总："] + detail_lines)
            status = _fmt_status("ok", f"检测完成，识别到 {len(lines)} 个目标", t0)

        summary = llm_process_defect_detection(raw_summary)

        if current_user:
            save_history(current_user, "缺陷检测", "上传图片", summary[:300])

        return plotted_rgb, status, summary
    except Exception as e:
        return input_image, _fmt_status("error", "Detection failed", t0), str(e)

def llm_process_defect_detection(raw_detection_text):
    """调用豆包API，把检测结果整理成中文巡检结论"""
    if not raw_detection_text or not raw_detection_text.strip():
        return "未检测到缺陷目标。"

    equipment_names = {
        "绝缘子", "防振锤", "间隔棒", "杆塔", "电力导线", "导线", "设备", "tower", "insulator", "damper", "spacer"
    }
    is_pure_equipment = True
    for line in raw_detection_text.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        cls_part = line.split(":", 1)[0].replace("-", "").strip().lower()
        cls_part = cls_part.split("（", 1)[0].strip()
        if cls_part not in {x.lower() for x in equipment_names}:
            is_pure_equipment = False
            break

    if is_pure_equipment:
        return """[检测目标]
识别对象为输电设备本体部件（如绝缘子、防振锤、间隔棒、杆塔和导线），未检测到明确缺陷类别。
[检测结论]
未检测到明显缺陷目标。
[关键观察]
1. 画面主要由设备本体目标构成；
2. 未检测到断裂、裂纹、腐蚀或异物等明显缺陷标签；
3. 建议结合历史图像持续对比观察。
[建议]
1. 继续开展周期性巡检，并保持基线对比；
2. 复核关键连接部位与紧固点；
3. 如出现发热、异响或放电迹象，建议开展红外或带电检测。"""

    prompt = f"""
你是一名输电设备巡检专家。请将下方检测结果整理为详细、专业的中文结论，结构如下：
1. [检测目标]（说明检测到的设备/部件）
2. [检测结论]（是否存在缺陷）
3. [主要缺陷项]（按风险从高到低列出名称和数量，不要输出置信度）
4. [关键观察]（结合统计结果和细节至少给出3点）
5. [建议]（至少给出4条可执行建议）

重要规则：
- 如果检测到的只是输电设备名称（如绝缘子、防振锤、间隔棒、杆塔、导线），而不是缺陷类别，则必须判断为“未发现明显缺陷”。
- 只输出面向巡检人员的结论，不要出现数据集名称，不要出现置信度数值。

原始检测结果：
{raw_detection_text}
"""
    try:
        return _call_doubao_text(prompt, temperature=0.2, max_tokens=700, fallback_fn=lambda: f"原始检测结果：\n{raw_detection_text}\n\n中文摘要生成失败：接口调用异常")
    except Exception as e:
        return f"原始检测结果：\n{raw_detection_text}\n\n中文摘要生成失败：{str(e)}"


# 已恢复为非流式输出，保留兼容占位

# ====================== 【第七部分：输电设备AI问答助手功能】 ======================
def power_equipment_qa(question):
    """输电设备AI问答核心函数"""
    if not question or not question.strip():
        return "请输入有效的问题"
    if not DOUBAO_API_KEY:
        return "未配置 DOUBAO_API_KEY，AI问答功能暂不可用；检索和检测功能不受影响。"

    system_prompt = """你是专业的输电设备运维专家，回答需包含原因分析和处理建议。
    输出格式严格按照以下：
    1. 原因分析
    2. 可能风险
    3. 处理建议
    """

    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DOUBAO_ENDPOINT_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.3,
        "max_tokens": 1500
    }

    try:
        response = REQUEST_SESSION.post(
            API_BASE_URL,
            headers=headers,
            json=payload,
            timeout=(10, 180)
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ AI问答请求失败：{str(e)}\n请检查网络连接或API配置后重试。"

# ====================== 【第七部分：Gradio完整界面】 ======================
init_db()

custom_css = """
:root {
  --bg: #eef3fb;
  --card: rgba(255, 255, 255, 0.92);
  --card-solid: #ffffff;
  --card-soft: #f8fbff;
  --text: #0f172a;
  --subtext: #475569;
  --border: #dbe4f0;
  --primary: #2563eb;
  --primary-dark: #1d4ed8;
  --success: #0f766e;
  --shadow: 0 18px 44px rgba(15, 23, 42, .08);
}

html, body {
  margin: 0 !important;
  padding: 0 !important;
  width: 100% !important;
  overflow-x: hidden !important;
  background: linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%) !important;
}

*, *::before, *::after {
  box-sizing: border-box !important;
}

.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
}

h1, h2, h3 {
  color: var(--text) !important;
  letter-spacing: .2px;
  margin-top: 4px !important;
  margin-bottom: 8px !important;
}

p, label, .prose, .markdown {
  color: var(--subtext) !important;
}

/* 清理默认边框，改为卡片化视觉 */
.gr-block, .gr-box, .gr-panel, .gr-form, .gr-group {
  border: none !important;
  border-radius: 18px !important;
  box-shadow: none !important;
  background: transparent !important;
}

*:focus, *:focus-visible {
  outline: none !important;
  box-shadow: none !important;
}

/* 顶部标题横幅 */
.hero-banner {
  max-width: 1260px;
  margin: 20px auto 14px;
  padding: 22px 26px;
  border: 1px solid rgba(219, 228, 240, .9);
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(37, 99, 235, .08), rgba(255, 255, 255, .9) 40%, rgba(15, 118, 110, .06));
  box-shadow: var(--shadow);
}
.hero-banner h1 {
  margin: 0 0 6px !important;
  font-size: clamp(28px, 3vw, 40px) !important;
}
.hero-banner p {
  margin: 0 !important;
  font-size: 15px;
  line-height: 1.7;
}
.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 14px;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,.9);
  color: var(--text);
  font-size: 13px;
  font-weight: 600;
}

/* 登录卡片 */
.auth-card {
  max-width: 980px;
  margin: 16px auto 0;
  padding: 22px 24px;
  border: 1px solid var(--border);
  border-radius: 22px;
  background: var(--card);
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
}
.auth-card textarea, .auth-card input {
  min-height: 44px !important;
  font-size: 15px !important;
  border: 1px solid #dbe3ef !important;
  background: #fcfdff !important;
}
.auth-card button { min-height: 44px !important; font-size: 15px !important; }

/* 主界面区域 */
.main-area {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}

div.main-area,
div.main-area > div,
div.main-area .gr-form,
div.main-area .gr-group,
div.main-area .gr-box,
div.main-area .gr-panel {
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
}

/* 内容卡片 */
.content-shell {
  max-width: 1260px;
  margin: 0 auto 18px;
  padding: 0 18px;
}
.content-card {
  margin-top: 14px;
  padding: 18px 18px 8px;
  border: 1px solid var(--border);
  border-radius: 22px;
  background: var(--card);
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
}
.tab-shell {
  padding: 8px 2px 4px;
}
.tab-shell h3 {
  margin-top: 0 !important;
  margin-bottom: 6px !important;
  font-size: 18px !important;
}
.tab-shell p {
  margin-top: 0 !important;
  margin-bottom: 12px !important;
  color: var(--subtext) !important;
}

/* 顶部用户栏 */
.user-topbar {
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px 14px;
  margin-bottom: 12px;
  box-shadow: 0 6px 16px rgba(15, 23, 42, .04);
}
.user-topbar p { margin: 0 !important; color: var(--subtext) !important; }

.gr-row {
  flex-wrap: wrap !important;
  row-gap: 10px !important;
}

.gr-slider {
  min-width: 320px !important;
}
button {
  min-width: 120px;
  border-radius: 12px !important;
  transition: transform .15s ease, box-shadow .15s ease, opacity .15s ease;
}
button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 20px rgba(15, 23, 42, .08);
}
button:active {
  transform: translateY(0);
}

button[role="tab"] {
  border-radius: 12px !important;
  margin-right: 8px !important;
  padding: 10px 16px !important;
  font-weight: 700 !important;
  border: 1px solid #d6deea !important;
  background: rgba(255,255,255,.92) !important;
  color: #334155 !important;
  min-width: auto !important;
}
button[role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, #eaf1ff, #ffffff) !important;
  color: #1e3a8a !important;
  border-color: #bcd0ff !important;
}

textarea, input {
  border-radius: 12px !important;
  border: 1px solid #dbe3ef !important;
  background: #ffffff !important;
}
textarea:focus, input:focus {
  border-color: #9bb7ff !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, .12) !important;
}

button.primary {
  background: linear-gradient(135deg, var(--primary), #4f8cff) !important;
  border: 1px solid var(--primary) !important;
  color: #fff !important;
  box-shadow: 0 10px 22px rgba(37, 99, 235, .24) !important;
}
button.primary:hover {
  background: linear-gradient(135deg, var(--primary-dark), #3f7cf7) !important;
  border-color: var(--primary-dark) !important;
}
button.secondary {
  background: #ffffff !important;
  border: 1px solid #d4dbe7 !important;
  color: #334155 !important;
}

.gr-markdown, .gr-dataframe, .gr-plot, .gr-image, .gr-textbox {
  border-radius: 14px !important;
}

footer { display: none !important; }
"""

with gr.Blocks(title="输电设备图文生成系统") as demo:
    gr.HTML("""
    <div class="hero-banner">
      <div style="display:flex; gap:18px; align-items:center; flex-wrap:wrap;">
        <div style="width:72px; height:72px; border-radius:20px; background:linear-gradient(135deg, #2563eb, #0f766e); display:flex; align-items:center; justify-content:center; box-shadow:0 14px 30px rgba(37, 99, 235, .22); flex:0 0 auto;">
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <circle cx="20" cy="20" r="18" stroke="rgba(255,255,255,.26)" stroke-width="2"/>
            <path d="M22 5L11 22H19L18 35L29 18H21L22 5Z" fill="white" opacity="0.98"/>
            <path d="M9 29C12.5 32.5 17.5 34.5 22.8 34" stroke="rgba(255,255,255,.85)" stroke-width="2.2" stroke-linecap="round"/>
            <path d="M28.5 8.5C31.2 10.2 33.3 12.8 34.3 16" stroke="rgba(255,255,255,.85)" stroke-width="2.2" stroke-linecap="round"/>
          </svg>
        </div>
        <div style="min-width:240px; flex:1 1 320px;">
          <h1>输电设备图文生成系统</h1>
          <p>面向输电线路巡检与设备理解，支持文搜图检索、图像描述、缺陷检测、批量处理、智能问答和历史分析。</p>
        </div>
      </div>
      <div class="hero-badges">
        <span class="hero-badge">文搜图</span>
        <span class="hero-badge">图像描述</span>
        <span class="hero-badge">缺陷检测</span>
        <span class="hero-badge">批量处理</span>
        <span class="hero-badge">问答</span>
        <span class="hero-badge">历史分析</span>
      </div>
    </div>
    """)

    is_logged_in = gr.State(False)
    current_user = gr.State("")
    system_init_status = gr.State("")
    system_status_text = gr.Markdown("正在初始化系统资源，请稍候...")
    gr.HTML("""
    <div class="content-shell">
      <div class="content-card" style="margin-top:0;">
        <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; justify-content:space-between;">
          <div>
            <div style="font-size:13px; color:#64748b; font-weight:700; letter-spacing:.6px; text-transform:uppercase;">系统状态</div>
            <div style="font-size:18px; color:#0f172a; font-weight:800; margin-top:4px;">输电设备图文工作台</div>
          </div>
          <div style="display:flex; gap:10px; flex-wrap:wrap;">
            <span class="hero-badge">图文生成</span>
            <span class="hero-badge">缺陷识别</span>
            <span class="hero-badge">结果记录</span>
          </div>
        </div>
        <div style="margin-top:12px; padding:12px 14px; border-radius:14px; background:rgba(37, 99, 235, .06); border:1px solid rgba(37, 99, 235, .14); color:#1e3a8a; font-weight:600;">
          系统状态：<span style="color:#0f172a;">已就绪，登录后可使用全部功能。</span>
        </div>
      </div>
    </div>
    """)

    def safe_load_resources():
        try:
            msg = load_global_resources()
            return msg, f"✅ {msg or '系统资源加载成功'}"
        except Exception as e:
            return f"❌ 系统初始化失败：{str(e)}", f"❌ 系统初始化失败：{str(e)}"

    demo.load(fn=safe_load_resources, outputs=[system_init_status, system_status_text])

    # 登录 / 注册
    with gr.Column(visible=True, elem_classes=["auth-card"]) as auth_panel:
        gr.Markdown("## 用户登录与注册")
        with gr.Tabs():
            with gr.Tab("登录"):
                login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                login_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                login_btn = gr.Button("登录", variant="primary", size="lg")
                login_status = gr.Textbox(label="登录状态", interactive=False)
            with gr.Tab("注册"):
                reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名（至少3个字符）")
                reg_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码（至少6个字符）")
                reg_confirm_password = gr.Textbox(label="确认密码", type="password", placeholder="请再次输入密码")
                reg_btn = gr.Button("注册", variant="secondary", size="lg")
                reg_status = gr.Textbox(label="注册状态", interactive=False)

    # 登录后主界面
    with gr.Column(visible=False, elem_classes=["main-area"]) as main_panel:
        with gr.Column(elem_classes=["content-shell"]):
            gr.Markdown("## 输电设备图文生成与分析")
            with gr.Row(elem_classes=["user-topbar"]):
                user_info = gr.Markdown("当前用户：-")
                logout_btn = gr.Button("退出登录", variant="secondary", size="sm")

            with gr.Column(elem_classes=["content-card"]):
                with gr.Tabs():
                    # 标签页1：文搜图
                    with gr.Tab("文搜图"):
                        with gr.Column(elem_classes=["tab-shell"]):
                            gr.Markdown("### 文搜图")
                            gr.Markdown("输入设备描述后，系统将生成相似图像结果并给出专业解释。")
                            query_text = gr.Textbox(label="设备描述", placeholder="请输入输电设备描述", lines=3)
                            with gr.Row():
                                search_btn = gr.Button("生成", variant="primary", size="lg")
                                clear_text_btn = gr.Button("清空", variant="secondary", size="lg")
                                text_progress = gr.Markdown("")
                            with gr.Row():
                                result_image = gr.Image(label="生成图像", type="numpy", height=450, sources=["upload"], placeholder="拖入图片或点击上传")
                            status_text = gr.Textbox(label="状态", interactive=False, show_label=False)
                            llm_result = gr.Textbox(label="解释结果", lines=10, interactive=False)
                            search_btn.click(show_progress="hidden", fn=lambda: "处理中...", outputs=[text_progress]).then(
                                fn=text_search_auto,
                                inputs=[query_text, current_user],
                                outputs=[result_image, status_text, llm_result]
                            ).then(fn=lambda: "", outputs=[text_progress])
                            clear_text_btn.click(fn=lambda: ("", None, "", "", ""), outputs=[query_text, result_image, status_text, llm_result, text_progress])

                    # 标签页2：图搜文
                    with gr.Tab("图搜文"):
                        with gr.Column(elem_classes=["tab-shell"]):
                            gr.Markdown("### 图搜文")
                            gr.Markdown("上传设备图片后，系统将生成对应文本描述并给出专业解释。")
                            with gr.Row():
                                input_image = gr.Image(label="上传设备图片", type="numpy", height=400, sources=["upload"], placeholder="拖入图片或点击上传")
                            with gr.Row():
                                image_search_btn = gr.Button("生成", variant="primary", size="lg")
                                clear_image_btn = gr.Button("清空", variant="secondary", size="lg")
                                image_progress = gr.Markdown("")
                            image_status = gr.Textbox(label="状态", interactive=False, show_label=False)
                            image_llm_result = gr.Textbox(label="描述结果", lines=10, interactive=False)
                            image_search_btn.click(fn=lambda: "处理中...", outputs=[image_progress]).then(
                                fn=image_search_auto,
                                inputs=[input_image, current_user],
                                outputs=[image_status, image_llm_result]
                            ).then(fn=lambda: "", outputs=[image_progress])
                            clear_image_btn.click(fn=lambda: (None, "", "", ""), outputs=[input_image, image_status, image_llm_result, image_progress])

            # 标签页3：缺陷检测
            with gr.Tab("缺陷检测"):
                with gr.Column(elem_classes=["tab-shell"]):
                    gr.Markdown("### 缺陷检测")
                    gr.Markdown("左右对比原始图像与检测结果，便于快速查看异常。")
                    with gr.Row():
                        defect_input = gr.Image(label="上传检测图片", type="numpy", height=420, sources=["upload"], placeholder="拖入图片或点击上传")
                        defect_output = gr.Image(label="检测结果", type="numpy", height=420)
                    with gr.Row():
                        defect_btn = gr.Button("开始检测", variant="primary", size="lg")
                        defect_clear_btn = gr.Button("清空", variant="secondary", size="lg")
                        defect_progress = gr.Markdown("")
                    defect_status = gr.Textbox(label="检测状态", interactive=False, show_label=False)
                    defect_summary = gr.Textbox(label="检测摘要", lines=8, interactive=False)
                    defect_btn.click(fn=lambda: "处理中...", outputs=[defect_progress]).then(
                        fn=detect_defect_on_image,
                        inputs=[defect_input, current_user],
                        outputs=[defect_output, defect_status, defect_summary]
                    ).then(fn=lambda: "", outputs=[defect_progress])
                    defect_clear_btn.click(
                        fn=lambda: (None, None, "", "", ""),
                        outputs=[defect_input, defect_output, defect_status, defect_summary, defect_progress]
                    )

            # 标签页4：批量处理
            with gr.Tab("批量处理"):
                with gr.Column(elem_classes=["tab-shell"]):
                    gr.Markdown("### 批量处理")
                    gr.Markdown("支持多行文本或 TXT 文件，一次处理多个设备描述。")
                    with gr.Row():
                        with gr.Column(scale=2):
                            batch_text_input = gr.Textbox(
                                label="粘贴设备描述（每行一条）",
                                placeholder="请输入描述，每行一条，例如：\n盘形悬式绝缘子\n防振锤\n间隔棒\n输电杆塔",
                                lines=8,
                                max_lines=15
                            )
                            batch_file_input = gr.File(
                                label="或上传 TXT 文件",
                                file_types=[".txt"],
                                file_count="single"
                            )
                            with gr.Row():
                                batch_start_btn = gr.Button("开始批量处理", variant="primary", size="lg")
                                batch_clear_btn = gr.Button("清空", variant="secondary", size="lg")
                                batch_progress = gr.Markdown("")
                            batch_status = gr.Textbox(label="处理状态", interactive=False, show_label=False)
                    
                    gr.Markdown("---")

                    batch_results_state = gr.State([])
                    
                    with gr.Column(visible=False) as batch_results_panel:
                        @gr.render(inputs=batch_results_state)
                        def render_batch_results(results):
                            if not results:
                                return
                            for idx, res in enumerate(results):
                                with gr.Accordion(f"结果 {idx+1}：{res['query']}", open=idx==0):
                                    with gr.Row():
                                        if res['success'] and res['image'] is not None:
                                            gr.Image(value=res['image'], label="匹配图像", height=300, width=400)
                                        else:
                                            gr.Markdown("❌ 未找到匹配图像")
                                        gr.Markdown(f"### 分析结果\n{res['analysis']}")
                    
                    def update_batch_results(text_input, file_input):
                        results, status_msg = batch_process_texts(text_input, file_input)
                        return results, status_msg, gr.update(visible=bool(results))
                    
                    batch_start_btn.click(fn=lambda: "处理中...", outputs=[batch_progress]).then(
                        fn=update_batch_results,
                        inputs=[batch_text_input, batch_file_input],
                        outputs=[batch_results_state, batch_status, batch_results_panel]
                    ).then(fn=lambda: "", outputs=[batch_progress])
                    batch_clear_btn.click(
                        fn=lambda: ("", None, [], "", gr.update(visible=False), ""),
                        outputs=[batch_text_input, batch_file_input, batch_results_state, batch_status, batch_results_panel, batch_progress]
                    )

            # 标签页5：智能问答
            with gr.Tab("智能问答"):
                with gr.Column(elem_classes=["tab-shell"]):
                    gr.Markdown("### 智能问答")
                    gr.Markdown("输入故障现象、设备名称或处理场景，可快速获得专业建议。")
                    qa_input = gr.Textbox(
                        label="请输入问题",
                        placeholder="例如：盘形悬式绝缘子发生闪络的原因是什么？应如何处理？",
                        lines=3
                    )
                    with gr.Row():
                        qa_btn = gr.Button("提交问题", variant="primary", size="lg")
                        qa_clear_btn = gr.Button("清空", variant="secondary", size="lg")
                        qa_progress = gr.Markdown("")
                    qa_output = gr.Textbox(
                        label="专业回答",
                        lines=15,
                        interactive=False
                    )
                    
                    qa_btn.click(fn=lambda: "处理中...", outputs=[qa_progress]).then(
                        fn=power_equipment_qa,
                        inputs=qa_input,
                        outputs=qa_output
                    ).then(fn=lambda: "", outputs=[qa_progress])
                    qa_clear_btn.click(fn=lambda: ("", "", ""), outputs=[qa_input, qa_output, qa_progress])

            # 标签页6：历史记录
            with gr.Tab("历史记录"):
                with gr.Column(elem_classes=["tab-shell"]):
                    with gr.Row():
                        history_type_filter = gr.Dropdown(
                            label="任务类型筛选",
                            choices=["全部", "文搜图", "图搜文", "缺陷检测"],
                            value="全部"
                        )
                        history_days_filter = gr.Dropdown(
                            label="时间范围",
                            choices=[("全部", 0), ("今天", 1), ("近7天", 7), ("近30天", 30)],
                            value=0
                        )
                        history_keyword = gr.Textbox(
                            label="关键词",
                            placeholder="输入关键词筛选查询内容或结果摘要"
                        )
                        history_limit = gr.Slider(
                            label="显示条数",
                            minimum=20,
                            maximum=500,
                            step=20,
                            value=200
                        )

                    with gr.Row():
                        refresh_history_btn = gr.Button("刷新历史记录", variant="secondary", size="lg")
                        export_history_btn = gr.Button("导出 CSV", variant="secondary", size="lg")
                        delete_latest_btn = gr.Button("删除最新", variant="stop", size="lg")
                        delete_all_btn = gr.Button("清空全部历史", variant="danger", size="lg")
                        history_progress = gr.Markdown("")

                    history_status = gr.Textbox(label="任务状态", interactive=False, show_label=False)
                    history_export_file = gr.File(label="导出文件", interactive=False)
                    history_table = gr.Dataframe(
                        label="历史记录",
                        headers=["任务类型", "查询内容", "结果摘要", "时间"],
                        wrap=True,
                        interactive=False
                    )

                def refresh_history(username, search_type, days, keyword, limit):
                    if not username:
                        return [["请先登录", "请先登录", "请先登录", "请先登录"]], "请先登录"

                    # 先直接查库，避免筛选条件或状态丢失导致看起来“刷新没反应”
                    user_id = get_user_id(username)
                    if not user_id:
                        return [["未找到用户", username, "请重新登录后再试", ""]], "未找到用户"

                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    sql = """
                        SELECT search_type, query_content, result_summary, create_time
                        FROM search_history
                        WHERE user_id = ?
                        ORDER BY create_time DESC
                        LIMIT ?
                    """
                    cursor.execute(sql, (user_id, max(1, min(int(limit), 1000))))
                    rows = cursor.fetchall()
                    conn.close()

                    if search_type and search_type not in ("全部", "All"):
                        rows = [r for r in rows if r[0] == search_type]
                    keyword = (keyword or "").strip()
                    if keyword:
                        rows = [r for r in rows if keyword in str(r[1]) or keyword in str(r[2])]
                    days = int(days or 0)
                    if days > 0:
                        cutoff = datetime.now() - timedelta(days=days - 1)
                        filtered_rows = []
                        for r in rows:
                            try:
                                row_dt = datetime.strptime(str(r[3]), "%Y-%m-%d %H:%M:%S")
                                if row_dt >= cutoff:
                                    filtered_rows.append(r)
                            except Exception:
                                filtered_rows.append(r)
                        rows = filtered_rows

                    if not rows:
                        rows = [["暂无记录", "暂无记录", "暂无记录", "暂无记录"]]
                    count = 0 if rows[0][0] in ["暂无记录", "查询失败", "请先登录"] else len(rows)
                    return rows, f"历史记录已刷新（{count}条），用户={username}，类型={search_type}，天数={days}，关键词={keyword}，限制={limit}"

                def export_history_csv(username, search_type, days, keyword, limit):
                    rows = get_user_display_history(
                        username,
                        search_type=search_type,
                        keyword=keyword,
                        limit=limit,
                        days=days
                    )
                    if not rows or rows[0][0] in ["暂无记录", "查询失败"]:
                        return None, "当前筛选条件下没有可导出的历史记录"

                    export_dir = os.path.join(os.getcwd(), "exports")
                    os.makedirs(export_dir, exist_ok=True)
                    file_name = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    file_path = os.path.join(export_dir, file_name)

                    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                        writer = csv.writer(f)
                        writer.writerow(["任务类型", "查询内容", "结果摘要", "时间"])
                        writer.writerows(rows)

                    return file_path, f"导出完成：{file_name}"

                refresh_history_btn.click(fn=lambda: "处理中...", outputs=[history_progress]).then(
                    fn=refresh_history,
                    inputs=[current_user, history_type_filter, history_days_filter, history_keyword, history_limit],
                    outputs=[history_table, history_status]
                ).then(fn=lambda: "", outputs=[history_progress])

                export_history_btn.click(fn=lambda: "处理中...", outputs=[history_progress]).then(
                    fn=export_history_csv,
                    inputs=[current_user, history_type_filter, history_days_filter, history_keyword, history_limit],
                    outputs=[history_export_file, history_status]
                ).then(fn=lambda: "", outputs=[history_progress])

                def delete_latest_and_refresh(username, search_type, days, keyword, limit):
                    if not username:
                        return "请先登录", [["请先登录", "请先登录", "请先登录", "请先登录"]]
                    status_msg, _ = delete_latest_history(username)
                    rows = get_user_display_history(
                        username,
                        search_type=search_type,
                        keyword=keyword,
                        limit=limit,
                        days=days
                    )
                    if not rows:
                        rows = [["暂无记录", "暂无记录", "暂无记录", "暂无记录"]]
                    return status_msg, rows

                def delete_all_and_refresh(username, search_type, days, keyword, limit):
                    if not username:
                        return "请先登录", [["请先登录", "请先登录", "请先登录", "请先登录"]]
                    status_msg, _ = delete_all_history(username)
                    rows = get_user_display_history(
                        username,
                        search_type=search_type,
                        keyword=keyword,
                        limit=limit,
                        days=days
                    )
                    if not rows:
                        rows = [["暂无记录", "暂无记录", "暂无记录", "暂无记录"]]
                    return status_msg, rows

                delete_latest_btn.click(fn=lambda: "处理中...", outputs=[history_progress]).then(
                    fn=delete_latest_and_refresh,
                    inputs=[current_user, history_type_filter, history_days_filter, history_keyword, history_limit],
                    outputs=[history_status, history_table]
                ).then(fn=lambda: "", outputs=[history_progress])
                delete_all_btn.click(fn=lambda: "处理中...", outputs=[history_progress]).then(
                    fn=delete_all_and_refresh,
                    inputs=[current_user, history_type_filter, history_days_filter, history_keyword, history_limit],
                    outputs=[history_status, history_table]
                ).then(fn=lambda: "", outputs=[history_progress])

            # 标签页7：数据看板
            with gr.Tab("数据看板"):
                with gr.Column(elem_classes=["tab-shell"]):
                    dashboard_time_cost = gr.Textbox(label="刷新耗时", interactive=False, show_label=False)
                    refresh_btn = gr.Button("刷新统计", variant="primary", size="lg")
                    try:
                        initial_stats = get_statistics_data()
                    except Exception as e:
                        initial_stats = {
                            "core_metrics": (0, 0, 0, 0),
                            "trend": {},
                            "search_type": {},
                            "category_dist": {},
                            "wordcloud_text": ""
                        }
                        dashboard_time_cost = gr.Textbox(label="刷新耗时", interactive=False, show_label=False, value=f"统计加载失败：{str(e)}")

                    core_metrics_plot = gr.Plot(value=create_core_metrics_cards(initial_stats["core_metrics"]), show_label=False)

                    with gr.Row():
                        trend_plot = gr.Plot(value=create_trend_line_chart(initial_stats["trend"]), show_label=False)
                        type_pie_plot = gr.Plot(value=create_search_type_pie(initial_stats["search_type"]), show_label=False)

                    with gr.Row():
                        category_bar_plot = gr.Plot(value=create_category_bar_chart(initial_stats["category_dist"]), show_label=False)
                        wordcloud_plot = gr.Plot(value=create_wordcloud_figure(initial_stats["wordcloud_text"]), show_label=False)
                    
                    def refresh_all_plots():
                        t0 = datetime.now()
                        new_stats = get_statistics_data()
                        elapsed_ms = int((datetime.now() - t0).total_seconds() * 1000)
                        return [
                            create_core_metrics_cards(new_stats["core_metrics"]),
                            create_trend_line_chart(new_stats["trend"]),
                            create_search_type_pie(new_stats["search_type"]),
                            create_category_bar_chart(new_stats["category_dist"]),
                            create_wordcloud_figure(new_stats["wordcloud_text"]),
                            f"刷新完成，用时 {elapsed_ms} 毫秒"
                        ]

                    refresh_btn.click(
                        fn=refresh_all_plots,
                        outputs=[core_metrics_plot, trend_plot, type_pie_plot, category_bar_plot, wordcloud_plot, dashboard_time_cost]
                    )

    # 登录/退出事件
    def handle_login(username, password):
        msg, success, user = login_user(username, password)
        if success:
            return (msg, gr.update(visible=False), gr.update(visible=True), True, user,
                    f"### 当前用户：{user}", get_user_display_history(user))
        else:
            return (msg, gr.update(visible=True), gr.update(visible=False), False, "", "", [])
    login_btn.click(fn=handle_login, inputs=[login_username, login_password],
                  outputs=[login_status, auth_panel, main_panel, is_logged_in, current_user, user_info, history_table])

    reg_btn.click(fn=register_user, inputs=[reg_username, reg_password, reg_confirm_password],
                outputs=reg_status)

    def handle_logout(username):
        empty_history = [["无记录", "无记录", "无记录", "无记录"]]
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            False,
            "",
            empty_history,
            "当前用户：-"
        )

    logout_btn.click(
        fn=handle_logout,
        inputs=[current_user],
        outputs=[auth_panel, main_panel, is_logged_in, current_user, history_table, user_info]
    )

def launch_app():
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    inbrowser = os.getenv("GRADIO_INBROWSER", "false").lower() == "true"

    try:
        demo.launch(
            share=share,
            inbrowser=inbrowser,
            server_name=host,
            server_port=port,
        )
    except ValueError as e:
        if "localhost 无法访问" in str(e) and not share:
            demo.launch(
                share=False,
                inbrowser=False,
                server_name="0.0.0.0",
                server_port=port,
            )
        else:
            raise

if __name__ == "__main__":
    init_environment()
    launch_app()
