import os
import os.path as osp
import argparse
import cv2
import numpy as np
import json
import time
import shutil
import tempfile
import faiss
import sqlite3
from collections import deque
from typing import Set, List, Dict

from fr.adaface_openvino import AdaFaceOpenVINO
from fd.scrfd_openvino_sd_blur_detect import SCRFD


# ============================================================================
# Constants
# ============================================================================

# Image file extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Model paths
ASSETS_DIR = osp.expanduser('buffalo_l')
FD_MODEL_PATH = './FD/F16/model.xml'
ADAFACE_MODEL_PATH = './Adaface/R50/F16/model.xml'

# Model configuration
BLUR_THRESHOLD = 50
DEVICE_ID = 0

# Similarity thresholds
SIMILARITY_THRESHOLD_LOW = 0.2
SIMILARITY_THRESHOLD_MEDIUM = 0.28
DEFAULT_VIDEO_THRESHOLD = 0.44

# Video processing constants
PERSIST_FRAMES = 6
MAX_OVERLAYS = 5
FAISS_NLIST = 1024
FAISS_NPROBE = 100
MAX_EMBEDDINGS_DEFAULT = 100000

# Paths (can be overridden)
DEFAULT_SRC_FOLDER = "/home/sr/edge_face"
DEFAULT_PHOTOS_FOLDER = "/home/sr/Photos"
DEFAULT_JSONL_PATH = 'final_embeddings_adaface_ov_fp16.jsonl'
DEFAULT_DB_PATH = 'embeddings.db'
DEFAULT_EMBEDDINGS_DIR = 'embeddings_storage'

# ============================================================================
# Model Initialization
# ============================================================================

def initialize_models(device_id: int = DEVICE_ID, blur_threshold: int = BLUR_THRESHOLD) -> tuple[SCRFD, AdaFaceOpenVINO]:
    """
    Initialize face detection and recognition models.
    
    Returns:
        Tuple of (detector, recognizer)
    """
    assets_dir = osp.expanduser(ASSETS_DIR)
    detector = SCRFD(os.path.join(assets_dir, FD_MODEL_PATH))
    detector.prepare(device_id, blur_threshold=blur_threshold)
    
    adaface_model_path = os.path.join(assets_dir, ADAFACE_MODEL_PATH)
    rec = AdaFaceOpenVINO(adaface_model_path)
    rec.prepare(device_id)
    
    return detector, rec


# Initialize models globally
detector, rec = initialize_models()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=str)
    parser.add_argument('img2', type=str)
    return parser.parse_args()


# ============================================================================
# Core Face Recognition Functions
# ============================================================================

def compare_two_images(args: argparse.Namespace) -> tuple[float, str]:
    """
    Compare two images and determine if they contain the same person.
    
    Args:
        args: Namespace with img1 and img2 paths
    
    Returns:
        Tuple of (similarity_score, conclusion_string)
    """
    image1 = cv2.imread(args.img1)
    image2 = cv2.imread(args.img2)
    
    if image1 is None:
        return -1.0, f"Could not load Image-1: {args.img1}"
    if image2 is None:
        return -1.0, f"Could not load Image-2: {args.img2}"
    
    bboxes1, kpss1 = detector.autodetect(image1, max_num=1)
    if bboxes1.shape[0] == 0:
        return -1.0, "Face not found in Image-1"
    
    bboxes2, kpss2 = detector.autodetect(image2, max_num=1)
    if bboxes2.shape[0] == 0:
        return -1.0, "Face not found in Image-2"
    
    kps1 = kpss1[0]
    kps2 = kpss2[0]
    feat1 = rec.get(image1, kps1)
    feat2 = rec.get(image2, kps2)
    sim = rec.compute_sim(feat1, feat2)
    
    if sim < SIMILARITY_THRESHOLD_LOW:
        conclu = 'They are NOT the same person'
    elif sim >= SIMILARITY_THRESHOLD_LOW and sim < SIMILARITY_THRESHOLD_MEDIUM:
        conclu = 'They are LIKELY TO be the same person'
    else:
        conclu = 'They ARE the same person'
    
    return sim, conclu


def embed_image(args: argparse.Namespace) -> np.ndarray | None:
    image = cv2.imread(args.img1)
    bboxes, kpss = detector.autodetect(image, max_num=1)
    if bboxes.shape[0]==0:
        return None
    kps = kpss[0]
    feat = rec.get(image, kps)
    return feat


# ============================================================================
# File Utility Functions
# ============================================================================

def is_image_file(filename: str) -> bool:
    return osp.splitext(filename)[1].lower() in IMAGE_EXTS

def base_id_from_filename(fname: str) -> str:
    """
    Given filename (without path), derive base id like:
    'VID_30872_001_02.jpg' -> 'VID_30872_001'
    Works by joining first three underscore-separated parts.
    If filename has fewer parts, returns the stem.
    """
    stem = osp.splitext(osp.basename(fname))[0]
    parts = stem.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])
    return stem

def load_jsonl_paths_and_bases(jsonl_path: str) -> (List[str], Set[str]):
    """
    Return list of JSONL lines and set of base_ids present in the file.
    We keep full lines to allow safe rewriting.
    """
    lines = []
    bases = set()
    if not osp.exists(jsonl_path):
        return lines, bases
    with open(jsonl_path, 'r') as fh:
        for ln in fh:
            ln_strip = ln.rstrip('\n')
            if not ln_strip:
                continue
            lines.append(ln_strip)
            try:
                obj = json.loads(ln_strip)
                p = obj.get('path')
                if p:
                    b = base_id_from_filename(osp.basename(p))
                    bases.add(b)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                # skip malformed for base gathering but keep line
                continue
    return lines, bases

# ============================================================================
# JSONL Processing Functions
# ============================================================================

def init_embeddings_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initialize SQLite database for storing embeddings metadata.
    Creates table if it doesn't exist.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            embeddings_path TEXT NOT NULL UNIQUE,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_person_id ON embeddings(person_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_path ON embeddings(embeddings_path)')
    conn.commit()
    conn.close()


def go_through(src_folder: str = "/home/sr/ov_fr_pipeline/videos/input/Train",
               jsonl_out: str = DEFAULT_JSONL_PATH,
               db_path: str = DEFAULT_DB_PATH,
               embeddings_dir: str = DEFAULT_EMBEDDINGS_DIR) -> int:
    """
    Scans src_folder (top-level only) for images (including names with _01/_02 suffixes).
    For any base_id for which new variant images are found, removes old JSONL entries
    with that base_id and replaces them with the new entries.
    Also maintains a SQLite database with person_id and embeddings_path.
    Returns number of new embeddings written.
    """
    # Initialize database and embeddings directory
    init_embeddings_db(db_path)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # 1) Gather images in folder (non-recursive)
    if not osp.isdir(src_folder):
        raise ValueError(f"src_folder does not exist: {src_folder}")

    all_files = sorted([f for f in os.listdir(src_folder) if is_image_file(f)])
    if not all_files:
        print("No image files found in", src_folder)
        return 0

    # group images by base_id
    imgs_by_base: Dict[str, List[str]] = {}
    for fn in all_files:
        base = base_id_from_filename(fn)
        imgs_by_base.setdefault(base, []).append(fn)

    # 2) Read existing jsonl and gather base ids already present
    existing_lines, existing_bases = load_jsonl_paths_and_bases(jsonl_out)

    # 3) Decide which base_ids we will replace (those for which we have new images)
    #    Only consider base_ids for which we have at least one new file in src_folder
    candidate_bases = set(imgs_by_base.keys())
    bases_to_replace = candidate_bases.intersection(existing_bases)

    # 3b) Handle database cleanup for bases_to_replace
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if bases_to_replace:
        # Get embeddings paths for old entries that will be deleted
        placeholders = ','.join(['?'] * len(bases_to_replace))
        cursor.execute(f'SELECT embeddings_path FROM embeddings WHERE person_id IN ({placeholders})',
                      list(bases_to_replace))
        old_emb_paths = [row[0] for row in cursor.fetchall()]
        
        # Delete old database entries
        cursor.execute(f'DELETE FROM embeddings WHERE person_id IN ({placeholders})',
                      list(bases_to_replace))
        
        # Delete corresponding .npy files
        for old_emb_path in old_emb_paths:
            if osp.exists(old_emb_path):
                try:
                    os.remove(old_emb_path)
                except OSError:
                    pass

    # 4) Prepare to rewrite JSONL: filter out lines whose path's base_id is in bases_to_replace
    #    We'll write a new temp file that contains filtered old lines, then append new items.
    tmp_fd, tmp_path = tempfile.mkstemp(prefix='jsonl_tmp_', suffix='.tmp')
    os.close(tmp_fd)
    try:
        with open(tmp_path, 'w') as tmp_f:
            # write old lines that we keep
            for ln in existing_lines:
                try:
                    obj = json.loads(ln)
                    p = obj.get('path')
                    b = base_id_from_filename(osp.basename(p)) if p else None
                    if b and b in bases_to_replace:
                        # drop this old entry (we will replace it)
                        continue
                    tmp_f.write(ln + '\n')
                except (json.JSONDecodeError, KeyError, TypeError):
                    # keep malformed lines as-is
                    tmp_f.write(ln + '\n')

        # Now append new embeddings for all images in src_folder.
        # We'll open tmp_path in append mode and write new JSON lines for images.
        added = 0
        with open(tmp_path, 'a') as tmp_f, open('failed_images.txt', 'a') as failed_f:
            for base, files in sorted(imgs_by_base.items()):
                # For base IDs that had old entries and for new bases both, we will (re)compute embeddings.
                for fn in sorted(files):
                    img_path = osp.join(src_folder, fn)
                    # Skip if exact path already present in the kept lines (avoid duplicates)
                    # For simplicity, we will check existing_lines + tmp_f current content by scanning existing_lines and bases_to_replace logic:
                    # If this exact path existed previously and we didn't drop it, skip; otherwise compute.
                    already_present = False
                    for ln in existing_lines:
                        try:
                            obj = json.loads(ln)
                            if obj.get('path') == img_path:
                                already_present = True
                                break
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue
                    if already_present:
                        continue

                    args = argparse.Namespace(img1=img_path)
                    try:
                        feat = embed_image(args)
                    except Exception as e:
                        print(f"[ERROR] embed_image failed for {img_path}: {e}")
                        failed_f.write(f"{img_path}\tEXC:{e}\n")
                        failed_f.flush()
                        continue

                    if feat is None:
                        print(f"[WARN] embed_image returned None for {img_path}")
                        failed_f.write(img_path + '\n')
                        failed_f.flush()
                        continue

                    feat_arr = np.array(feat, dtype=np.float32).reshape(-1)
                    
                    # Write to JSONL (existing functionality)
                    record = {'path': img_path, 'feat': feat_arr.tolist()}
                    tmp_f.write(json.dumps(record) + '\n')
                    tmp_f.flush()
                    
                    # Also save to database and as .npy file
                    person_id = base  # person_id is the base_id
                    embedding_filename = f"{person_id}_{fn}_{int(time.time())}.npy"
                    embedding_path = osp.join(embeddings_dir, embedding_filename)
                    np.save(embedding_path, feat_arr)
                    
                    # Insert into database
                    try:
                        cursor.execute('''
                            INSERT INTO embeddings (person_id, embeddings_path, image_path)
                            VALUES (?, ?, ?)
                        ''', (person_id, embedding_path, img_path))
                    except sqlite3.IntegrityError:
                        # If embeddings_path already exists, skip or update
                        print(f"Warning: Embedding path already exists: {embedding_path}")
                    
                    added += 1

        # Commit database changes
        conn.commit()

        # Replace original jsonl with temp (atomic move)
        shutil.move(tmp_path, jsonl_out)
        print(f"Rewrote {jsonl_out}: removed {len(bases_to_replace)} base(s) and added {added} new embeddings.")
        print(f"Updated database {db_path}: removed {len(bases_to_replace)} person_id(s) and added {added} new embeddings.")
        return added

    finally:
        # cleanup temp if it still exists
        if osp.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        # Ensure database connection is closed
        try:
            conn.close()
        except:
            pass


# ============================================================================
# Video Processing Functions
# ============================================================================

def compare_video_detailed(video_path: str, threshold: float = DEFAULT_VIDEO_THRESHOLD, 
                          output_path: str = "output_video.mp4") -> None:
    """
    Shows all matched reference images side-by-side (top-right) for each frame,
    and displays the full image filename (without extension) as ID label.
    """
    src_folder = DEFAULT_SRC_FOLDER

    # Load embeddings DB
    img_paths, person_labels, features_matrix = load_embeddings_from_jsonl(
        DEFAULT_JSONL_PATH, max_embeddings=MAX_EMBEDDINGS_DEFAULT
    )

    # Build FAISS index
    index = build_faiss_index(features_matrix, use_ivf=True, nlist=FAISS_NLIST, nprobe=FAISS_NPROBE)

    print(f'Loaded {len(img_paths)} embeddings')

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_faces = 0
    matched_unknown = correctly_rejected = 0

    ref_cache: Dict[str, np.ndarray] = {}
    current_refs: List[tuple[np.ndarray, str]] = []
    frames_since_match = 999
    MATCH_SCORE_MIN = threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        bboxes, kpss = detector.autodetect(frame, max_num=8)

        if bboxes.shape[0] == 0:
            if current_refs and frames_since_match <= PERSIST_FRAMES:
                overlay_refs_montage(frame, current_refs)
                frames_since_match += 1
            out.write(frame)
            continue

        total_faces += bboxes.shape[0]

        matches = {}
        for i in range(bboxes.shape[0]):
            kps = kpss[i]
            box = bboxes[i].astype(int)
            feat = rec.get(frame, kps)
            if feat is None:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                continue

            feat = np.array(feat, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(feat)
            D, I = index.search(feat, 1)
            sim = float(D[0][0])
            match_index = int(I[0][0])
            matched_img = img_paths[match_index]
            label = osp.splitext(osp.basename(matched_img))[0]

            if sim >= MATCH_SCORE_MIN:
                prev = matches.get(label)
                if prev is None or sim > prev[0]:
                    matches[label] = (sim, matched_img)
                color = (0, 255, 0)
                matched_unknown += 1
            else:
                color = (0, 0, 255)
                correctly_rejected += 1

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"sim ({sim:.2f})", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if matches:
            sorted_matches = sorted(matches.items(), key=lambda kv: kv[1][0], reverse=True)
            display = []
            for idx, (label, (sim_val, path)) in enumerate(sorted_matches[:MAX_OVERLAYS]):
                ref_img = get_ref_image(path, src_folder, ref_cache)
                if ref_img is not None:
                    display.append((ref_img, label))
            current_refs = display
            frames_since_match = 0
        else:
            frames_since_match += 1
            if frames_since_match > PERSIST_FRAMES:
                current_refs = []

        if current_refs and frames_since_match <= PERSIST_FRAMES:
            overlay_refs_montage(frame, current_refs)

        out.write(frame)

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            avg_fps_now = frame_count / elapsed if elapsed > 0 else 0.0
            print(f"[{frame_count} frames] FPS: {avg_fps_now:.2f}")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0.0

    print(f"\n==== Final Stats ====")
    print(f"Total frames: {frame_count}, Total faces: {total_faces}, Average FPS: {avg_fps:.2f}")
    print(f"Matched faces: {matched_unknown}")
    print(f"Rejected faces: {correctly_rejected}")
    print(f"Output video saved to: {output_path}")


# ============================================================================
# Display and Visualization Functions
# ============================================================================

def get_display_label_from_path(path: str) -> str:
    """
    Given a path like '/.../VID_30872_001_01.jpg' returns '30872_001'.
    Fallbacks:
      - if name starts with 'VID_' and has >=3 underscore parts -> join parts[1] and parts[2]
      - elif has >=3 parts -> join parts[0] and parts[1]
      - else returns stem (filename without extension)
    """
    stem = osp.splitext(osp.basename(path))[0]
    parts = stem.split('_')
    if len(parts) >= 3 and parts[0].upper() == 'VID':
        return f"{parts[1]}_{parts[2]}"
    if len(parts) >= 2:
        # fallback: join first two parts (works for many common patterns)
        return f"{parts[0]}_{parts[1]}"
    return stem


def get_ref_image(path: str, src_folder: str, ref_cache: Dict[str, np.ndarray]) -> np.ndarray | None:
    """
    Load a reference image with caching. Tries the full path first, then falls back to src_folder.
    """
    if path in ref_cache:
        return ref_cache[path]
    img = cv2.imread(path)
    if img is None:
        alt = osp.join(src_folder, osp.basename(path))
        img = cv2.imread(alt)
        if img is None:
            print(f"Warning: cannot load reference image: {path}")
            return None
    ref_cache[path] = img
    return img


def load_embeddings_from_jsonl(jsonl_path: str, max_embeddings: int = None, 
                               use_last_n: int = None, label_func=None) -> tuple[List[str], List[str], np.ndarray]:
    """
    Load embeddings from a JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        max_embeddings: Maximum number of embeddings to load (from start)
        use_last_n: Load only last N embeddings (uses deque)
        label_func: Optional function to extract label from path
    
    Returns:
        Tuple of (img_paths, labels, features_matrix)
    """
    img_paths = []
    labels = []
    feats_list = []
    
    if use_last_n:
        q = deque(maxlen=use_last_n)
        with open(jsonl_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                q.append(line)
        lines = list(q)
        print(f"Loaded {len(lines)} jsonl lines (last {use_last_n}) from {jsonl_path}")
    else:
        lines = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                lines.append(line.strip())
                if max_embeddings and len(lines) >= max_embeddings:
                    break
    
    for line in lines:
        if not line:
            continue
        try:
            data = json.loads(line)
            path = data.get('path')
            feat = np.array(data.get('feat'), dtype='float32')
            if path is None or feat is None:
                continue
            
            if label_func:
                label = label_func(path)
            else:
                label = osp.splitext(osp.basename(path))[0]
            
            img_paths.append(path)
            labels.append(label)
            feats_list.append(feat)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            continue
    
    if len(feats_list) == 0:
        raise RuntimeError("No embeddings loaded from jsonl. Aborting.")
    
    features_matrix = np.stack(feats_list).astype('float32')
    return img_paths, labels, features_matrix


def build_faiss_index(features_matrix: np.ndarray, use_ivf: bool = True, 
                      nlist: int = 1024, nprobe: int = 100) -> faiss.Index:
    """
    Build a FAISS index from features matrix.
    
    Args:
        features_matrix: Numpy array of shape (n_vectors, dim)
        use_ivf: Whether to use IVF index (approximate) or Flat (exact)
        nlist: Number of clusters for IVF
        nprobe: Number of clusters to probe for IVF
    
    Returns:
        FAISS index
    """
    num_vectors, dim = features_matrix.shape
    faiss.normalize_L2(features_matrix)
    
    if use_ivf and num_vectors >= max(2, nlist):
        print(f"Building IndexIVFFlat (approximate) with {num_vectors} vectors.")
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(features_matrix)
        index.add(features_matrix)
        index.nprobe = nprobe
    else:
        print(f"Using IndexFlatIP (exact) with {num_vectors} vectors.")
        index = faiss.IndexFlatIP(dim)
        index.add(features_matrix)
    
    return index


def overlay_refs_montage(frame: np.ndarray,
                         refs: List[tuple[np.ndarray, str]],
                         max_height: int = 100,
                         border: int = 5,
                         spacing: int = 18,
                         right_padding: int = 20,
                         caption_font_scale: float = 0.5) -> None:
    """
    Draw a horizontal montage of reference images (side-by-side) at top-right on the frame.
    
    Args:
        frame: Video frame to draw on
        refs: List of tuples (ref_img, ref_label)
        max_height: Maximum height for reference images
        border: Border width around images
        spacing: Spacing between images
        right_padding: Extra padding on right side
        caption_font_scale: Font scale for captions
    """
    if not refs:
        return

    H, W = frame.shape[:2]
    resized_imgs, captions = [], []

    for ref_img, ref_label in refs:
        if ref_img is None:
            continue
        ih, iw = ref_img.shape[:2]
        if ih == 0:
            continue
        scale = max_height / float(ih)
        w = int(iw * scale)
        img_resized = cv2.resize(ref_img, (w, max_height), interpolation=cv2.INTER_AREA)
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        resized_imgs.append(img_resized)
        captions.append(ref_label)

    if len(resized_imgs) == 0:
        return

    caption_space = 20  # room for labels
    total_w = (
        sum(img.shape[1] for img in resized_imgs)
        + spacing * (len(resized_imgs) - 1)
        + 2 * border * len(resized_imgs)
        + right_padding
    )
    total_h = max_height + 2 * border + caption_space

    padding = 10
    x2 = W - padding
    x1 = x2 - total_w
    y1 = padding
    y2 = y1 + total_h

    # Scale down if montage too wide for screen
    if x1 < 0:
        scale_down = (W - 2 * padding) / float(total_w)
        if scale_down <= 0:
            return
        new_imgs = []
        for img in resized_imgs:
            ih, iw = img.shape[:2]
            nw = max(1, int(iw * scale_down))
            nh = max(1, int(ih * scale_down))
            new_imgs.append(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA))
        resized_imgs = new_imgs
        total_w = (
            sum(img.shape[1] for img in resized_imgs)
            + spacing * (len(resized_imgs) - 1)
            + 2 * border * len(resized_imgs)
            + right_padding
        )
        total_h = resized_imgs[0].shape[0] + 2 * border + caption_space
        x2 = W - padding
        x1 = x2 - total_w
        y2 = y1 + total_h
        if x1 < 0:
            x1 = padding
            x2 = padding + total_w

    # Background rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)

    # Paste each image left-to-right
    x_cursor = x1
    for idx, img in enumerate(resized_imgs):
        w = img.shape[1]
        h = img.shape[0]
        dst_x1 = x_cursor + border
        dst_y1 = y1 + border
        dst_x2 = dst_x1 + w
        dst_y2 = dst_y1 + h

        frame[dst_y1:dst_y2, dst_x1:dst_x2] = img
        cv2.rectangle(frame, (dst_x1, dst_y1), (dst_x2 - 1, dst_y2 - 1), (255, 255, 255), 1)

        caption = captions[idx]
        if caption:
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(caption, font, caption_font_scale, 1)[0]
            txt_x = dst_x1
            txt_y = dst_y2 + 16
            if txt_y + 12 < H:
                # Prevent label from overflowing right side
                if txt_x + txt_size[0] > W - 10:
                    txt_x = W - txt_size[0] - 10
                cv2.putText(frame, caption, (txt_x, txt_y),
                            font, caption_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        x_cursor = dst_x2 + border + spacing

def compare_video_last_detailed(video_path: str, threshold: float = DEFAULT_VIDEO_THRESHOLD, 
                                output_path: str = "output_video.mp4",
                                jsonl_path: str = DEFAULT_JSONL_PATH,
                                load_last_n: int = MAX_EMBEDDINGS_DEFAULT) -> None:
    """
    Loads only the last `load_last_n` embeddings from jsonl_path and uses them for matching.
    Shows all matched reference images side-by-side (top-right). Displays compressed label (e.g. 30872_001).
    """
    src_folder = DEFAULT_SRC_FOLDER

    # Load embeddings using helper function
    img_paths, person_labels, features_matrix = load_embeddings_from_jsonl(
        jsonl_path, use_last_n=load_last_n, label_func=get_display_label_from_path
    )

    # Build FAISS index
    index = build_faiss_index(features_matrix, use_ivf=True, nlist=FAISS_NLIST, nprobe=FAISS_NPROBE)

    print(f'Loaded {len(img_paths)} embeddings')

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_faces = 0
    matched_unknown = correctly_rejected = 0

    ref_cache: Dict[str, np.ndarray] = {}
    current_refs: List[tuple[np.ndarray, str]] = []
    frames_since_match = 999
    MATCH_SCORE_MIN = threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        bboxes, kpss = detector.autodetect(frame, max_num=8)

        if bboxes.shape[0] == 0:
            if current_refs and frames_since_match <= PERSIST_FRAMES:
                overlay_refs_montage(frame, current_refs)
                frames_since_match += 1
            out.write(frame)
            continue

        total_faces += bboxes.shape[0]

        matches = {}
        for i in range(bboxes.shape[0]):
            kps = kpss[i]
            box = bboxes[i].astype(int)
            feat = rec.get(frame, kps)
            if feat is None:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                continue

            feat = np.array(feat, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(feat)
            D, I = index.search(feat, 1)
            sim = float(D[0][0])
            match_index = int(I[0][0])
            matched_img = img_paths[match_index]
            # compute compressed label for display (e.g., 30872_001)
            label = get_display_label_from_path(matched_img)

            if sim >= MATCH_SCORE_MIN:
                prev = matches.get(label)
                if prev is None or sim > prev[0]:
                    matches[label] = (sim, matched_img)
                color = (0, 255, 0)
                matched_unknown += 1
            else:
                color = (0, 0, 255)
                correctly_rejected += 1

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"sim ({sim:.2f})", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if matches:
            sorted_matches = sorted(matches.items(), key=lambda kv: kv[1][0], reverse=True)
            display = []
            for idx, (label, (sim_val, path)) in enumerate(sorted_matches[:MAX_OVERLAYS]):
                ref_img = get_ref_image(path, src_folder, ref_cache)
                if ref_img is not None:
                    display.append((ref_img, label))
            current_refs = display
            frames_since_match = 0
        else:
            frames_since_match += 1
            if frames_since_match > PERSIST_FRAMES:
                current_refs = []

        if current_refs and frames_since_match <= PERSIST_FRAMES:
            overlay_refs_montage(frame, current_refs)

        out.write(frame)

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            avg_fps_now = frame_count / elapsed if elapsed > 0 else 0.0
            print(f"[{frame_count} frames] FPS: {avg_fps_now:.2f}")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0.0

    print(f"\n==== Final Stats ====")
    print(f"Total frames: {frame_count}, Total faces: {total_faces}, Average FPS: {avg_fps:.2f}")
    print(f"Matched faces: {matched_unknown}")
    print(f"Rejected faces: {correctly_rejected}")
    print(f"Output video saved to: {output_path}")


def compare_video_live_display(video_path: str, threshold: float = DEFAULT_VIDEO_THRESHOLD, 
                               output_path: str = "output_video.mp4") -> None:
    """
    Process video with live display window showing face recognition results.
    """
    src_folder = DEFAULT_PHOTOS_FOLDER

    # Load embeddings DB
    img_paths = []
    person_ids = []
    all_feats = []

    num_embeddings = 0
    with open('final_embeddings_adaface_ov.jsonl', 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                path = data['path']
                feats = np.array(data['feat'])
                img_data = path.split('/')[-1]
                parts = img_data.split('_')
                if len(parts) < 2:
                    continue
                person_no = parts[1]
                img_paths.append(path)
                all_feats.append(feats)
                person_ids.append(person_no)
                num_embeddings += 1
                if num_embeddings == 50:
                    break
            except (json.JSONDecodeError, KeyError, ValueError, IndexError):
                continue

    if len(all_feats) == 0:
        raise RuntimeError("No embeddings loaded from jsonl. Aborting.")

    features_matrix = np.stack(all_feats).astype('float32')
    dim = features_matrix.shape[1]
    faiss.normalize_L2(features_matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(features_matrix)

    print(f'Loaded {len(img_paths)} embeddings')

    # Define known persons present in DB
    known_persons = ['0005', '0010', '0015']

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Stats counters
    total_faces = 0
    matched_known = 0
    matched_unknown = 0
    missed_known = 0
    correctly_rejected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        bboxes, kpss = detector.autodetect(frame, max_num=5)
        if bboxes.shape[0] > 0:
            total_faces += bboxes.shape[0]

            for i in range(bboxes.shape[0]):
                kps = kpss[i]
                box = bboxes[i].astype(int)
                feat = rec.get(frame, kps)
                if feat is None:
                    continue

                feat = np.array(feat, dtype='float32').reshape(1, -1)
                faiss.normalize_L2(feat)
                D, I = index.search(feat, 1)

                sim = float(D[0][0])
                match_index = int(I[0][0])
                matched_img = img_paths[match_index]
                match_person_no = matched_img.split('_')[1]

                # Determine match status
                if sim >= threshold:
                    if match_person_no in known_persons:
                        status = f"sim ({sim:.2f})"
                        color = (0, 255, 0)
                        matched_known += 1
                    else:
                        status = f"sim ({sim:.2f})"
                        color = (0, 255, 0)
                        matched_unknown += 1
                else:
                    if match_person_no in known_persons:
                        status = f"sim ({sim:.2f})"
                        color = (0, 0, 255)
                        missed_known += 1
                    else:
                        status = f"sim ({sim:.2f})"
                        color = (0, 0, 255)
                        correctly_rejected += 1

                # Draw bounding box and label
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, status, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS display overlay
        elapsed = time.time() - start_time
        fps_now = frame_count / elapsed
        fps_text = f"FPS: {fps_now:.2f}"
        cv2.putText(frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # Show live video window
        cv2.imshow("Face Recognition Live", frame)

        # Write frame to output video
        out.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Quit by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time

    print(f"\n--- Performance Stats ---")
    print(f"Total frames: {frame_count}, Total faces: {total_faces}, Average FPS: {avg_fps:.2f}")
    print(f"Matched known faces: {matched_known}")
    print(f"Matched unknown faces (false positive): {matched_unknown}")
    print(f"Missed known faces (false negative): {missed_known}")
    print(f"Correctly rejected unknown faces: {correctly_rejected}")
    print(f"Output video saved to: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    st = time.time()
    # compare_video('videos/input/classroom.gif', threshold=0.45)
    go_through("/home/sr/ov_fr_pipeline/videos/input/Train", "final_embeddings_adaface_ov_trial.jsonl")
    # compare_video_last_detailed('/home/sr/ov_fr/videos/input/15724-865412877_medium.mp4', threshold=0.49, output_path='videos/output/ov/Train_50_Blur_fp16_fp16.mp4')
    # compare_video_live_display('videos/input/39837-424360872_small.mp4', threshold=0.45, output_path='videos/output/output_video_walking_with_match_live_1920-1080_45.mp4')
    print(f"Time taken: {time.time() - st} seconds")
