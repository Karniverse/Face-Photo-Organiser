import os
import sys
import shutil
import pickle
import ctypes
import cv2
import numpy as np

from dotenv import load_dotenv
from insightface.app import FaceAnalysis


load_dotenv()


# =========================================================
# PRE-FILLED INPUT (Windows console API)
# =========================================================

def _win32_inject(text):

    from ctypes import wintypes

    class CHAR_UNION(ctypes.Union):
        _fields_ = [
            ("UnicodeChar",
                ctypes.c_wchar),
            ("AsciiChar",
                ctypes.c_char),
        ]

    class KEY_EVENT(ctypes.Structure):
        _fields_ = [
            ("bKeyDown",
                wintypes.BOOL),
            ("wRepeatCount",
                wintypes.WORD),
            ("wVirtualKeyCode",
                wintypes.WORD),
            ("wVirtualScanCode",
                wintypes.WORD),
            ("uChar",
                CHAR_UNION),
            ("dwControlKeyState",
                wintypes.DWORD),
        ]

    class INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("KeyEvent", KEY_EVENT)
        ]

    class INPUT_RECORD(ctypes.Structure):
        _fields_ = [
            ("EventType",
                wintypes.WORD),
            ("Event",
                INPUT_UNION),
        ]

    handle = (
        ctypes.windll.kernel32
        .GetStdHandle(
            wintypes.DWORD(
                0xFFFFFFF6
            )
        )
    )

    records = []

    for ch in text:

        r = INPUT_RECORD()
        r.EventType = 0x0001
        r.Event.KeyEvent.bKeyDown = True
        r.Event.KeyEvent.wRepeatCount = 1
        r.Event.KeyEvent.uChar\
            .UnicodeChar = ch
        records.append(r)

        r2 = INPUT_RECORD()
        r2.EventType = 0x0001
        r2.Event.KeyEvent.bKeyDown = False
        r2.Event.KeyEvent.wRepeatCount = 1
        r2.Event.KeyEvent.uChar\
            .UnicodeChar = ch
        records.append(r2)

    arr = (
        INPUT_RECORD * len(records)
    )(*records)

    written = wintypes.DWORD(0)

    ctypes.windll.kernel32\
        .WriteConsoleInputW(
            handle,
            arr,
            len(records),
            ctypes.byref(written)
        )


def input_prefilled(prompt, prefill=''):

    if not prefill:
        return input(prompt)

    if sys.platform == 'win32':

        _win32_inject(prefill)
        return input(prompt)

    # Linux/Mac fallback
    try:
        import readline
        readline.set_startup_hook(
            lambda:
                readline.insert_text(
                    prefill
                )
        )
        try:
            return input(prompt)
        finally:
            readline.set_startup_hook()
    except ImportError:
        return input(prompt)


# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = "photos"
OUTPUT_DIR = "sorted_photos"
MEMORY_FILE = "face_memory.pkl"

SIMILARITY_THRESHOLD = 0.45
VIDEO_CONFIRM_COUNT = 4

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD MEMORY
# =========================================================

if os.path.exists(MEMORY_FILE):

    print("Loading known faces from memory...")

    with open(MEMORY_FILE, "rb") as f:
        known_faces = pickle.load(f)

else:

    print("No memory found")

    known_faces=[]


# =========================================================
# LOAD GPU MODEL
# =========================================================

print("Loading InsightFace models...")

app = FaceAnalysis(
    name='buffalo_l',
    root='models',
    providers=[
        'CUDAExecutionProvider'
    ]
)

app.prepare(
    ctx_id=0,
    det_size=(640,640)
)

print("GPU Face Recognition Ready!")


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(a,b):

    a=np.array(a)
    b=np.array(b)

    return np.dot(a,b)/(
        np.linalg.norm(a)*
        np.linalg.norm(b)
    )


# =========================================================
# MATCH FACE (SILENT - NO PROMPTING)
# =========================================================

def match_face(embedding):

    best_name="Unknown"

    best_similarity=-1

    for saved_name,saved_embedding in known_faces:

        similarity=cosine_similarity(
            embedding,
            saved_embedding
        )

        if similarity>best_similarity:

            best_similarity=similarity

            if similarity>SIMILARITY_THRESHOLD:

                best_name=saved_name

    return best_name


# =========================================================
# VIDEO FRAME EXTRACTION
# =========================================================

def extract_video_frames(
        video_path
):

    cap=cv2.VideoCapture(
        video_path
    )

    fps=cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps<=0:
        fps=30

    interval=int(fps)

    frames=[]

    frame_number=0

    while True:

        success,frame=cap.read()

        if not success:
            break

        if frame_number%interval==0:

            frames.append(
                frame
            )

        frame_number+=1

    cap.release()

    return frames


# =========================================================
# CROP FACE FROM IMAGE
# =========================================================

def crop_face(image,face):

    bbox=face.bbox.astype(
        int
    )

    x1,y1,x2,y2=bbox

    pad=30

    x1=max(0,x1-pad)

    y1=max(0,y1-pad)

    x2=min(
        image.shape[1],
        x2+pad
    )

    y2=min(
        image.shape[0],
        y2+pad
    )

    return image[y1:y2,x1:x2]


# =========================================================
# SORT FILE TO PERSON FOLDER
# =========================================================

def sort_file(
        filepath,
        filename,
        person_name
):

    if not os.path.exists(
            filepath
    ):
        return

    person_dir=os.path.join(
        OUTPUT_DIR,
        person_name
    )

    os.makedirs(
        person_dir,
        exist_ok=True
    )

    destination=os.path.join(
        person_dir,
        filename
    )

    if os.path.exists(destination):
        # Safety check: only delete if the file sizes match
        if os.path.getsize(filepath) == os.path.getsize(destination):
            try:
                os.remove(filepath)
            except OSError:
                pass
        else:
            print(f"\n  -> Warning: {filename} exists in {person_name} but has a different file size! Skipping deletion.")
    else:
        shutil.move(
            filepath,
            destination
        )

# =========================================================
# SUGGEST MATCH (EMBEDDING-BASED)
# =========================================================

# Minimum similarity to be suggested
SUGGEST_THRESHOLD = 0.25


def suggest_match(embedding):

    best_name = None
    best_sim = -1

    for saved_name, saved_emb in (
            known_faces
    ):

        sim = cosine_similarity(
            embedding,
            saved_emb
        )

        if (
            sim > SUGGEST_THRESHOLD
            and
            sim > best_sim
        ):

            best_sim = sim
            best_name = saved_name

    if best_name:

        print(
            f"  Best match: "
            f"{best_name} "
            f"({best_sim:.2f})"
        )

    return best_name


# =========================================================
# DATA STRUCTURES
# =========================================================

# Queue of unknown face encounters
# Each entry: {
#     'embedding', 'face_crop',
#     'filepath', 'filename'
# }
unknown_queue=[]

# Track detected names per file
# filepath -> {
#     'filename', 'names': set()
# }
file_detections={}


# =========================================================
# PASS 1 : SILENT RECOGNITION
# =========================================================

print(
    "\n========== PASS 1: "
    "Recognising known faces "
    "=========="
)

for filename in os.listdir(
        INPUT_DIR
):

    filepath=os.path.join(
        INPUT_DIR,
        filename
    )

    extension=os.path.splitext(
        filename
    )[1].lower()

    if (
        extension not in IMAGE_EXTENSIONS
        and
        extension not in VIDEO_EXTENSIONS
    ):
        continue

    print(
        f"\nProcessing: "
        f"{filename}"
    )

    detected_names=set()


    # ====================================
    # IMAGE
    # ====================================

    if extension in IMAGE_EXTENSIONS:

        image=cv2.imread(
            filepath
        )

        if image is None:
            continue

        faces=app.get(
            image
        )

        if not faces:
            continue
            
        # Only process the largest face (main person)
        faces = [max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))]

        for face in faces:

            embedding=face.embedding

            face_crop=crop_face(
                image,
                face
            )

            name=match_face(
                embedding
            )

            if name!="Unknown":

                detected_names.add(
                    name
                )

            else:

                unknown_queue.append({
                    'embedding':embedding,
                    'face_crop':face_crop,
                    'bbox':face.bbox,
                    'filepath':filepath,
                    'filename':filename
                })


    # ====================================
    # VIDEO
    # ====================================

    elif extension in VIDEO_EXTENSIONS:

        frames=extract_video_frames(
            filepath
        )

        vote_counter={}

        video_unknowns=[]

        for frame in frames:

            faces=app.get(
                frame
            )

            if not faces:
                continue
                
            # Only process the largest face (main person) in the frame
            faces = [max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))]

            for face in faces:

                embedding=face.embedding

                name=match_face(
                    embedding
                )

                # Known face - vote
                if name!="Unknown":

                    vote_counter[
                        name
                    ]=vote_counter.get(
                        name,
                        0
                    )+1

                    if vote_counter[
                        name
                    ]>=VIDEO_CONFIRM_COUNT:

                        detected_names.add(
                            name
                        )

                # Unknown face - cluster
                else:

                    matched_existing=False

                    for existing in video_unknowns:

                        sim=cosine_similarity(
                            embedding,
                            existing[
                                'embedding'
                            ]
                        )

                        if sim>SIMILARITY_THRESHOLD:

                            existing[
                                'count'
                            ]+=1

                            existing[
                                'face_crop'
                            ]=crop_face(
                                frame,
                                face
                            )

                            existing[
                                'frame'
                            ]=frame.copy()

                            existing[
                                'bbox'
                            ]=face.bbox

                            matched_existing=True

                            break

                    if not matched_existing:

                        video_unknowns.append({
                            'embedding':
                                embedding,
                            'face_crop':
                                crop_face(
                                    frame,
                                    face
                                ),
                            'frame':
                                frame.copy(),
                            'bbox':
                                face.bbox,
                            'count':1
                        })

        # Queue confirmed unknowns
        for unknown in video_unknowns:

            if (
                unknown['count']
                >=
                VIDEO_CONFIRM_COUNT
            ):

                unknown_queue.append({
                    'embedding':
                        unknown['embedding'],
                    'face_crop':
                        unknown['face_crop'],
                    'frame':
                        unknown['frame'],
                    'bbox':
                        unknown['bbox'],
                    'filepath':filepath,
                    'filename':filename
                })


    # ====================================
    # SORT KNOWN FACES
    # ====================================

    if detected_names:

        file_detections[filepath]={
            'filename':filename,
            'names':set(detected_names)
        }

        for person_name in detected_names:

            sort_file(
                filepath,
                filename,
                person_name
            )

        print(
            "  -> "
            +", ".join(
                detected_names
            )
        )

    else:

        print(
            "  -> No known faces"
        )


# =========================================================
# PASS 2 : RESOLVE UNKNOWNS
# =========================================================

if len(unknown_queue)>0:

    print(
        f"\n========== PASS 2: "
        f"{len(unknown_queue)} "
        f"unknown face(s) "
        f"to identify =========="
    )

    while len(unknown_queue)>0:

        entry=unknown_queue.pop(0)

        embedding=entry['embedding']
        face_crop=entry['face_crop']
        filepath=entry['filepath']
        filename=entry['filename']

        # Re-check (may have been identified
        # by a previous iteration)
        name=match_face(embedding)

        if name!="Unknown":

            if filepath not in file_detections:

                file_detections[filepath]={
                    'filename':filename,
                    'names':set()
                }

            file_detections[
                filepath
            ]['names'].add(name)

            sort_file(
                filepath,
                filename,
                name
            )

            print(
                f"\nAuto-matched: "
                f"{filename} -> {name}"
            )

            continue

        # Show face
        if face_crop is not None:

            cv2.imshow(
                "Unknown Face",
                face_crop
            )

            cv2.waitKey(1)

        print(
            f"\nFile: {filename}"
        )

        # Try auto-suggestion
        suggested=suggest_match(
            embedding
        )

        if suggested:

            print(
                f"  AI suggests: "
                f"{suggested}"
            )

        name=input_prefilled(
            "Enter name "
            "(empty to skip): ",
            suggested or ""
        ).strip().strip("[](){}")

        cv2.destroyAllWindows()

        if (
            name==""
            or
            name.lower()=="skip"
        ):
            continue

        # Save to memory
        known_faces.append(
            (
                name,
                embedding
            )
        )

        with open(
            MEMORY_FILE,
            "wb"
        ) as f:

            pickle.dump(
                known_faces,
                f
            )

        print(
            f"Saved {name}"
        )

        # Sort this file
        if filepath not in file_detections:

            file_detections[filepath]={
                'filename':filename,
                'names':set()
            }

        file_detections[
            filepath
        ]['names'].add(name)

        sort_file(
            filepath,
            filename,
            name
        )

        # Re-scan remaining queue
        auto_matched=0

        remaining=[]

        for queued in unknown_queue:

            q_name=match_face(
                queued['embedding']
            )

            if q_name!="Unknown":

                q_filepath=queued[
                    'filepath'
                ]

                q_filename=queued[
                    'filename'
                ]

                if (
                    q_filepath
                    not in
                    file_detections
                ):

                    file_detections[
                        q_filepath
                    ]={
                        'filename':
                            q_filename,
                        'names':set()
                    }

                file_detections[
                    q_filepath
                ]['names'].add(
                    q_name
                )

                sort_file(
                    q_filepath,
                    q_filename,
                    q_name
                )

                auto_matched+=1

            else:

                remaining.append(
                    queued
                )

        unknown_queue[:]=remaining

        if auto_matched>0:

            print(
                f"Auto-matched "
                f"{auto_matched} more "
                f"file(s) as {name}"
            )

else:

    print(
        "\nNo unknown faces found."
    )


print(
    "\nAll done!"
)