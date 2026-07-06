import os
import shutil
import pickle
import base64
import time
import cv2
import numpy as np
import requests

from dotenv import load_dotenv
from insightface.app import FaceAnalysis


load_dotenv()


# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = "photos"
OUTPUT_DIR = "sorted_photos"
MEMORY_FILE = "face_memory.pkl"

SIMILARITY_THRESHOLD = 0.45
VIDEO_CONFIRM_COUNT = 4

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
)

GEMINI_MODEL = "gemini-2.0-flash"

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    ""
)

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Seconds between API calls
API_COOLDOWN = 4

last_api_call = 0

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

    if not os.path.exists(
            destination
    ):

        shutil.copy(
            filepath,
            destination
        )


# =========================================================
# IDENTIFY CELEBRITY (API)
# =========================================================

CELEB_PROMPT = (
    "Identify this person. "
    "Reply with ONLY their "
    "full name in UPPERCASE. "
    "If you cannot identify "
    "them, reply with "
    "exactly 'Unknown'."
)


def identify_via_gemini(image_b64):

    url = (
        "https://generativelanguage"
        ".googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}"
        ":generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [{
            "parts": [
                {
                    "text": CELEB_PROMPT
                },
                {
                    "inline_data": {
                        "mime_type":
                            "image/jpeg",
                        "data":
                            image_b64
                    }
                }
            ]
        }]
    }

    response = requests.post(
        url,
        json=payload,
        timeout=15
    )

    data = response.json()

    # Check for API errors
    if 'error' in data:

        msg = data[
            'error'
        ].get('message', '')

        print(
            f"  [Gemini: {msg[:80]}]"
        )

        return None

    # Check for blocked response
    if (
        'candidates' not in data
        or
        len(data['candidates']) == 0
    ):

        block_reason = data.get(
            'promptFeedback', {}
        ).get(
            'blockReason',
            'unknown reason'
        )

        print(
            f"  [Gemini blocked: "
            f"{block_reason}]"
        )

        return None

    # Check finish reason
    finish = data[
        'candidates'
    ][0].get(
        'finishReason',
        ''
    )

    if finish == 'SAFETY':

        print(
            "  [Gemini blocked by "
            "safety filter]"
        )

        return None

    name = data[
        'candidates'
    ][0][
        'content'
    ][
        'parts'
    ][0][
        'text'
    ].strip()

    if (
        name.lower() == "unknown"
        or
        name == ""
    ):
        return None

    return name


def identify_via_groq(image_b64):

    url = (
        "https://api.groq.com"
        "/openai/v1/"
        "chat/completions"
    )

    headers = {
        "Authorization":
            f"Bearer {GROQ_API_KEY}",
        "Content-Type":
            "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": CELEB_PROMPT
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url":
                            "data:image/jpeg;"
                            "base64,"
                            + image_b64
                    }
                }
            ]
        }],
        "max_tokens": 100
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=15
    )

    data = response.json()

    # Check for API errors
    if 'error' in data:

        msg = data[
            'error'
        ].get('message', '')

        print(
            f"  [Groq: {msg[:80]}]"
        )

        return None

    name = data[
        'choices'
    ][0][
        'message'
    ][
        'content'
    ].strip()

    if (
        name.lower() == "unknown"
        or
        name == ""
    ):
        return None

    return name


def identify_celebrity(face_crop):

    global last_api_call

    if (
        not GEMINI_API_KEY
        and
        not GROQ_API_KEY
    ):
        return None

    try:

        # Rate limiting
        elapsed = time.time() - last_api_call

        if elapsed < API_COOLDOWN:

            time.sleep(
                API_COOLDOWN - elapsed
            )

        # Encode face crop
        _, buffer = cv2.imencode(
            '.jpg',
            face_crop
        )

        image_b64 = base64.b64encode(
            buffer
        ).decode()

        last_api_call = time.time()

        # Try Gemini first
        if GEMINI_API_KEY:

            result = identify_via_gemini(
                image_b64
            )

            if result:
                return result

        # Fallback to Groq
        if GROQ_API_KEY:

            result = identify_via_groq(
                image_b64
            )

            if result:
                return result

        return None

    except Exception as e:

        print(
            f"  [API error: {e}]"
        )

        return None


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

        # Try Gemini auto-identification
        suggested=identify_celebrity(
            face_crop
        )

        if suggested:

            print(
                f"Gemini suggests: "
                f"{suggested}"
            )

            confirm=input(
                "Confirm? "
                "(y/yes/name/skip): "
            ).strip()

            cv2.destroyAllWindows()

            confirm=confirm.strip(
                "[](){}"
            )

            if (
                confirm.lower()=="y"
                or
                confirm.lower()=="yes"
            ):

                name=suggested

            elif (
                confirm==""
                or
                confirm.lower()=="skip"
            ):

                continue

            else:

                name=confirm

        else:

            if suggested is None and GEMINI_API_KEY:

                print(
                    "Could not auto-identify."
                )

            name=input(
                "Enter name "
                "(skip to ignore): "
            ).strip()

            cv2.destroyAllWindows()

            name=name.strip("[](){}")

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


# =========================================================
# CLEANUP : DELETE SORTED ORIGINALS
# =========================================================

print(
    "\n========== Cleanup =========="
)

for filepath in file_detections:

    if os.path.exists(filepath):

        os.remove(filepath)

        print(
            f"Removed: "
            f"{file_detections[filepath]['filename']}"
        )


print(
    "\nAll done!"
)