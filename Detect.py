import os
import shutil
import pickle
import cv2
import numpy as np

from insightface.app import FaceAnalysis


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
# RECOGNIZE FACE
# =========================================================

def recognize_face(
        embedding,
        image=None,
        face_crop=None
):

    global known_faces

    name="Unknown"

    best_similarity=-1

    for saved_name,saved_embedding in known_faces:

        similarity=cosine_similarity(
            embedding,
            saved_embedding
        )

        if similarity>best_similarity:

            best_similarity=similarity

            if similarity>SIMILARITY_THRESHOLD:

                name=saved_name

    if name=="Unknown":

        if face_crop is not None:

            cv2.imshow(
                "Unknown Face",
                face_crop
            )

            cv2.waitKey(1)

        name=input(
            "\nNew face detected.\n"
            "Enter name "
            "(skip to ignore): "
        ).strip()

        cv2.destroyAllWindows()

        name=name.strip("[](){}")

        if (
            name!=""
            and
            name.lower()!="skip"
        ):

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

    return name


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
# PROCESS IMAGE
# =========================================================

def process_image(
        image,
        detected_names
):

    faces=app.get(
        image
    )

    for face in faces:

        embedding=face.embedding

        bbox=face.bbox.astype(
            int
        )

        x1,y1,x2,y2=bbox

        pad=30

        x1=max(
            0,
            x1-pad
        )

        y1=max(
            0,
            y1-pad
        )

        x2=min(
            image.shape[1],
            x2+pad
        )

        y2=min(
            image.shape[0],
            y2+pad
        )

        face_crop=image[
            y1:y2,
            x1:x2
        ]

        name=recognize_face(
            embedding,
            image,
            face_crop
        )

        if (
            name!=""
            and
            name.lower()!="skip"
        ):

            detected_names.add(
                name
            )


# =========================================================
# MAIN LOOP
# =========================================================

for filename in os.listdir(
        INPUT_DIR
):

    filepath=os.path.join(
        INPUT_DIR,
        filename
    )

    print(
        f"\nProcessing: "
        f"{filename}"
    )

    detected_names=set()

    extension=os.path.splitext(
        filename
    )[1].lower()


    # ====================================
    # IMAGE
    # ====================================

    if extension in IMAGE_EXTENSIONS:

        image=cv2.imread(
            filepath
        )

        if image is None:

            continue

        process_image(
            image,
            detected_names
        )


    # ====================================
    # VIDEO
    # ====================================

    elif extension in VIDEO_EXTENSIONS:

        frames=extract_video_frames(
            filepath
        )

        vote_counter={}

        for frame in frames:

            faces=app.get(
                frame
            )

            for face in faces:

                embedding=face.embedding

                name=recognize_face(
                    embedding
                )

                if (
                    name=="Unknown"
                    or
                    name=="skip"
                ):
                    continue

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

                    print(
                        f"Video identified as "
                        f"{name}"
                    )

                    break

            if len(
                detected_names
            )>0:

                break


    else:

        continue


    # ====================================
    # SORT
    # ====================================

    for person_name in detected_names:

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


    if detected_names:

        print(
            "Saved to:",
            ", ".join(
                detected_names
            )
        )

        os.remove(
            filepath
        )

    else:

        print(
            "No faces saved."
        )


print(
    "\nAll done!"
)