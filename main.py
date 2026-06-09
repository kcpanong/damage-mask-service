import os
import cv2
import boto3
import json
import requests
import tempfile
import firebase_admin
import numpy as np
import onnxruntime as ort

from PIL import Image

from firebase_admin import (
    credentials,
    firestore
)

# ==========================================================
# 1. CONFIG
# ==========================================================

SERVICE_ACCOUNT_FILE = (
    "serviceAccountKey.json"
)

MODEL_PATH = (
    "crack_seg_cnn.onnx"
)

IDENTITY_POOL_ID = (
    "ap-southeast-2:e70f96d9-6860-4f80-9bf8-082d2661b665"
)

REGION = (
    "ap-southeast-2"
)

BUCKET_NAME = (
    "my-angular-test-bucket-12345"
)

IMG_SIZE = 416

TEMP_DOWNLOAD_FOLDER = (
    "temp_downloads"
)

TEMP_MASK_FOLDER = (
    "temp_masks"
)

# ==========================================================
# 2. FIREBASE INIT
# ==========================================================

print("\n========== FIREBASE DEBUG ==========\n")

print(
    "serviceAccountKey exists:",
    os.path.exists(
        SERVICE_ACCOUNT_FILE
    )
)

if os.path.exists(
    SERVICE_ACCOUNT_FILE
):

    print(
        "File size:",
        os.path.getsize(
            SERVICE_ACCOUNT_FILE
        )
    )

    with open(
        SERVICE_ACCOUNT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    print(
        "\nFirst 300 chars:\n"
    )

    print(
        text[:300]
    )

    print(
        "\nContains private key:",
        "PRIVATE KEY" in text
    )

print(
    "\n==============================\n"
)

if not firebase_admin._apps:

    firebase_json = os.environ.get(
        "FIREBASE_SERVICE_ACCOUNT"
    )

    cred = credentials.Certificate(
        json.loads(firebase_json)
    )

    firebase_admin.initialize_app(
        cred
    )

db = firestore.client()

# ==========================================================
# 3. TEMP FOLDERS
# ==========================================================

os.makedirs(
    TEMP_DOWNLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    TEMP_MASK_FOLDER,
    exist_ok=True
)

# ==========================================================
# 4. LOAD MODEL
# ==========================================================

print("\nLoading ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("ONNX model loaded.")

# ==========================================================
# 5. GET AWS TEMP CREDS
# ==========================================================

print(
    "\nGetting Cognito credentials..."
)

identity_client = boto3.client(
    "cognito-identity",
    region_name=REGION
)

identity = identity_client.get_id(
    IdentityPoolId=IDENTITY_POOL_ID
)

identity_id = identity[
    "IdentityId"
]

creds_response = (
    identity_client.get_credentials_for_identity(
        IdentityId=identity_id
    )
)

creds = creds_response[
    "Credentials"
]

print(
    "Temporary credentials acquired."
)

s3 = boto3.client(
    "s3",
    region_name=REGION,
    aws_access_key_id=creds[
        "AccessKeyId"
    ],
    aws_secret_access_key=creds[
        "SecretKey"
    ],
    aws_session_token=creds[
        "SessionToken"
    ]
)

# ==========================================================
# 6. FIRESTORE QUERY
# ==========================================================

def get_unprocessed_images():

    docs = (
        db.collection("images")
        .stream()
    )

    results = []

    for doc in docs:

        data = doc.to_dict()

        image_type = data.get(
            "type"
        )

        if image_type not in (
            "original",
            "resized"
        ):
            continue

        if data.get(
            "retrievedForProcessing",
            False
        ):
            continue

        if not data.get(
            "storageUrl"
        ):
            continue

        results.append({

            "doc_ref":
                doc.reference,

            "sessionId":
                data.get(
                    "sessionId"
                ),

            "original_id":
                data.get(
                    "original_id"
                ),

            "filename":
                data.get(
                    "filename"
                ),

            "type":
                image_type,

            "storageUrl":
                data.get(
                    "storageUrl"
                )
        })

    return results

# ==========================================================
# 7. DOWNLOAD
# ==========================================================

def sanitize_filename(
    filename
):

    bad_chars = (
        ':',
        '/',
        '\\',
        '*',
        '?',
        '"',
        '<',
        '>',
        '|'
    )

    for char in bad_chars:

        filename = (
            filename.replace(
                char,
                "_"
            )
        )

    return filename

def download_image(
    url,
    output_path
):

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    with open(
        output_path,
        "wb"
    ) as f:

        f.write(
            response.content
        )

# ==========================================================
# 8. SEGMENT
# ==========================================================

def create_mask(
    image_path,
    output_mask_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_np = np.array(
        image
    ).astype(
        np.float32
    )

    image_np = (
        image_np / 255.0
    )

    image_np = (
        image_np - 0.5
    ) / 0.5

    image_np = np.transpose(
        image_np,
        (2, 0, 1)
    )

    image_np = np.expand_dims(
        image_np,
        axis=0
    )

    outputs = session.run(
        None,
        {
            input_name:
                image_np.astype(
                    np.float32
                )
        }
    )

    prob_map = outputs[0]

    prob_map = np.squeeze(
        prob_map
    )

    mask = (
        prob_map >= 0.5
    ).astype(
        np.uint8
    ) * 255

    cv2.imwrite(
        output_mask_path,
        mask
    )

# ==========================================================
# 9. S3 UPLOAD
# ==========================================================

def upload_mask(
    mask_path,
    session_id,
    image_id
):

    s3_key = (
        f"masks/"
        f"{session_id}/"
        f"{image_id}_mask.png"
    )

    s3.upload_file(
        mask_path,
        BUCKET_NAME,
        s3_key
    )

    url = (
        f"https://{BUCKET_NAME}"
        f".s3.{REGION}.amazonaws.com/"
        f"{s3_key}"
    )

    return (
        s3_key,
        url
    )

# ==========================================================
# 10. UPDATE FIRESTORE
# ==========================================================

def mark_processed(
    doc_ref,
    mask_url
):

    doc_ref.update({

        "maskS3Url":
            mask_url,

        "retrievedForProcessing":
            True
    })

# ==========================================================
# 11. PROCESS PIPELINE
# ==========================================================

def process_all_images():

    images = (
        get_unprocessed_images()
    )

    print(
        f"\nFound "
        f"{len(images)} "
        f"images to process.\n"
    )

    processed_count = 0

    for index, image in enumerate(
        images,
        start=1
    ):

        print(
            f"\n[{index}/{len(images)}]"
        )

        try:

            safe_filename = (
                sanitize_filename(
                    image["filename"]
                )
            )

            local_image = os.path.join(
                TEMP_DOWNLOAD_FOLDER,
                safe_filename
            )

            image_id = (
                f"{image['type']}_"
                f"{image['original_id']}"
            )

            local_mask = os.path.join(
                TEMP_MASK_FOLDER,
                image_id + "_mask.png"
            )

            print(
                "Downloading..."
            )

            download_image(
                image["storageUrl"],
                local_image
            )

            print(
                "Running segmentation..."
            )

            create_mask(
                local_image,
                local_mask
            )

            print(
                "Uploading mask..."
            )

            (
                _,
                mask_url
            ) = upload_mask(

                local_mask,

                image[
                    "sessionId"
                ],

                image_id
            )

            print(
                "Updating Firestore..."
            )

            mark_processed(

                image[
                    "doc_ref"
                ],

                mask_url
            )

            processed_count += 1

            print(
                "SUCCESS"
            )

        except Exception as e:

            print(
                f"FAILED: {e}"
            )

    result = {

        "processed":
            processed_count,

        "total":
            len(images)

    }

    print(
        "\nFinished."
    )

    print(result)

    return result


# ==========================================================
# 12. LOCAL EXECUTION
# ==========================================================

if __name__ == "__main__":

    process_all_images()