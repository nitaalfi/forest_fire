import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np

from PIL import Image

from torchvision import transforms
from torchvision.models import mobilenet_v2
from torchvision.models import MobileNet_V2_Weights

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cpu")

# =====================================================
# CLASS NAMES
# =====================================================
class_names = ['fire', 'nofire']

# =====================================================
# CLAHE TRANSFORM
# =====================================================
class CLAHETransform:

    def __init__(self):

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )

    def __call__(self, img):

        img = np.array(img)

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2LAB
        )

        l, a, b = cv2.split(img)

        l = self.clahe.apply(l)

        img = cv2.merge((l,a,b))

        img = cv2.cvtColor(
            img,
            cv2.COLOR_LAB2RGB
        )

        return Image.fromarray(img)

# =====================================================
# TRANSFORM
# =====================================================
transform = transforms.Compose([

    transforms.Resize((224,224)),

    CLAHETransform(),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():

    model = mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT
    )

    model.classifier[1] = nn.Linear(
        model.last_channel,
        2
    )

    model.load_state_dict(
        torch.load(
            "mobilenet_clahe_wildfire.pth",
            map_location=device
        )
    )

    model.eval()

    return model

model = load_model()

# =====================================================
# TITLE
# =====================================================
st.title("🔥 Wildfire Detection System")

st.write(
    "Deteksi Kebakaran Hutan "
    "Menggunakan MobileNetV2 + CLAHE"
)

# =====================================================
# FILE UPLOADER
# =====================================================
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

# =====================================================
# PREDICTION
# =====================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # PREPROCESS
    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    # PREDICTION
    with torch.no_grad():

        outputs = model(input_tensor)

        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs,1)

    prediction = class_names[pred.item()]

    confidence = confidence.item() * 100

    # RESULT
    st.subheader("Prediction Result")

    if prediction == "fire":

        st.error(
            f"🔥 FIRE "
            f"({confidence:.2f}%)"
        )

    else:

        st.success(
            f"✅ NO FIRE "
            f"({confidence:.2f}%)"
        )