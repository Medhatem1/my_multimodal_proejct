import streamlit as st
import requests
from PIL import Image
import os

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Multimodal Audiovisual Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Multimodal Audiovisual Search Engine")
st.markdown("Search your image catalog using **Text**, **Image-to-Image**, or **Hybrid Search**.")

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Search Settings")
search_mode = st.sidebar.radio("Search Mode", ["Text Search", "Image Search", "Hybrid Search"])
top_k = st.sidebar.slider("Number of Results (Top K)", min_value=1, max_value=20, value=6)
min_score = st.sidebar.slider("Minimum Confidence Threshold (%)", min_value=0, max_value=100, value=30, step=5)

st.sidebar.info(f"Results scoring below **{min_score}%** confidence will be filtered out.")

# ---------------------------------------------------------
# HELPER TO DISPLAY RESULTS GRID
# ---------------------------------------------------------
def display_results(results):
    if not results:
        st.warning("⚠️ No matching results found above the selected confidence threshold. Try lowering the threshold slider.")
        return

    st.subheader(f"Top {len(results)} Matches Found")
    
    cols = st.columns(3)
    for idx, item in enumerate(results):
        col = cols[idx % 3]
        file_path = item.get("file_path", "")
        score = item.get("score", 0.0)
        raw_score = item.get("raw_score", 0.0)
        
        with col:
            if os.path.exists(file_path):
                img = Image.open(file_path)
                st.image(img, use_container_width=True)
            else:
                filename = os.path.basename(file_path)
                st.image(f"{BACKEND_URL}/data/{filename}", use_container_width=True)
            
            st.caption(f"**Filename:** {os.path.basename(file_path)}")
            st.write(f"**Match Confidence:** {score}%")
            st.progress(int(score) / 100.0)
            st.caption(f"*(Raw Cosine Score: {raw_score:.3f})*")
            st.markdown("---")

# ---------------------------------------------------------
# SEARCH MODES
# ---------------------------------------------------------
if search_mode == "Text Search":
    st.header("🔤 Search by Text")
    query_text = st.text_input("Enter search query (e.g., 'a cat sitting on a couch', 'red sports car'):")
    
    if st.button("Search", type="primary"):
        if not query_text.strip():
            st.error("Please enter a query string.")
        else:
            with st.spinner("Searching vector index..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/api/search/text",
                        data={
                            "query": query_text, 
                            "top_k": top_k,
                            "min_score": min_score
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        display_results(data.get("results", []))
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend server: {e}")

elif search_mode == "Image Search":
    st.header("🖼️ Search by Image")
    uploaded_file = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Query Image", width=250)
        
        if st.button("Search Similar Images", type="primary"):
            with st.spinner("Searching vector index..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {
                        "top_k": top_k,
                        "min_score": min_score
                    }
                    response = requests.post(f"{BACKEND_URL}/api/search/image", files=files, data=data)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        display_results(res_data.get("results", []))
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend server: {e}")

elif search_mode == "Hybrid Search":
    st.header("🔀 Hybrid Search (Image + Text)")
    uploaded_file = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png", "webp"])
    text_query = st.text_input("Enter refining text description:")
    alpha = st.slider("Text Weight vs. Image Weight (Alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    if uploaded_file is not None and text_query.strip():
        if st.button("Run Hybrid Search", type="primary"):
            with st.spinner("Combining visual and text vectors..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {
                        "text_query": text_query,
                        "alpha": alpha,
                        "top_k": top_k,
                        "min_score": min_score
                    }
                    response = requests.post(f"{BACKEND_URL}/api/search/hybrid", files=files, data=data)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        display_results(res_data.get("results", []))
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend server: {e}")