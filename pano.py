import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import shutil
from PIL import Image
import io

def rescale(img, scale=0.5):
    """Rescale image by given scale factor"""
    h, w = img.shape[:2]
    h = int(h * scale)
    w = int(w * scale)
    return cv2.resize(img, (w, h))

def extract_keyframes(video_path, scale=0.5, cutoff=50, ratio_threshold=0.5):
    """Extract keyframes from video based on feature matching"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, "Error: Could not open video file"
    
    # Create ORB detector and matcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    frames = []
    frame_count = 0
    
    # Get first frame
    ret, last = cap.read()
    if not ret:
        return None, "Error: Could not read first frame"
    
    last = rescale(last, scale)
    frames.append(last.copy())
    kp1, des1 = orb.detectAndCompute(last, None)
    
    if des1 is None:
        return None, "Error: Could not detect features in first frame"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get total frame count for progress
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frames += 1
        progress_bar.progress(processed_frames / total_frames)
        
        frame = rescale(frame, scale)
        kp2, des2 = orb.detectAndCompute(frame, None)
        
        if des2 is None:
            prev_frame = frame
            continue
        
        # Match features
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except:
            prev_frame = frame
            continue
        
        # Apply Lowe's ratio test
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good.append(m)
        
        status_text.text(f"Processing frame {processed_frames}/{total_frames} - Good matches: {len(good)}")
        
        # Check if we should save this frame
        if len(good) < cutoff:
            frame_count += 1
            last = frame.copy()
            kp1 = kp2
            des1 = des2
            frames.append(last)
            
        prev_frame = frame
    
    # Add last frame if it's different
    if prev_frame is not None and len(frames) > 0:
        if not np.array_equal(frames[-1], prev_frame):
            frames.append(prev_frame)
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return frames, None

def stitch_images(frames):
    """Stitch frames into panorama"""
    if len(frames) < 2:
        return None, "Error: Need at least 2 frames to create panorama"
    
    # Use OpenCV's built-in stitcher
    stitcher = cv2.Stitcher.create()
    status, stitched = stitcher.stitch(frames)
    
    if status == cv2.Stitcher_OK:
        return stitched, None
    else:
        error_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed"
        }
        return None, f"Stitching failed: {error_messages.get(status, 'Unknown error')}"

def main():
    st.set_page_config(page_title="Video to Panorama", page_icon="🎬", layout="wide")
    
    st.title("🎬 Video to Panorama Converter")
    st.markdown("Upload a video file and convert it to a panoramic image using keyframe extraction and image stitching.")
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    scale = st.sidebar.slider("Scale Factor", 0.1, 1.0, 0.5, 0.1, 
                             help="Resize factor for frames (smaller = faster processing)")
    cutoff = st.sidebar.slider("Keypoint Cutoff", 10, 200, 50, 10,
                              help="Minimum number of matching keypoints to consider frames similar")
    ratio_threshold = st.sidebar.slider("Lowe's Ratio Threshold", 0.1, 0.9, 0.5, 0.1,
                                       help="Threshold for Lowe's ratio test (lower = more strict)")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a video file", 
                                    type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'])
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Display video info
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{duration:.1f}s")
            with col2:
                st.metric("FPS", f"{fps:.1f}")
            with col3:
                st.metric("Frames", frame_count)
            with col4:
                st.metric("Resolution", f"{width}x{height}")
            
            if st.button("🔄 Process Video", type="primary"):
                with st.spinner("Extracting keyframes..."):
                    frames, error = extract_keyframes(tmp_path, scale, cutoff, ratio_threshold)
                
                if error:
                    st.error(error)
                else:
                    st.success(f"Extracted {len(frames)} keyframes")
                    
                    # Show some sample frames
                    if len(frames) > 0:
                        st.subheader("Sample Keyframes")
                        cols = st.columns(min(5, len(frames)))
                        for i, col in enumerate(cols):
                            if i < len(frames):
                                # Convert BGR to RGB for display
                                frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                                col.image(frame_rgb, caption=f"Frame {i+1}", use_column_width=True)
                    
                    # Stitch images
                    with st.spinner("Creating panorama..."):
                        panorama, stitch_error = stitch_images(frames)
                    
                    if stitch_error:
                        st.error(stitch_error)
                        st.info("Try adjusting the parameters or use a video with more overlapping content.")
                    else:
                        st.success("Panorama created successfully!")
                        
                        # Convert BGR to RGB for display
                        panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
                        
                        # Display panorama
                        st.subheader("Generated Panorama")
                        st.image(panorama_rgb, use_column_width=True)
                        
                        # Download button
                        pil_image = Image.fromarray(panorama_rgb)
                        buf = io.BytesIO()
                        pil_image.save(buf, format='PNG')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Download Panorama",
                            data=buf.getvalue(),
                            file_name="panorama.png",
                            mime="image/png"
                        )
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    # Instructions
    with st.expander("How it works"):
        st.markdown("""
        1. **Upload a video**: Choose a video file that shows a scene from different angles
        2. **Keyframe extraction**: The app analyzes the video and extracts frames where the scene changes significantly
        3. **Feature matching**: Uses ORB features and Lowe's ratio test to find corresponding points between frames
        4. **Panorama stitching**: Combines the keyframes into a single panoramic image
        
        **Tips for best results:**
        - Use videos with good overlap between consecutive frames
        - Avoid videos with too much motion blur
        - Videos with distinctive features work better
        - Try adjusting the parameters if the initial result isn't satisfactory
        """)
    
    with st.expander("⚙️ Parameter Guide"):
        st.markdown("""
        - **Scale Factor**: Reduces frame size for faster processing. Smaller values = faster but potentially lower quality
        - **Keypoint Cutoff**: Minimum number of matching features between frames. Higher values = fewer keyframes
        - **Lowe's Ratio Threshold**: Controls feature matching strictness. Lower values = more strict matching
        """)

if __name__ == "__main__":
    main()