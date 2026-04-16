import streamlit as st
import requests
import time

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Poster Generator", page_icon="🎨", layout="wide")

st.title("🎨 AI Poster Generation System")
st.markdown("Multi-agent poster generation powered by LangGraph")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_iterations = st.slider("Max Iterations", 1, 5, 3)
    st.markdown("---")
    st.info("💡 Tip: Provide detailed descriptions for better results")

# Main content
tab1, tab2, tab3 = st.tabs(["Generate Poster", "History", "System Status"])

with tab1:
    # Input section
    user_input = st.text_area(
        "Describe your poster",
        placeholder="Example: Create a seafood noodle poster with warm tones",
        height=100
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)

    # Generation section
    if generate_btn:
        if not user_input:
            st.error("Please enter a poster description!")
        else:
            with st.spinner("Generating Poster..."):
                # Start generation
                try:
                    response = requests.post(
                        f"{BASE_URL}/api/v1/poster/generate",
                        json={
                            "user_input": user_input,
                            "max_iterations": max_iterations
                        },
                        timeout=10
                    )

                    if response.status_code == 202:
                        data = response.json()
                        task_id = data["task_id"]

                        st.success(f"✅ Task created! Task ID: `{task_id}`")

                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Poll for status
                        max_wait = 100  # 5 minutes
                        start_time = time.time()

                        while time.time() - start_time < max_wait:
                            try:
                                status_response = requests.get(
                                    f"{BASE_URL}/api/v1/poster/status/{task_id}",
                                    timeout=5
                                )

                                if status_response.status_code == 200:
                                    status_data = status_response.json()
                                    progress = status_data["progress"]
                                    current_step = status_data.get("current_step", "Processing...")

                                    progress_bar.progress(int(progress) / 100)
                                    status_text.text(f"Status: {status_data['status']} | {current_step}")

                                    if status_data["status"] == "completed":
                                        st.success("🎉 Poster generated successfully!")

                                        # Get result
                                        result_response = requests.get(
                                            f"{BASE_URL}/api/v1/poster/result/{task_id}",
                                            timeout=10
                                        )

                                        if result_response.status_code == 200:
                                            result = result_response.json()

                                            # Display poster
                                            poster_url = f"{BASE_URL}{result['poster_url']}"
                                            st.image(poster_url, caption="Generated Poster", use_container_width=True)

                                            # iterations = result.get("iteration_count", 0)

                                            # Download button
                                            img_response = requests.get(poster_url, timeout=10)
                                            st.download_button(
                                                label="📥 Download Poster",
                                                data=img_response.content,
                                                file_name=f"poster_{task_id}.png",
                                                mime="image/png"
                                            )

                                            # Show metrics
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                                            with col2:
                                                st.metric("Iterations", max_iterations)
                                            with col3:
                                                st.metric("Status", "Completed")

                                        break

                                    elif status_data["status"] == "failed":
                                        st.error(f"❌ Generation failed: {status_data.get('error')}")
                                        break

                            except requests.exceptions.RequestException as e:
                                st.warning(f"Connection error: {e}")

                            time.sleep(3)

                        else:
                            st.warning("⏱️ Processing timeout, please check history later")

                    else:
                        st.error(f"Request failed: {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Cannot connect to backend: {e}")
                    st.info("Make sure FastAPI server is running on http://localhost:8000")

with tab2:
    st.header("📜 History")

    if st.button("🔄 Refresh"):
        st.rerun()

    # Get task list
    try:
        list_response = requests.get(f"{BASE_URL}/api/v1/poster/list?limit=10", timeout=5)

        if list_response.status_code == 200:
            tasks = list_response.json()

            if tasks:
                for task in tasks:
                    with st.expander(f"Task {task['task_id'][:8]}... - {task['status']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text(f"Status: {task['status']}")
                            st.text(f"Progress: {task['progress']:.1f}%")
                            st.text(f"Created: {task['created_at']}")
                        with col2:
                            if task['status'] == 'completed':
                                if st.button("View Poster", key=task['task_id']):
                                    result_response = requests.get(
                                        f"{BASE_URL}/api/v1/poster/result/{task['task_id']}",
                                        timeout=5
                                    )
                                    if result_response.status_code == 200:
                                        result = result_response.json()
                                        poster_url = f"{BASE_URL}{result['poster_url']}"
                                        st.image(poster_url)
            else:
                st.info("No history yet")
        else:
            st.error("Cannot retrieve history")

    except requests.exceptions.RequestException as e:
        st.error(f"Cannot connect to backend: {e}")

with tab3:
    st.header("🏥 System Status")

    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)

        if health_response.status_code == 200:
            health_data = health_response.json()

            if health_data['status'] == 'healthy':
                st.success(f"System Status: {health_data['status']}")
            else:
                st.warning(f"System Status: {health_data['status']}")

            st.subheader("Service Status")
            services = health_data.get('services', {})

            if services:
                cols = st.columns(len(services))
                for idx, (service, status) in enumerate(services.items()):
                    with cols[idx]:
                        if status == "healthy":
                            st.success(f"✅ {service}")
                        else:
                            st.error(f"❌ {service}\n{status}")
        else:
            st.error("Cannot connect to backend")

    except requests.exceptions.RequestException as e:
        st.error(f"Cannot connect to backend: {e}")
        st.info("Make sure FastAPI server is running on http://localhost:8000")