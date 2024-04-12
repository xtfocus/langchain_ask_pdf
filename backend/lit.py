import os
from time import sleep

import streamlit as st
from vectorstore import (create_history_from_st_messages, digest,
                         generate_response, write_tmp_file)

st.title("🖥️ Ask Your PDF")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


placeholder = st.empty()


def empty():
    placeholder.empty()
    sleep(0.01)


if "submitted" not in st.session_state:
    st.session_state["submitted"] = False


if not st.session_state["submitted"]:
    with placeholder.container():
        st.markdown("#### Upload pdf")
        uploaded_file = st.file_uploader(
            "Upload your PDF and Click Submit & Process button",
            accept_multiple_files=False,
            key="pdf_uploader",
        )

        if uploaded_file:
            tmp_file_path = os.path.join(os.getcwd(), uploaded_file.name)

            assert uploaded_file.name.endswith(".pdf"), "Only supports pdf now"

            write_tmp_file(uploaded_file, tmp_file_path)

            with st.empty():
                submit_button = st.button("Submit & Process", key="process_button")

                if submit_button:
                    st.session_state["file_name"] = uploaded_file.name

                    with st.spinner("Creating vectorstore..."):
                        #            # Add ingestion logic here
                        st.session_state["submitted"] = True

                        st.session_state["vectorstore_path"] = digest(tmp_file_path)

                        st.success("Done")

                    empty()

                    os.remove(tmp_file_path)


if st.session_state.submitted:
    empty()

    st.markdown(f"## Chat with {st.session_state['file_name']} now!")

    if prompt := st.chat_input(
        "Your question"
    ):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt:
        # if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = create_history_from_st_messages(st.session_state.messages)

                response = generate_response(
                    prompt, st.session_state["vectorstore_path"], history
                )
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(
                    message
                )  # Add response to message history
