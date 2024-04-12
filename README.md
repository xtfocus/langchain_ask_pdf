# Chat with your pdf. 


![](https://github.com/xtfocus/langchain_ask_pdf/blob/master/app.gif)

## Features:

- [x] Open source LLMs: no OpenAI API key needed
- [ ] OOD Detection
- [x] Contextualized query

---

## Configuration:

-  `backend.vectorstores.get_embedding_model`: replace `model_name` with your sentence-transformers of choice
-  `get_llm`: replace the logic to get your chat model (another gguf file, or Ollama, or OpenAI)

---

## Known issues:

- Creating vectorstores taking too long: probaly downloading embedding model for the first time.
- How to delete vectorstores: vectorstore for each session is under `backend/docstores/`.
- How to start over with another PDF: refresh the tab.

---

## TODO:

- [ ] streaming tokens
- [ ] config file for ease of switching models
- [ ] requirements.txt
- [ ] option to browse your own gguf (chat models are better)
- [ ] customized memory/chat history management: keep n-last, clear, delete vectorstores, more [here](https://python.langchain.com/docs/use_cases/chatbots/memory_management/)
- [ ] fusion
- [ ] better encoding for q-a (e.g., DPR)
- [ ] Save sessions' chat history (sqlite + txt)

