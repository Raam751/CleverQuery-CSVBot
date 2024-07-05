# Large PDF Chat
 This project is the comprehensive implementation of RAG application of PDF and CSV data which have large pages.


The project is implemented using langchai, FAISS and the concepts of LLMs. The code is a slow implementation Because the vector embeddings and the Phi's index takes a lot of time to compute And that is why we have created another notebook for faster implementation, which uses the smart document data frame from Pandas Ai Library which computes the embedding faster so that we can make an inference.


To setup this project, you need to run the following commands:

```bash
pip install -r requirements.txt
```

To run this project, you need to run the following command:

```bash
streamlit run app.py
```
The faster inference code use PANDASAI and GROQ library to compute the embeddings faster. The code is implemented in the app.py under the comments.
