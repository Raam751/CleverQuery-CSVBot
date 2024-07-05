# CSV Querying Bot system
 This project is the comprehensive implementation of RAG application of PDF and CSV data which have large pages.


The project is implemented using langchai, FAISS and the concepts of LLMs. The code is a slow implementation Because the vector embeddings and the Phi's index takes a lot of time to compute And that is why we have created another notebook for faster implementation, which uses the smart document data frame from Pandas Ai Library which computes the embedding faster so that we can make an inference.


### Here the main target was to run electoral bonds files and make a query system on it.

This repository contains three different approaches to analyze and query electoral bonds data. Each approach offers unique features and methodologies for data processing and querying.

## Table of Contents

1. [Approaches](#approaches)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Approaches

### 1. RAG (Retrieval-Augmented Generation) with PDF

This approach uses RAG to process PDF documents containing electoral bonds data. It creates embeddings and indexes for efficient querying using cosine similarity.

### 2. PDF to CSV with Gemini SQL Generation

This method converts PDF data to CSV format and uses Google's Gemini AI to generate SQL queries for data retrieval from the CSV files.

### 3. PandasAI SmartDataframe

This approach utilizes PandasAI's SmartDataframe, which internally generates queries and executes them using a SQL engine to produce results.

## Features

### RAG Approach
- PDF document processing
- Text embedding generation
- Indexing for fast retrieval
- Cosine similarity-based querying
- Efficient for large documents

### Gemini SQL Generation
- PDF to CSV conversion
- AI-powered SQL query generation
- Flexible querying of structured data
- Integration with Google's Gemini AI

### PandasAI SmartDataframe
- Automated query generation
- Built-in SQL engine execution
- User-friendly interface
- Efficient for tabular data analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CleverQuery-CSVBot.git
```
2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  #MacOs or Unix/linux
```
```bash
# On Windows, use venv\Scripts\activate
```

3. Install the required packages:
```bash 
pip install -r requirements.txt
```

4. Set up API keys in env file (make one, if not):
- For the Gemini approach, set your Google API key as an environment variable:
  ```
  export GOOGLE_API_KEY='your_api_key_here'
  ```
- Also, setup Groq API key:
    ```bash
    GROQ_API_KEY=<your api key>
    ```

## Usage

1. Navigate to the Different Apps directory.
2. Run the Streamlit app (depending on your use case):
```bash 
streamlit run using_gemini.py
```
```bash
streamlit run using_RAG.py
```
```bash
streamlit run using_pandasai.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For more detailed information on each approach, please refer to the README files in their respective directories.

If you encounter any issues or have questions, please open an issue in the GitHub repository.
