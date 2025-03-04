# Legal QA Web

A Streamlit-based web application for Legal Question Answering (Legal QA).

## Installation

### 1. Clone the Repository
To get started, clone the repository and navigate into the project directory:

```sh
git clone https://github.com/AESir2412/legal_qa_web.git
cd legal_qa_web
```

### 2. Set Up Environment Variables
Create a `.env` file by following the structure provided in the `.env-example` template.

### 3. Install Dependencies
Set up your virtual environment (change the `venv_name` to your own, add it to `.gitignore` afterwards), and ensure you have all required dependencies installed:

```sh
python -m venv venv_name
.\venv_name\Scripts\activate.bat

pip install -r requirements.txt
```

### 4. Prepare Data
Rename the folder `data-example` to `data`. The file `example_law_data.json` inside it is only a small sample of our dataset.

### 5. Run the BM25 Retrieval Model
Before launching the web application, run the BM25 retrieval model:

```sh
python bm25.py
```

### 6. Start the Streamlit Web Application
Launch the Streamlit application using:

```sh
streamlit run streamlit.py
```

### 7. Access the Web Interface
Once the application is running, you can access it via your web browser at:

```
http://localhost:8501/ 
OR
http://localhost:8502/
```

## Important Notes
- The dataset used in this project is based on a research paper that has not been publicly released. Therefore, only a sample data file is provided for demonstration purposes.
- For the best performance, ensure your queries align with the topics covered in the provided sample dataset.

---

For any issues or further inquiries, please refer to the project's GitHub repository or contact the maintainers. 🚀
