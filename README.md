# legal_qa_web
A basic Streamlit Website for Legal QA

## Installation

Pull the repo first, add data to folder data (law_nondup copy 14), add .env file

Create venv
```
python -m venv env_name_here
.\env_name_here\Scripts\activate.bat
```

Install requirements
```
pip install -r requirements.txt
```

Run BM25 
```
python bm25.py
```

Run Streamlit
```
streamlit run streamlit.py
```

## Note
- Dùng bản data14 (không dùng 12 nữa) --> DONEDONE


## To-do list

- (Further) restrict the no of relevant doc display
- Change to Vietnamese --> DONE
- Make it more pretty??? Idk 
- Add other features: Feedback to email?, Report error??
- Nên làm thêm cái stream generate
