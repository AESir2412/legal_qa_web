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
python bm25.py
streamlit run streamlit.py
```

## Note
- Dùng bản data14 (không dùng 12 nữa) --> DONE


## To-do list

- (Further) restrict the no of relevant doc display --> Done-ish? (để tạm threshold bừa, còn cả th lọc lại vì score cao chưa chắc chuẩn nma dẹp mẹ đi :> )
- Change to Vietnamese --> DONE
- Make it more pretty??? Idk 
- Add other features: Feedback to email?, Report error?? --> cô bảo k cần (low priority)
- Nên làm thêm cái stream generate
- BM25 elastic search --> Done (BM25s nhanh gấp 2-3 lần elk)
- highlight text in law

