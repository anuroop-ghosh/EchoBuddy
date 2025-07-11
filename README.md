**EchoBuddy: Intelligent Lecture Capture and Q&A System**

**Overview**

EchoBuddy is an innovative project designed to transform traditional lecture capture content into an intelligent, queryable knowledge base. Many universities in the UK and US utilize lecture capture software and technology, leading to a significant rise in available content resources. However, a primary problem associated with this is the lower utilization of these resources by students, largely due to the lack of integration of appropriate AI models in the student interface.

This project addresses these limitations by leveraging artificial intelligence to convert raw lecture audio (converting .mp4 or other video formats to .mp3) and scanned documents (like the lecture PDF shared by a professor) into a dynamic, searchable resource. The core idea is to enable users (students, educators, researchers) to ask precise questions about lecture material and receive instant, accurate answers, even when the information is embedded within spoken discourse or scanned documents. This significantly enriches the learning experience and enhances the utility of archived lectures. For demonstration purposes and due to the lack of real-life lecture data, a financial report and a sample voiceover audio file were used as replicas of real-world examples.

**Features**

* Audio Transcription: Converts MP3 lecture recordings into accurate text transcripts using advanced Automatic Speech Recognition (ASR) models.

* PDF Content Extraction: Extracts text from PDF documents, including OCR for scanned images within PDFs, to create a comprehensive text corpus.

* Intelligent Text Chunking: Divides the combined lecture content into smaller, semantically coherent chunks for efficient retrieval.

* Hybrid Retrieval System (RAG): Combines the power of vector embeddings (FAISS) and traditional keyword matching (BM25) to retrieve the most relevant passages for a given query.

* Re-ranking of Retrieved Documents: Utilizes a specialized re-ranker model to refine the initial retrieval results, ensuring higher precision and relevance.

* Gemini-Powered Question Answering: Integrates with the Google Gemini 1.5 Flash model to generate concise and accurate answers grounded only in the retrieved context, mitigating hallucinations.

* Simulated Learning Progress Visualization (Optional): Generates a video demonstrating the processing of document chunks and a simulated learning progress curve.

**How it Works**

The system operates through a multi-stage pipeline:


1. **Audio & PDF Ingestion**:

* MP3 audio files are converted to WAV format.

* PDF documents are processed, with OCR applied to image-based pages.

* The extracted text from both sources is combined.


2. **Text Processing & Chunking**:

* The combined text is split into overlapping, semantically meaningful chunks.


3. **Embedding Generation**:

* Each text chunk is converted into a high-dimensional vector embedding using a Sentence Transformer model (`all-MiniLM-L6-v2`).

* These embeddings are stored in a FAISS index for fast similarity search.


4. **Hybrid Retrieval**:

* When a user submits a query, it's used to perform both a vector similarity search (FAISS) and a keyword-based search (BM25) across the document chunks.

The results from both methods are combined.


5. **Re-ranking**:

* A specialized re-ranker model (`BAAI/bge-reranker-base`) evaluates the relevance of the combined retrieved chunks to the original query, ordering them by pertinence.


6. **Generative Question Answering (RAG)**:

The top-ranked relevant chunks are used as context to augment the user's query.

This augmented prompt is sent to the Gemini 1.5 Flash model, which generates an answer strictly based on the provided context.


**Getting Started**

**Prerequisites**

* Python 3.8+

* `ffmpeg` (for audio conversion and video generation)

* `poppler-utils` (for PDF processing)

* A Google Gemini API Key


**Installation**

1. Clone the repository:

`git clone https://github.com/your-username/EchoBuddy.git
cd EchoBuddy`



2. Install system dependencies:

`sudo apt-get update
sudo apt-get install tesseract-ocr libtesseract-dev poppler-utils ffmpeg -y`



3. Install Python dependencies:

`pip install --upgrade pip
pip install pydub PyPDF2 ffmpeg-python Pillow pytesseract pdf2image
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]
pip install langchain_community sentence-transformers rank_bm25 faiss-cpu`



**Configuration**

Set your Google Gemini API Key. You can do this by:

* Setting it as an environment variable: `export GOOGLE_API_KEY='YOUR_API_KEY'`

* If using Google Colab, adding it as a Colab secret named `GOOGLE_API_KEY`.

**Running the Notebook**

* Upload your MP3 lecture file (e.g., `Chicken Cottage 4.mp3`) and any relevant PDF documents to your working directory (or update the paths in the notebook).

* Open the `EchoBuddy.ipynb` notebook in a Jupyter environment (e.g., Google Colab, Jupyter Lab, Jupyter Notebook).

* Run all cells in the notebook sequentially.

The notebook will guide you through:

* Audio processing and transcription.

* PDF text extraction.

* Building the RAG system.

* Running automated test queries with recall evaluation.

* An optional interactive session where you can ask your own questions.

* Optional video generation to visualize the process.

**Project Structure**

* `EchoBuddy.ipynb`: The main Jupyter Notebook containing all the code and explanations for the project.

* `Chicken Cottage 4.mp3` (example): A sample audio file used for demonstration.

* `financial_report.pdf` (example): A sample PDF file used for demonstration.

* `output.wav`: Intermediate WAV file generated from the MP3.

**Future Enhancements**

* Multi-modal Retrieval: Incorporate image and video analysis directly into the retrieval process.

* Improved Evaluation Metrics: Implement more sophisticated RAG evaluation metrics (e.g., faithfulness, answer relevance).

* User Interface: Develop a web-based or desktop application for a more user-friendly experience.

* Real-time Processing: Explore options for near real-time transcription and question answering during live lectures.

* Support for Diverse File Types: Extend support to other document formats (e.g., DOCX, PPTX).
