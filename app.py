import streamlit as st
import fitz
from PIL import Image
import io
import base64
import os
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import pytesseract
import inspect

load_dotenv()


# Check for Gemini API key and show warning if missing
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error(
        "Gemini API key not found! Please set the GEMINI_API_KEY environment variable."
    )
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemma-3-27b-it")

# Initialize CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


def embed_image(image_data):
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


def embed_text(text):
    inputs = clip_processor(
        text=text, return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


def process_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    all_docs = []
    all_embeddings = []
    image_data_store = {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    for i, page in enumerate(doc):
        text = page.get_text()
        images = list(page.get_images(full=True))
        text_chunks = []
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])
            for chunk in text_chunks:
                embedding = embed_text(chunk.page_content)
                all_embeddings.append(embedding)
                all_docs.append(chunk)
        # If either text or images exist, extract both modalities for this page
        if text_chunks or images:
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    image_id = f"page_{i}_img_{img_index}"
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    image_data_store[image_id] = img_base64
                    # OCR: extract text from image
                    ocr_text = pytesseract.image_to_string(pil_image)
                    embedding = embed_image(pil_image)
                    all_embeddings.append(embedding)
                    # Use OCR text in page_content for better retrieval
                    image_doc = Document(
                        page_content=f"[Image: {image_id}] OCR: {ocr_text.strip()}",
                        metadata={"page": i, "type": "image", "image_id": image_id},
                    )
                    all_docs.append(image_doc)
                except Exception as e:
                    continue
    doc.close()
    return all_docs, np.array(all_embeddings), image_data_store


def build_vector_store(all_docs, embeddings_array):
    return FAISS.from_embeddings(
        text_embeddings=[
            (doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)
        ],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs],
    )


def retrieve_multimodal(query, vector_store, k=5):
    query_embedding = embed_text(query)
    results = vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)
    return results


def multimodal_pdf_rag_pipeline(query, vector_store, image_data_store):
    context_docs = retrieve_multimodal(query, vector_store, k=5)
    # Always include image docs from the same pages as the retrieved text docs
    frame = inspect.currentframe()
    all_docs = frame.f_back.f_locals.get("all_docs", [])
    text_docs = [doc for doc in context_docs if doc.metadata.get("type") == "text"]
    text_pages = {doc.metadata.get("page") for doc in text_docs}
    # Find image docs from the same pages
    paired_image_docs = [
        doc
        for doc in all_docs
        if doc.metadata.get("type") == "image"
        and doc.metadata.get("page") in text_pages
    ]
    # Remove duplicate image docs if already present
    image_ids_included = {
        doc.metadata.get("image_id")
        for doc in context_docs
        if doc.metadata.get("type") == "image"
    }
    new_image_docs = [
        doc
        for doc in paired_image_docs
        if doc.metadata.get("image_id") not in image_ids_included
    ]
    context_docs += new_image_docs

    gemini_inputs = []
    gemini_inputs.append(query)
    text_docs = [doc for doc in context_docs if doc.metadata.get("type") == "text"]
    if text_docs:
        text_context = "\n\n".join(
            [f"[Page {doc.metadata['page']}]: {doc.page_content}" for doc in text_docs]
        )
        gemini_inputs.append(f"Text excerpts:\n{text_context}\n")
    image_docs = [doc for doc in context_docs if doc.metadata.get("type") == "image"]
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            img_bytes = base64.b64decode(image_data_store[image_id])
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            gemini_inputs.append(pil_image)
    gemini_inputs.append(
        "Please answer the question based on the provided text and images(compulsary)."
    )
    response = model.generate_content(gemini_inputs)
    return context_docs, response.text if hasattr(response, "text") else response


def display_context(context_docs, image_data_store):
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = (
                doc.page_content[:100] + "..."
                if len(doc.page_content) > 100
                else doc.page_content
            )
            st.markdown(f"**Text from page {page}:** {preview}")
        else:
            st.markdown(f"**Image from page {page}:**")
            image_id = doc.metadata.get("image_id")
            if image_id and image_id in image_data_store:
                img_bytes = base64.b64decode(image_data_store[image_id])
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                st.image(pil_image)


st.title("Multimodal RAG (PDF + Images)")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf:
    with st.spinner("Processing PDF and building vector store..."):
        all_docs, embeddings_array, image_data_store = process_pdf(uploaded_pdf)
        # Debug: show how many text and image docs were created
        num_text = sum(1 for doc in all_docs if doc.metadata.get("type") == "text")
        num_image = sum(1 for doc in all_docs if doc.metadata.get("type") == "image")
        st.write(
            f"PDF processed: {len(all_docs)} docs total, {num_text} text, {num_image} image."
        )
        vector_store = build_vector_store(all_docs, embeddings_array)
    st.success("PDF processed and vector store created!")
    query = st.text_input("Enter your question about the PDF:")
    if query:
        with st.spinner("Retrieving and generating answer..."):
            context_docs, answer = multimodal_pdf_rag_pipeline(
                query, vector_store, image_data_store
            )
        st.subheader("Retrieved Context:")
        # Debug output: show type and metadata of each retrieved doc
        st.write("### Debug: Retrieved Document Types and Metadata")
        for i, doc in enumerate(context_docs):
            st.write(
                f"Doc {i+1}: type={doc.metadata.get('type')}, metadata={doc.metadata}"
            )
        display_context(context_docs, image_data_store)
        st.subheader("Gemini Answer:")
        st.write(answer)
