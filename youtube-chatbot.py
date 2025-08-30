from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
import os
# from dotenv import load_dotenv
import asyncio 

    
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())    
    
# load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def extract_youtube_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    else:
        return None

# === Streamlit Interface ===
st.title("🎥 YouTube Video Transcript Q&A")
st.markdown("Provide a YouTube URL and ask your question based on the video transcript.")

youtube_url = st.text_input("Enter YouTube Video URL")
user_query = st.text_input("Enter your question")

if st.button("Submit"):
    if not youtube_url or not user_query:
        st.warning("Please enter both the video URL and your question.")
    else:
        st.info("Processing, please wait...")

        video_id = extract_youtube_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL.")
        else:
            try:
                data = YouTubeTranscriptApi().fetch(video_id)
                transcript = "".join(snippet.text for snippet in data.snippets)
            except TranscriptsDisabled:
                st.error("Captions are disabled for this video.")
                st.stop()

            spliter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            chunks = spliter.create_documents([transcript])

            if not chunks:
                st.error("Transcript could not be processed.")
                st.stop()

            embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
            vector_store = FAISS.from_documents(chunks, embedding_model)
            retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 3})

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant. Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                Context:
                {context}

                Question:
                {query}
                """,
                input_variables=['context', 'query']
            )

            def format_doc(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_doc),
                'query': RunnablePassthrough()
            })

            model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')
            parser = StrOutputParser()

            final_chain = parallel_chain | prompt | model | parser

            try:
                result = final_chain.invoke(user_query)
                st.subheader("Answer")
                st.write(result)
            except Exception as e:
                st.error(f"Error during model response: {e}")