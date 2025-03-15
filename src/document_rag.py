from PyPDF2 import PdfReader
import yaml, os, json, pymongo
from bson.objectid import ObjectId
from llama_index.core.prompts import (
                                ChatMessage,
                                MessageRole,
                                ChatPromptTemplate,
                                )
# from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.llms.groq import Groq
from llama_index.core import Settings

with open('secrets.yaml') as f:
    secrets = yaml.load(f, Loader=yaml.FullLoader)

os.environ["GROQ_API_KEY"] = secrets['GROQ_API_KEY']
os.environ["MONGO_DB_URI"] = secrets['MONGO_DB_URI']
os.environ["VOYAGE_API_KEY"] = secrets['VOYAGE_API_KEY']

completion_llm = Groq(
                    model="llama3-70b-8192", 
                    api_key=os.environ["GROQ_API_KEY"],
                    temperature=0.0
                    )

# embed_model = VoyageEmbedding(
#                             model_name="voyage-2", 
#                             voyage_api_key=os.environ["VOYAGE_API_KEY"]
#                             )
embed_model = HuggingFaceEmbedding(
                                    model_name="Alibaba-NLP/gte-base-en-v1.5",
                                    trust_remote_code=True,
                                    device="cpu"
                                    )

Settings.embed_model = embed_model
Settings.llm = completion_llm

try:
    client = pymongo.MongoClient(os.environ["MONGO_DB_URI"])
    db = client['Elearning']
    jds_collection = db['jds']

except Exception as e:
    print(e)

extraction_fields_cv = """
Personal Information:
Name
Phone number

Education:
Degree obtained
Field of study

Work Experience:
Job title 
Job Domain (Machine Learning Engineer, Data Scientist, Frontend Developer, Backend Developer, DevOps...)
Company name
Employment dates (start and end)
Job description/tasks
Achievements or responsibilities

Skills:
Technical skills (e.g., programming languages, software proficiency)
Soft skills (e.g., communication, teamwork)
Industry-specific skills

Certifications:
Certification name
Issuing organization
Issue date
Expiration date (if applicable)

Languages:
Proficiency level in spoken and written languages

Volunteer Experience:
Organization name
Role/title
Dates of service
Responsibilities or achievements

Additional Information:
Publications
Professional affiliations

Capabilities:
technical_capabilities (based on the details (focus on the tools , tech person handled and has experience in) provide a description of what this person is capable of in technical perspective, what seems to be the strongest skills and what would this person handle the best)
non_technical_capabilities (based on the details (focus on the nature of the projects / education / experiences this person involved ) provide a description of what this person is capable of in non technical perspective, what seems to be the strongest skills and what would this person handle the best)

Use CamelCase to for naming the keys"""

extraction_fields_jd = """
job_title: This field refers to the title of the job position, such as "Software Engineer," "Data Scientist," or "Project Manager.
job_type: Onsite / Remote / Hybrid
job_description: This field provides a brief overview of the job role and its responsibilities. It describes what the job entails and what the company expects from the employee in that role.
key_responsibilities: This field outlines the primary tasks and duties that the employee will be responsible for in the job role. It typically includes specific responsibilities related to the job title.
required_skills: Technical skills (e.g., programming languages, software proficiency, frameworks, tools, etc.)
preferred_education_qualifications: This field specifies the educational background or qualifications preferred by the employer. It may include specific degrees, certifications, or coursework that the employer values.
preferred_experience: This field indicates the level of experience preferred by the employer for the job role. It may specify the number of years of experience required in a similar role or industry or as Intern, Junior, Senior, Expert, etc.
technical_capabilities: This field refers to the technical skills and abilities required based on the intensity of the job. It may include proficiency in programming languages, software tools, or technical knowledge relevant to the job role.
non_technical_capabilities: This field encompasses the non-technical skills and qualities desired in a candidate based on the intensity of the job. It may include communication skills, problem-solving abilities, teamwork, adaptability, and other soft skills necessary for success in the job role.
"""

JD2PERSONA_PRMT_TMPL = """
You are an skilled assitant who can extract provided content from Job Description. 
Based on the Job Description `Context` Please provide information in `extraction_fields`
After extraction unstructured data, and your task is to parse it into JSON format.

context: {context}
extraction_fields: {extraction_fields_jd}

Return only the Expected JSON format. Do not provide any other text.
"""

CV2PERSONA_PRMT_TMPL = """
You are an skilled assitant who can extract provided content from CV / Resume. 
Based on the CV / Resume `Context` Please provide information in `extraction_fields`
After extraction unstructured data, and your task is to parse it into JSON format.

context: {context}
extraction_fields: {extraction_fields_cv}

Return only the Expected JSON format. Do not provide any other text.
"""

jd2persona_gen_template = ChatPromptTemplate(
                                        message_templates=[
                                                        ChatMessage(
                                                                    role=MessageRole.SYSTEM, 
                                                                    content=JD2PERSONA_PRMT_TMPL
                                                                    )
                                                        ]
                                        )

cv2persona_gen_template = ChatPromptTemplate(
                                        message_templates=[
                                                        ChatMessage(
                                                                    role=MessageRole.SYSTEM, 
                                                                    content=CV2PERSONA_PRMT_TMPL
                                                                    )
                                                        ]
                                        )

def processing_with_llama3(
                            context, 
                            doc_type="cv"
                            ):
    if doc_type == "jd":
        fmt_messages = jd2persona_gen_template.format_messages(
                                                            extraction_fields_jd=extraction_fields_jd,
                                                            context=context
                                                            )
    else:
        fmt_messages = cv2persona_gen_template.format_messages(
                                                            extraction_fields_cv=extraction_fields_cv,
                                                            context=context
                                                            )
    chat_response = completion_llm.chat(fmt_messages)
    raw_output = chat_response.message.content
    return raw_output

def pdf2text(uploaded_file):
    if uploaded_file.endswith('.pdf'):
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            full_text += page_text
        return full_text
    
    elif uploaded_file.endswith('.txt'):
        with open(uploaded_file, 'r') as f:
            full_text = f.read()
        return full_text
    
    else:
        assert False, "File format not supported"

# def post_process_output(raw_output):
#     raw_output = raw_output.replace("```", "").replace("json", "")
#     json_output = json.loads(raw_output)
#     return json_output

def post_process_output(raw_output):
    index_start = raw_output.find("{")  
    index_end = raw_output.rfind("}") + 1
    raw_output = raw_output[index_start:index_end]
    json_output = json.loads(raw_output)
    return json_output

def process_jd_pipeline(pdf_file):
    while True:
        try:
            text_from_file = pdf2text(pdf_file)
            raw_output = processing_with_llama3(
                                            text_from_file, 
                                            doc_type="jd"
                                            )
            json_output = post_process_output(raw_output)
            return json_output
                
        except Exception as e:
            print("Error: ", e)

def process_cv_pipeline(pdf_file):
    while True:
        try:
            text_from_file = pdf2text(pdf_file)
            raw_output = processing_with_llama3(
                                            text_from_file, 
                                            doc_type="cv"
                                            )
            json_output = post_process_output(raw_output)
            return json_output
                
        except Exception as e:
            print("Error: ", e)

def jd_to_persona_pipeline(
                        jd_dir = "data/jobs/JDs",
                        persona_dir = "data/jobs/PERSONAs"
                        ):
    for jd_file in os.listdir(jd_dir):
        persona_file_path = os.path.join(persona_dir, jd_file).replace(".pdf", ".json").replace(".txt", ".json")
        if not os.path.exists(persona_file_path):
            jd_file_path = os.path.join(jd_dir, jd_file)
            json_output = process_jd_pipeline(jd_file_path)
            with open(persona_file_path, 'w') as f:
                json.dump(json_output, f, indent=4)
            print(f"Persona file saved at: {persona_file_path}")

def anomaly_detection(
                        pdf_file,
                        anomaly_detection_prompt = """
                        Use the provided `context` to identify its a CV, JD or Anomaly. Provide output in below format:

                        context: {context}

                        CV / JD -> NonAnomalyFile
                        Anomaly -> AnomalyFile

                        Return only the 'NonAnomalyFile' or 'AnomalyFile' string.
                        """):
    text_from_file = pdf2text(pdf_file)
    fmt_messages = anomaly_detection_prompt.format(context=text_from_file)
    chat_response = completion_llm.complete(fmt_messages)
    raw_output = str(chat_response)
    return raw_output

def build_document_index(persona_dir = "data/jobs/PERSONAs"):
    jd_to_persona_pipeline(persona_dir=persona_dir)

    documents = []
    for persona_file in os.listdir(persona_dir):
        with open(os.path.join(persona_dir, persona_file), 'r') as f:
            persona = json.load(f)
        persona_doc = Document(text=str(persona))
        persona_doc.metadata['file_path'] = os.path.join(persona_dir, persona_file)
        documents.append(persona_doc)

    index = VectorStoreIndex.from_documents(documents)
    vector_retriever = index.as_retriever(similarity_top_k=5)
    return vector_retriever

def retrieve_documents(cv_path):
    anomaly_flag = anomaly_detection(cv_path).strip()
    if ('NonAnomalyFile' in anomaly_flag) or (anomaly_flag == "NonAnomalyFile"):
        vector_retriever = build_document_index()
        cv_persona = process_cv_pipeline(cv_path)
        results = vector_retriever.retrieve(str(cv_persona))
        jds = [result.metadata['file_path'].replace('\\', '/') for result in results]
        db_response = {
                    "CV": cv_path,
                    "JDs": jds
                    }
        jds_collection.insert_one(db_response)
        return jds

    elif ('AnomalyFile' in anomaly_flag) or (anomaly_flag == "AnomalyFile"):
        return "Anomaly CV Detected"

    else:
        return "Anomaly Detection Function Failed"