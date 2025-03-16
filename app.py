import json, os
from src.document_rag import *
from src.flow_analyzer import *
from src.answer_evaluation import *
from src.face_monitoring_inference import *
from flask import Flask, request, Response
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
app.config['UPLOAD_IMAGE_FOLDER'] = 'store/images'
app.config['UPLOAD_AUDIO_FOLDER'] = 'store/audios'
app.config['UPLOAD_CV_FOLDER'] = 'store/cvs'
CORS(app)

# Ensure upload folders exist
for folder in [app.config['UPLOAD_IMAGE_FOLDER'], app.config['UPLOAD_AUDIO_FOLDER'], app.config['UPLOAD_CV_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/api/face_detection', methods=['POST'])
def api_face_detection():
    try:
        username = request.form['username']
        image_file = request.files['image_file']

        save_path = os.path.join(app.config['UPLOAD_IMAGE_FOLDER'], secure_filename(image_file.filename))
        image_file.save(save_path)

        head_pose_text, det_username = face_image_inference(
            username,
            save_path
        )

        return Response(
            response=json.dumps({
                "Head Pose": head_pose_text,
                "Username": det_username
            }),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        return Response(
            response=json.dumps({
                "message": "Face detection failed",
                "error": str(e)
            }),
            status=400,
            mimetype="application/json"
        )

@app.route('/api/face_monitoring', methods=['POST'])
def api_face_monitoring():
    try:
        username = request.form['username']
        response = face_analysis(username)
        return Response(
            response=json.dumps(response),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        return Response(
            response=json.dumps({
                "message": "Face monitoring failed",
                "error": str(e)
            }),
            status=400,
            mimetype="application/json"
        )

    try:
        audio_file = request.files['audio_file']
        save_path = os.path.join(app.config['UPLOAD_AUDIO_FOLDER'], secure_filename(audio_file.filename))
        audio_file.save(save_path)

        response = flowAnalyzerPipeline(save_path)

        return Response(
            response=json.dumps(response),
            status=200,
            mimetype="application/json"
        )
    except Exception as e:
        return Response(
            response=json.dumps({
                "message": "Flow analysis failed",
                "error": str(e)
            }),
            status=400,
            mimetype="application/json"
        )

@app.route('/api/answer_evaluation', methods=['POST'])
def api_answer_evaluation():
    try:
        question = request.form['question']
        correct_answer = request.form['correct_answer']
        user_answer = request.form['user_answer']

        response = inference_answer_evaluation(question, correct_answer, user_answer)

        return Response(
            response=json.dumps({
                "Score": response
            }),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        return Response(
            response=json.dumps({
                "message": "Answer evaluation failed",
                "error": str(e)
            }),
            status=400,
            mimetype="application/json"
        )

@app.route('/api/document_rag', methods=['POST'])
def api_document_rag():
    try:
        cv = request.files['cv']
        cv_path = os.path.join(app.config['UPLOAD_CV_FOLDER'], secure_filename(cv.filename))
        cv.save(cv_path)

        response = retrieve_documents(cv_path)
        response = json.loads(response)
        return Response(
            response=json.dumps({
                "JDs": response
            }),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        return Response(
            response=json.dumps({
                "message": "Document RAG failed",
                "error": str(e)
            }),
            status=400,
            mimetype="application/json"
        )

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5002
    )