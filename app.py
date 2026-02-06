from flask import Flask, render_template, Response, jsonify
from web_stream import generate_frames, control

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/start", methods=["POST"])
def start_camera():
    control["running"] = True
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop_camera():
    control["running"] = False
    return jsonify({"status": "stopped"})

@app.route("/toggle_privacy", methods=["POST"])
def toggle_privacy():
    control["privacy"] = not control["privacy"]
    return jsonify({"privacy": control["privacy"]})

if __name__ == "__main__":
    app.run(debug=True)
