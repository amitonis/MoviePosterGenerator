# app.py
from flask import Flask, render_template, request, send_file
from utils.poster_gen import generate_poster
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching during development

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        description = request.form['description']
        poster_path = generate_poster(description)
        filename = os.path.basename(poster_path)
        return render_template('index.html', image_url=f"/generated/posters/{filename}")
    return render_template('index.html')

@app.route('/generated/posters/<filename>')
def serve_poster(filename):
    return send_file(os.path.join('generated/posters', filename), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)