import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
from demo import make_animation
from skimage import img_as_ubyte
from flask import Flask, jsonify, make_response, send_file
from flask import request, Response
import cv2
import uuid
import codecs
from crop_faces import crop_faces, highlight_faces
import subprocess

app = Flask(__name__)


from demo import load_checkpoints


generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                            checkpoint_path='chkpts/vox-cpk.pth.tar')

driving_video = imageio.mimread('videos/11.mp4', memtest=float('inf'))
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

def process_request(image):
    source_image = resize(image, (256, 256))[..., :3]
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)
    filename = f'output_videos/{uuid.uuid1()}.mp4'
    # img, *imgs = [img_as_ubyte(frame) for frame in predictions]
    # # img.save(fp = filename,format='GIF', append_images=imgs,
    # #      save_all=True, duration=200, loop=0)
    imageio.mimsave(filename, [img_as_ubyte(frame) for frame in predictions])
    #optimize(filename)
    #Если сразу сохранять в гифку, то почему-то размер файла раз в 100 больше
    #subprocess.call(['ffmpeg', "-i", f"{filename}.mp4", "-vf", '"fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"', '-loop', '0', f"{filename}.gif"])
    return filename#f"{filename}.gif"




@app.route('/reenact', methods=["POST"])
def reenact():
    file = request.files.get('image', '')
    filename = f"images/{uuid.uuid1()}.png"
    file.save(filename)
    img = imageio.imread(filename)
    faces_num, faces_folder = crop_faces(img)

    if faces_num == 0:
        return "There is no faces on image", 400

    if faces_num > 1 and 'face_num' not in request.form.keys():
        response = make_response(send_file(highlight_faces(img), as_attachment=True))
        response.headers['faces_num'] = faces_num
        return response, 300

    face_idx = int(request.form["face_num"]) - 1 if 'face_num' in request.form.keys() else 0

    if face_idx >= faces_num:
        return "There is no such face on image", 404

    face_img = imageio.imread(f"images/{faces_folder}/face_{face_idx}.jpg")

    filename = process_request(face_img)
    file_data = codecs.open(filename, 'rb').read()

    return Response(file_data, 206, mimetype='video/mp4', content_type='video/mp4', direct_passthrough=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')