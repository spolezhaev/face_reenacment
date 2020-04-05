import urllib.request
import requests
import uuid
from os import environ
from pathlib import Path

from kutana import Plugin, Attachment
from kutana.backends.telegram import Telegram
#from kutana.manager.tg.environment import TGEnvironment, TGAttachmentTemp
#from kutana.manager.vk.environment import VKEnvironment
#from kutana.plugin import Message, Attachment


db = {}

plugin = Plugin(name="Memory")




@plugin.on_start()
async def initiation(kutana):
    pass


async def send_instruction_info(ctx):
    await ctx.reply("Отправьте фотографию Вашего Героя!")

@plugin.on_any_message(user_state="")
async def lol(msg, ctx):
    await send_instruction_info(ctx)

@plugin.on_any_message(user_state="photo:number")
async def _(msg, ctx):
    if not msg.text.isdigit() or int(msg.text) > int(db[msg.sender_id][0]) :
        await ctx.reply("Пожалуйста выберите своего героя")
        return
    r = requests.post('http://localhost:5000/reenact', data={"face_num": msg.text}, files={'image': open(db[msg.sender_id][1], "rb")})

    output_video = Attachment.new(r.content, type="video" if isinstance(ctx.backend, Telegram) else "doc",
                                  file_name="output.mp4")
    await ctx.reply("Ваш результат", attachments=output_video)
    await ctx.set_state(user_state="")

@plugin.on_attachments(["image"])
async def apply_reenacment(message, ctx):
    original_filename = f"originals/{message.attachments[0].file_name}.jpg"
    if isinstance(ctx.backend, Telegram):
        file_id = message.attachments[0].id
        image_content = await ctx.backend._request_file(file_id)
        with open(original_filename, "wb") as f:
            f.write(image_content)
    else:
        image_link = next(filter(lambda x: x['type'] == 'y', message.attachments[0].raw["sizes"]))['url']
        _ = urllib.request.urlretrieve(url=image_link, filename=original_filename)
    r = requests.post('http://localhost:5000/reenact', files={'image': open(original_filename, "rb")})
    if r.status_code == 300:
        await ctx.reply("Выберите вашего Героя", attachments=Attachment.new(r.content))
        await ctx.set_state(user_state="photo:number")
        db[message.sender_id] = (r.headers['faces_num'], original_filename)
    else:
        #with open('plugins/output.gif', 'rb') as f:
        output_video = Attachment.new(r.content, type="video" if isinstance(ctx.backend, Telegram) else "doc", file_name="output.mp4")
        await ctx.reply("Ваш результат", attachments=output_video)
        # folder_name = str(uuid.uuid4())
    # filepath = Path("tmp") / folder_name
    # filepath.mkdir(exist_ok=True, parents=True)
    # original_filename = filepath / "original.ogg"
    #
    # if isinstance(env, TGEnvironment):
    #     try:
    #         file_id = message.raw_update["message"]["voice"]["file_id"]
    #     except:
    #         await send_instruction_info(env)
    #         return
    #
    #     file_tg = await env.manager.request("getFile", file_id=file_id)
    #     file_content = await env.manager.request_file(file_tg.response["file_path"])
    #
    #     with open(original_filename, mode="w+b") as fp:
    #         fp.write(file_content)
    #
    #     voice_path, noise_path = process_audio(original_filename)
    #
    #     with open(voice_path, "rb") as fh:
    #         voice_message = TGAttachmentTemp("voice", fh.read(), {})
    #     with open(noise_path, "rb") as fh:
    #         noise_message = TGAttachmentTemp("voice", fh.read(), {})
    #
    # elif isinstance(env, VKEnvironment):
    #     if not message.attachments or message.attachments[0].type != "audio_message":
    #         await send_instruction_info(env)
    #         return
    #
    #     attachment: Attachment = message.attachments[0]
    #     file_link = attachment.raw_attachment["audio_message"]["link_ogg"]
    #     _ = urllib.request.urlretrieve(url=file_link, filename=original_filename)
    #     voice_path, noise_path = process_audio(original_filename)
    #
    #     with open(voice_path, "rb") as fh:
    #         voice_message = await env.upload_doc(fh.read(), type="audio_message", filename="voice.ogg")
    #     with open(noise_path, "rb") as fh:
    #         noise_message = await env.upload_doc(fh.read(), type="audio_message", filename="noise.ogg")

    # await ctx.reply("Cleaned voice", attachment=voice_message)
    # await ctx.reply("Separated noise", attachment=noise_message)
    # await ctx.reply("Готово, обращайся ещё!")