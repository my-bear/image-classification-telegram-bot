import os
from aiogram import F, Router, flags
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext

from bot.states import Gen
from bot import kb, text, utils


router = Router()

@router.message(Command("start"))
async def start_handler(msg: Message):
    await msg.answer(text.greet.format(name = msg.from_user.full_name), reply_markup = kb.menu)

@router.message(F.text == "Меню")
@router.message(F.text == "Выйти в меню")
@router.message(F.text == "◀️ Выйти в меню")
async def menu(msg: Message):
    await msg.answer(text.menu, reply_markup = kb.menu)


#gpt-3.5-turbo
@router.callback_query(F.data == "generate_text")
async def input_text_prompt(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.text_prompt)
    await clbck.message.edit_text(text.gen_text)
    await clbck.message.answer(text.gen_exit, reply_markup = kb.exit_kb)

@router.message(Gen.text_prompt)
@flags.chat_action("typing")
async def generate_text(msg: Message, state: FSMContext):
    prompt = msg.text
    mesg = await msg.answer(text.gen_wait)
    res = await utils.generate_text(prompt)
    if not res:
        return await mesg.edit_text(text.gen_error, reply_markup = kb.iexit_kb)
    await mesg.edit_text(res[0] + text.text_watermark, disable_web_page_preview = True)


#dalle
@router.callback_query(F.data == "generate_image")
async def input_image_prompt(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.img_prompt)
    await clbck.message.edit_text(text.gen_image)
    await clbck.message.answer(text.gen_exit, reply_markup = kb.exit_kb)

@router.message(Gen.img_prompt)
@flags.chat_action("upload_photo")
async def generate_image(msg: Message, state: FSMContext):
    prompt = msg.text
    mesg = await msg.answer(text.gen_wait)
    img_res = await utils.generate_image(prompt)
    if len(img_res) == 0:
        return await mesg.edit_text(text.gen_error, reply_markup = kb.iexit_kb)
    await mesg.delete()
    await mesg.answer_photo(photo=img_res[0], caption=text.img_watermark)


#image predict
@router.callback_query(F.data == "predict_image_num")
async def input_image_prompt(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.num_predict)
    await clbck.message.edit_text(text.num_predict_state)
    await clbck.message.answer(text.gen_exit, reply_markup = kb.exit_kb)

@router.message(Gen.num_predict)
@flags.chat_action("typing")
async def predict_image(msg: Message, state: FSMContext):
    file_id = msg.photo[-1].file_id
    mesg = await msg.answer(text.gen_wait)
    path_save = os.path.abspath(os.path.join(os.getcwd(),"./data/forward/%s.png")) % file_id
    await msg.bot.download(file=file_id, destination = path_save)
    img_res = utils.predict_image(file_id)
    if img_res == None:
        return await mesg.edit_text(text.num_error, reply_markup = kb.iexit_kb)
    await msg.answer(text.num_ansver + img_res)

