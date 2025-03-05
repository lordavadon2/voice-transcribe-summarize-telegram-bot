import os
import json
import tempfile
import requests

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.ext import CallbackQueryHandler

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Configuration Loading and Saving ---
def load_users():
    try:
        with open('authorized_users.json', 'r') as f:
            return json.load(f).get("users", [])
    except FileNotFoundError:
        return [123456789]  # Default user

def load_config():
    default = {
        "whisper_model": "distil-whisper-large-v3-en",
        "summary_model": "llama-3.3-70b-versatile"
    }
    try:
        with open('config.json', 'r') as f:
            return {**default, **json.load(f)}
    except FileNotFoundError:
        return default

CONFIG = load_config()
AUTHORIZED_USERS = load_users()
ADMIN_ID = ... # ID администратора

# --- Groq Client Initialization ---    
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def is_admin(user_id: int) -> bool:
    '''Check if user is admin'''
    return user_id == ADMIN_ID

async def fetch_models(client: Groq):
    """Fetch available models from Groq API"""
    try:
        response = client.models.list()
        return [model.id for model in response.data]
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return [
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "whisper-large-v3",
            "distil-whisper-large-v3-en"
        ]

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    print(f"Received start command from user: {update.effective_user.id}")
    user_name = update.effective_user.first_name
    try:
        await update.message.reply_text(
            f"👋 Hi {user_name}! I'm a Voice Note Summarizer Bot.\n\n"
            "You can:\n"
            "1. Forward me voice messages\n"
            "2. Send me direct voice recordings\n\n"
            "I'll transcribe them and provide you with both the transcription and a summary!"
        )
        print(f"Sent welcome message to user: {update.effective_user.id}")
    except Exception as e:
        print(f"Error in start handler: {str(e)}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages and voice notes."""
    try:
        if update.effective_user.id not in AUTHORIZED_USERS:
            await update.message.reply_text("⛔ Sorry, you are not authorized to use this bot. Contact @administrator.")
            return

        status_message = await update.message.reply_text("🎵 Processing your voice note...")
        voice_file = await update.message.voice.get_file()

        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
            await voice_file.download_to_drive(temp_file.name)
            temp_path = temp_file.name

        transcription = await transcribe_audio(temp_path, context)
        summary = await generate_summary(transcription, context)

        if len(transcription) > 3000:
            await status_message.edit_text("📝 *Transcription (Part 1):*", parse_mode='Markdown')
            chunk_size = 4000
            transcription_chunks = [transcription[i:i + chunk_size] for i in range(0, len(transcription), chunk_size)]

            for i, chunk in enumerate(transcription_chunks, 1):
                await update.message.reply_text(
                    f"*Transcription (Part {i}):*\n{chunk}",
                    parse_mode='Markdown'
                )

            await update.message.reply_text(
                "📌 *Summary:*\n"
                f"{summary}",
                parse_mode='Markdown'
            )
        else:
            await status_message.edit_text(
                "📝 *Transcription:*\n"
                f"{transcription}\n\n"
                "📌 *Summary:*\n"
                f"{summary}",
                parse_mode='Markdown'
            )

        os.unlink(temp_path)

    except Exception as e:
        await update.message.reply_text(f"❌ Sorry, an error occurred: {str(e)}")

async def transcribe_audio(file_path: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Transcribe audio using Whisper via Groq API."""
    try:
        with open(file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model=CONFIG["whisper_model"],
            )
            return transcription.text.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Transcription failed: {e}")
        return ""

async def generate_summary(text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Generate a summary using LLama 3 via Groq API."""
    try:
        completion = groq_client.chat.completions.create(
            model=CONFIG["summary_model"],
            messages=[
                {"role": "system", "content": "Generate a concise summary of the following text:"},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Summary error: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Summary failed: {e}")
        return ""

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages and generate summaries."""
    try:
        if update.effective_user.id not in AUTHORIZED_USERS:
            await update.message.reply_text("⛔ Sorry, you are not authorized to use this bot. Contact @administartor.")
            return

        status_message = await update.message.reply_text("📝 Generating summary...")
        text = update.message.text
        summary = await generate_summary(text, context)
        await status_message.edit_text(
            "📌 *Summary:*\n"
            f"{summary}",
            parse_mode='Markdown'
        )

    except Exception as e:
        await update.message.reply_text(f"❌ Sorry, an error occurred: {str(e)}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log Errors caused by Updates."""
    print(f"Update {update} caused error {context.error}")

# --- Admin Commands ---
async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle admin commands."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Only the administrator can remote this bot.")
        return

    command = update.effective_message.text.split()[0]
    args = context.args
    available_models = context.application.bot_data["available_models"]

    try:
        if command == "/set_whisper_model":
            model_name = " ".join(args)
            if any(m for m in available_models if "whisper" in m.lower()):
                if model_name in available_models:
                    CONFIG["whisper_model"] = model_name
                    save_config()
                    await update.message.reply_text(f"✅ Whisper model updated: {model_name}")
                else:
                    whisper_models = [m for m in available_models if "whisper" in m.lower()]
                    await update.message.reply_text(
                        "❌ Invalid Whisper model. Available options:\n" +
                        "\n".join(whisper_models)
                    )

        elif command == "/set_summary_model":
            model_name = " ".join(args)
            if model_name in available_models:
                CONFIG["summary_model"] = model_name
                save_config()
                await update.message.reply_text(f"✅ Summary model updated: {model_name}")
            else:
                await update.message.reply_text(
                    "❌ Invalid model. Available LLM models:\n" +
                    "\n".join([m for m in available_models if "whisper" not in m.lower()])
                )

        elif command == "/add_user":
            user_id = int(args[0])
            if user_id not in AUTHORIZED_USERS:
                AUTHORIZED_USERS.append(user_id)
                save_users()
                await update.message.reply_text(f"✅ User {user_id} added")

        elif command == "/remove_user":
            user_id = int(args[0])
            if user_id in AUTHORIZED_USERS:
                AUTHORIZED_USERS.remove(user_id)
                save_users()
                await update.message.reply_text(f"❌ User {user_id} removed")

        elif command == "/list_models":
            models = "\n".join([
                f"🔊 *Whisper Models:*\n{', '.join([m for m in available_models if 'whisper' in m.lower()])}\n\n"
                f"🧠 *LLM Models:*\n{', '.join([m for m in available_models if 'whisper' not in m.lower()])}"
            ])
            await update.message.reply_text(models, parse_mode='Markdown')

    except IndexError:
        await update.message.reply_text("❌ Missing arguments!")

def save_users():
    """Save authorized users to file."""
    with open('authorized_users.json', 'w') as f:
        json.dump({"users": AUTHORIZED_USERS}, f)

def save_config():
    """Save configuration to file."""
    with open('config.json', 'w') as f:
        json.dump(CONFIG, f)

async def post_init(application: Application):
    models = await fetch_models(groq_client)
    application.bot_data["available_models"] = models

# Добавь функцию для создания админской клавиатуры
def get_admin_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("Add User", callback_data="add_user"),
            InlineKeyboardButton("Remove User", callback_data="remove_user")
        ],
        [
            InlineKeyboardButton("Set Whisper Model", callback_data="set_whisper"),
            InlineKeyboardButton("Set Summary Model", callback_data="set_summary")
        ],
        [
            InlineKeyboardButton("List Models", callback_data="list_models")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show admin panel with inline keyboard."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Only the administrator can access this panel.")
        return
    
    await update.message.reply_text(
        "🔧 Admin Control Panel",
        reply_markup=get_admin_keyboard()
    )

# async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if not is_admin(update.effective_user.id):
        await query.message.edit_text("⛔ Access denied")
        return

    if query.data == "add_user":
        await query.message.edit_text("Please send the user ID to add using:\n/add_user <user_id>")
    
    elif query.data == "remove_user":
        await query.message.edit_text("Please send the user ID to remove using:\n/remove_user <user_id>")
    
    elif query.data == "set_whisper":
        models = [m for m in context.application.bot_data["available_models"] if 'whisper' in m.lower()]
        await query.message.edit_text(
            "Available Whisper models:\n" + "\n".join(models) +
            "\n\nUse: /set_whisper_model <model_name>"
        )
    
    elif query.data == "set_summary":
        models = [m for m in context.application.bot_data["available_models"] if 'whisper' not in m.lower()]
        await query.message.edit_text(
            "Available Summary models:\n" + "\n".join(models) +
            "\n\nUse: /set_summary_model <model_name>"
        )
    
    elif query.data == "list_models":
        available_models = context.application.bot_data["available_models"]
        models = "\n".join([
            f"🔊 *Whisper Models:*\n{', '.join([m for m in available_models if 'whisper' in m.lower()])}\n\n"
            f"🧠 *LLM Models:*\n{', '.join([m for m in available_models if 'whisper' not in m.lower()])}"
        ])
        await query.message.edit_text(models, parse_mode='Markdown')

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if not is_admin(update.effective_user.id):
        await query.message.edit_text("⛔ Access denied")
        return

    if query.data == "add_user":
        await query.message.edit_text("Please send the user ID to add using:\n/add_user <user_id>")
    
    elif query.data == "remove_user":
        await query.message.edit_text("Please send the user ID to remove using:\n/remove_user <user_id>")
    
    elif query.data == "set_whisper":
        models = [m for m in context.application.bot_data["available_models"] if 'whisper' in m.lower()]
        keyboard = [[InlineKeyboardButton(m, callback_data=f"select_whisper_{m}")] for m in models]
        keyboard.append([InlineKeyboardButton("« Back", callback_data="back_to_menu")])
        await query.message.edit_text(
            "Select Whisper model:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif query.data == "set_summary":
        models = [m for m in context.application.bot_data["available_models"] if 'whisper' not in m.lower()]
        keyboard = [[InlineKeyboardButton(m, callback_data=f"select_summary_{m}")] for m in models]
        keyboard.append([InlineKeyboardButton("« Back", callback_data="back_to_menu")])
        await query.message.edit_text(
            "Select Summary model:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif query.data.startswith("select_whisper_"):
        model_name = query.data.replace("select_whisper_", "")
        CONFIG["whisper_model"] = model_name
        save_config()
        await query.message.edit_text(
            f"✅ Whisper model has been updated to:\n*{model_name}*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="back_to_menu")]])
        )
    
    elif query.data.startswith("select_summary_"):
        model_name = query.data.replace("select_summary_", "")
        CONFIG["summary_model"] = model_name
        save_config()
        await query.message.edit_text(
            f"✅ Summary model has been updated to:\n*{model_name}*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="back_to_menu")]])
        )
    
    elif query.data == "back_to_menu":
        await query.message.edit_text(
            "🔧 Admin Control Panel",
            reply_markup=get_admin_keyboard()
        )
    
    elif query.data == "list_models":
        available_models = context.application.bot_data["available_models"]
        models_text = (
            "🔊 *Whisper Models:*\n" +
            "\n".join([m for m in available_models if 'whisper' in m.lower()]) +
            "\n\n🧠 *LLM Models:*\n" +
            "\n".join([m for m in available_models if 'whisper' not in m.lower()])
        )
        keyboard = [[InlineKeyboardButton("« Back", callback_data="back_to_menu")]]
        await query.message.edit_text(
            models_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

def setup_admin(application: Application):
    """Set up admin command handlers."""
    application.add_handler(CommandHandler("admin", admin_panel))
    application.add_handler(CommandHandler("add_user", admin_command))
    application.add_handler(CommandHandler("remove_user", admin_command))
    application.add_handler(CommandHandler("set_whisper_model", admin_command))
    application.add_handler(CommandHandler("set_summary_model", admin_command))
    application.add_handler(CommandHandler("list_models", admin_command))
    application.add_handler(CallbackQueryHandler(button_callback))

# --- Main ---
if __name__ == '__main__':
    application = (
        Application.builder()
        .token(os.getenv("TELEGRAM_BOT_TOKEN"))
        .post_init(post_init)
        .build()
    )

    setup_admin(application)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)

    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
