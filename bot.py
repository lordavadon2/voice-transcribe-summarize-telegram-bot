import os
import json
import logging
import tempfile
import requests
from logging.handlers import RotatingFileHandler

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.ext import CallbackQueryHandler

from groq import Groq

from dotenv import load_dotenv

load_dotenv()

# --- Настройка системы логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'bot.log',
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=3,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        "summary_model": "llama-3.3-70b-versatile",
        "user_modes": {}
    }
    
    try:
        with open('config.json', 'r') as f:
            return {**default, **json.load(f)}
    except FileNotFoundError:
        return default

CONFIG = load_config()
AUTHORIZED_USERS = load_users()
ADMIN_ID = int(os.environ['ADMIN_ID']) # ID администратора

# --- Groq Client Initialization ---    
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def is_admin(user_id: int) -> bool:
    '''Check if user is admin'''
    return user_id == ADMIN_ID

def get_user_details(user) -> dict:
    """Извлекает данные пользователя с обработкой отсутствующих полей"""
    return {
        'id': user.id,
        'first_name': user.first_name or 'N/A',
        'last_name': user.last_name or 'N/A',
        'username': f"@{user.username}" if user.username else 'N/A'
    }

def get_message_keyboard(user_id: int):
    """Generate keyboard for regular messages based on user permissions"""
    if is_admin(user_id):
        keyboard = [[InlineKeyboardButton("Адмін", callback_data="admin_panel")]]
    else:
        keyboard = [[InlineKeyboardButton("Режим", callback_data="change_mode")]]
    
    return InlineKeyboardMarkup(keyboard)

async def fetch_models(client: Groq):
    """Fetch available models from Groq API"""
    try:
        response = client.models.list()
        return [model.id for model in response.data]
    except Exception as e:
        logger.error(f"Помилка отримання моделей: {str(e)}")
        return [
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "whisper-large-v3",
            "distil-whisper-large-v3-en"
        ]

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    logger.info(f"Отримання команди /start: {update.effective_user.id}")
    user_name = update.effective_user.first_name
    
    try:
        user_id = update.effective_user.id
        reply_markup = get_message_keyboard(user_id)
        
        await update.message.reply_text(
            f"👋 Привіт {user_name}! Я бот для транскрипції та резюме голосових повідомлень.\n\n"
            "Ви можете:\n"
            "1. Переслати мені голосові повідомлення\n"
            "2. Надіслати мені прямі записи голосу\n\n"
            "Я перекодую їх та надам вам транскрицію та резюме!",
            reply_markup=reply_markup
        )
        
        logger.info(f"Надіслано привітальне повідомлення користувачу: {update.effective_user.id}")
    except Exception as e:
        logger.error(f"Помилка в обробнику запуску: {str(e)}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages and voice notes."""
    user = update.effective_user
    user_data = get_user_details(user)

    try:
        if user.id not in AUTHORIZED_USERS:
            logger.warning(
            "Спроба несанкціонованого доступу | Дані: %s",
            user_data
            )
            await update.message.reply_text("⛔ Доступ заборонено")
            return
        
        status_message = await update.message.reply_text("🎵 Обробка голосового повідомлення...")

        logger.info(
            "Обробка голосового повідомлення | Користувач: %s | ID файла: %s",
            user_data['username'],
            update.message.voice.file_id
        )
        
        voice_file = await update.message.voice.get_file()
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
            await voice_file.download_to_drive(temp_file.name)
            temp_path = temp_file.name
        
        logger.info("Початок транскрипції | Користувач: %s", user_data['username'])
        transcription = await transcribe_audio(temp_path, context)
        logger.info(
            "Успішна транскрипція | Користувач: %s | Символів: %d",
            user_data['username'],
            len(transcription)
        )

        user_id = update.effective_user.id
        reply_markup = get_message_keyboard(user_id)
        user_mode = CONFIG.get("user_modes", {}).get(user_id, "both")
        
        if user_mode == "transcription":
            if len(transcription) > 3000:
                await status_message.edit_text("📝 *транскрипція (Частина 1):*", parse_mode='Markdown')
                chunk_size = 4000
                transcription_chunks = [transcription[i:i + chunk_size] for i in range(0, len(transcription), chunk_size)]
                for i, chunk in enumerate(transcription_chunks, 1):
                    await update.message.reply_text(
                        f"*Транскрипція (Частина {i}):*\n{chunk}",
                        parse_mode='Markdown',
                        reply_markup=reply_markup
                    )
            else:
                await status_message.edit_text(
                    "📝 *Транскрипція:*\n"
                    f"{transcription}",
                    parse_mode='Markdown',
                    reply_markup=reply_markup
                )
        else:
            logger.info("Початок резюме | Користувач: %s", user_data['username'])
            summary = await generate_summary(transcription, context)
            logger.info(
                "Успішне резюме | Користувач: %s | Символів: %d",
                user_data['username'],
                len(summary)
            )
            
            if len(transcription) > 3000:
                await status_message.edit_text("📝 *транскрипція (Частина 1):*", parse_mode='Markdown')
                chunk_size = 4000
                transcription_chunks = [transcription[i:i + chunk_size] for i in range(0, len(transcription), chunk_size)]
                for i, chunk in enumerate(transcription_chunks, 1):
                    await update.message.reply_text(
                        f"*Транскрипція (Частина {i}):*\n{chunk}",
                        parse_mode='Markdown',
                        reply_markup=reply_markup
                    )
                
                await update.message.reply_text(
                    "📌 *Резюме:*\n"
                    f"{summary}",
                    parse_mode='Markdown',
                    reply_markup=reply_markup
                )
            else:
                await status_message.edit_text(
                    "📝 *Транскрипція:*\n"
                    f"{transcription}\n\n"
                    "📌 *Резюме:*\n"
                    f"{summary}",
                    parse_mode='Markdown',
                    reply_markup=reply_markup
                )
        
        os.unlink(temp_path)

    except Exception as e:
        logger.error(
            "Помилка при обробці голосового повідомлення | Користувач: %s | Повідомлення: %s",
            user_data['username'],
            str(e),
            exc_info=True
        )

async def transcribe_audio(file_path: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Transcribe audio using Whisper via Groq API."""
    try:
        with open(file_path, "rb") as file:
            logger.info("Запит до API Whisper | Модель: %s", CONFIG["whisper_model"])
            transcription = groq_client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model=CONFIG["whisper_model"],
            )
            return transcription.text.strip()
    except Exception as e:
        logger.error("Ошибка транскрипції: %s", str(e), exc_info=True)
        raise


async def generate_summary(text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Generate a summary using LLama 3 via Groq API."""
    try:
        logger.info("Запит до API Groq | Модель: %s", CONFIG["summary_model"])
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
        logger.error("Ошибка сумаризації: %s", str(e), exc_info=True)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Summary failed: {e}")
        return ""

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages and generate summaries."""
    user = update.effective_user
    user_data = get_user_details(user)

    try:
        if user.id not in AUTHORIZED_USERS:
            logger.warning(
            "Спроба несанкціонованого доступу | Дані: %s",
            user_data
            )
            await update.message.reply_text("⛔ Доступ заборонено")
            return

        status_message = await update.message.reply_text("📝 Генерую резюме...")
        text = update.message.text
        logger.info("Початок сумаризації | Користувач: %s", user_data['username'])
        summary = await generate_summary(text, context)
        logger.info(
                "Успішна резюме | Користувач: %s | Символів: %d",
                user_data['username'],
                len(summary)
            )
        user_id = update.effective_user.id
        reply_markup = get_message_keyboard(user_id)

        await status_message.edit_text(
            "📌 *Резюме:*\n"
            f"{summary}",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    except Exception as e:
        logger.error(
            "Помилка при сумаризації | Користувач: %s | Повідомлення: %s",
            user_data['username'],
            str(e),
            exc_info=True
        )
        user_id = update.effective_user.id
        reply_markup = get_message_keyboard(user_id)
        await update.message.reply_text(
            f"❌ Вибачте, сталася помилка: {str(e)}",
            reply_markup=reply_markup
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log Errors caused by Updates."""
    error = context.error
    user_data = {}
    
    if update.effective_user:
        user_data = get_user_details(update.effective_user)
    
    logger.error(
        "Критична помилка | Користувач: %s | Помилка: %s",
        user_data.get('username', 'N/A'),
        str(error),
        exc_info=error
    )
    
    if update.effective_message:
        logger.debug(
            "Контекст помилки | Чат: %d | Повідомлення: %s",
            update.effective_chat.id,
            update.effective_message.text
        )

# --- Admin Commands ---
async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle admin commands."""
    if not is_admin(update.effective_user.id):
        logger.warning(
            "Спроба несанкціонованого доступу | Дані: %s",
            user_data
            )
        await update.message.reply_text("⛔ Тільки адміністратор може виконати цю команду.")
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
                    await update.message.reply_text(f"✅ Оновлено модель Whisper: {model_name}")
                else:
                    whisper_models = [m for m in available_models if "whisper" in m.lower()]
                    await update.message.reply_text(
                        "❌ Неправильна модель Whisper. Доступні опції:\n" +
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
                    "❌ Неправильна модель. Доступні моделі LLM:\n" +
                    "\n".join([m for m in available_models if "whisper" not in m.lower()])
                )

        elif command == "/add_user":
            user_id = int(args[0])
            if user_id not in AUTHORIZED_USERS:
                AUTHORIZED_USERS.append(user_id)
                save_users()
                await update.message.reply_text(f"✅ Користувача {user_id} додано")

        elif command == "/remove_user":
            user_id = int(args[0])
            if user_id in AUTHORIZED_USERS:
                AUTHORIZED_USERS.remove(user_id)
                save_users()
                await update.message.reply_text(f"❌ Користувача {user_id} видалено")

        elif command == "/list_models":
            models = "\n".join([
                f"🔊 *Моделі Whisper:*\n{', '.join([m for m in available_models if 'whisper' in m.lower()])}\n\n"
                f"🧠 *Моделі LLM:*\n{', '.join([m for m in available_models if 'whisper' not in m.lower()])}"
            ])
            await update.message.reply_text(models, parse_mode='Markdown')

    except IndexError:
        logger.error(
            "Контекст помилки | Чат: %d | Повідомлення: %s",
            update.effective_chat.id,
            update.effective_message.text
        )
        await update.message.reply_text("❌ Не знайдено аргумент")

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
def get_admin_keyboard(auth):
    keyboard = [
        [
            InlineKeyboardButton("Додати користувача", callback_data="add_user"),
            InlineKeyboardButton("Видалити користувача", callback_data="remove_user")
        ],
        [
            InlineKeyboardButton("Встановити модель Whisper", callback_data="set_whisper"),
            InlineKeyboardButton("Встановити модель LLM", callback_data="set_summary")
        ],
        [
            InlineKeyboardButton("Список моделей", callback_data="list_models"),
            *([InlineKeyboardButton("Режим", callback_data="change_mode")] if auth else [])
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show admin panel with inline keyboard."""
    if not is_admin(update.effective_user.id):
        user_data = get_user_details(user)
        logger.warning(
            "Спроба доступу до адмінки | Дані: %s",
            user_data
        )
        await update.message.reply_text("⛔ Потрібні права адміністратора")
        return

    auth=update.effective_user.id not in AUTHORIZED_USERS
    await update.message.reply_text(
        "🔧 Адмінська панель",
        reply_markup=get_admin_keyboard(auth=auth)
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    reply_markup = get_message_keyboard(user_id)
    auth=update.effective_user.id in AUTHORIZED_USERS
    user_data = get_user_details(update.effective_user)

    if not auth or not is_admin(update.effective_user.id):
        logger.warning(
            "Спроба несанкціонованого доступу | Дані: %s",
            user_data
            )
        if update.message:
            await update.message.reply_text("⛔ Доступ заборонено")
        elif update.callback_query:
            await update.callback_query.answer("⛔ Доступ заборонено")
        return

    if query.data == "change_mode":
        keyboard = [
            [
                InlineKeyboardButton("Тільки транскрипція", callback_data="mode_transcription"),
                InlineKeyboardButton("Транскрипція + Резюме", callback_data="mode_both")    
            ],
            [InlineKeyboardButton("« Back", callback_data="back_to_main")]
        ]
        await query.message.edit_text(
            "Оберіть режим обробки:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif query.data == "mode_transcription":
        if "user_modes" not in CONFIG:
            CONFIG["user_modes"] = {}
        CONFIG["user_modes"][user_id] = "transcription"
        save_config()
        await query.message.edit_text(
            "✅ Режим встановлено на: *Тільки транскрипція*\n\nНадішліть мені голосове повідомлення, щоб спробувати!",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )

    elif query.data == "mode_both":
        if "user_modes" not in CONFIG:
            CONFIG["user_modes"] = {}
        CONFIG["user_modes"][user_id] = "both"
        save_config()
        await query.message.edit_text(
            "✅ Режим встановлено на: *Транскрипція + Резюме*\n\nНадішліть мені голосове повідомлення, щоб спробувати!",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )   

    elif query.data == "back_to_main":
        user_id = update.effective_user.id
        reply_markup = get_message_keyboard(user_id)
        await query.message.edit_text(
            "What would you like to do?",
            reply_markup=reply_markup
        )

    elif query.data == "add_user":
        await query.message.edit_text("Будь ласка, надішліть ідентифікатор користувача для додавання за допомогою:\n/add_user <user_id>")
    
    elif query.data == "remove_user":
        await query.message.edit_text("Будь ласка, надішліть ID користувача для видалення за допомогою:\n/remove_user <user_id>")
    
    elif query.data == "set_whisper":
        models = [m for m in context.application.bot_data["available_models"] if 'whisper' in m.lower()]
        keyboard = [[InlineKeyboardButton(m, callback_data=f"select_whisper_{m}")] for m in models]
        keyboard.append([InlineKeyboardButton("« Back", callback_data="back_to_menu")])
        await query.message.edit_text(
            "Виберіть модель Whisper:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif query.data == "set_summary":
        models = [m for m in context.application.bot_data["available_models"] if 'whisper' not in m.lower()]
        keyboard = [[InlineKeyboardButton(m, callback_data=f"select_summary_{m}")] for m in models]
        keyboard.append([InlineKeyboardButton("« Back", callback_data="back_to_menu")])
        await query.message.edit_text(
            "Вибиріть модель для резюме:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif query.data.startswith("select_whisper_"):
        model_name = query.data.replace("select_whisper_", "")
        CONFIG["whisper_model"] = model_name
        save_config()
        await query.message.edit_text(
            f"✅ Whisper модель була оновлена до:\n*{model_name}*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="back_to_menu")]])
        )
    
    elif query.data.startswith("select_summary_"):
        model_name = query.data.replace("select_summary_", "")
        CONFIG["summary_model"] = model_name
        save_config()
        await query.message.edit_text(
            f"✅ Модель для резюме була оновлена до:\n*{model_name}*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("« Back", callback_data="back_to_menu")]])
        )

    elif query.data == "list_models":
        available_models = context.application.bot_data["available_models"]
        models_text = (
            "🔊 *Whisper моделі:*\n" +
            "\n".join([m for m in available_models if 'whisper' in m.lower()]) +
            "\n\n🧠 *LLM моделі:*\n" +
            "\n".join([m for m in available_models if 'whisper' not in m.lower()])
        )
        keyboard = [[InlineKeyboardButton("« Back", callback_data="back_to_menu")]]
        await query.message.edit_text(
            models_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif query.data == "admin_panel" or query.data == "back_to_menu":
        await query.message.edit_text(
            "🔧 Адмінська панель",
            reply_markup=get_admin_keyboard(auth=auth)
        )
    
    return
    
async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /mode command to change processing mode."""
    user_id = str(update.effective_user.id)
    
    keyboard = [
        [
            InlineKeyboardButton("Тільки транскрипція", callback_data="mode_transcription"),
            InlineKeyboardButton("Транскрипція + Резюме", callback_data="mode_both")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    current_mode = CONFIG.get("user_modes", {}).get(user_id, "both")
    mode_text = "Транскрипція + Резюме" if current_mode == "both" else "Тільки транскрипція"
    
    await update.message.reply_text(
        f"Виберіть режим обробки:\n\nПоточний режим: *{mode_text}*",
        parse_mode='Markdown',
        reply_markup=reply_markup
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
    logger.info("Запуск бота...")
    application = (
        Application.builder()
        .token(os.getenv("TELEGRAM_BOT_TOKEN"))
        .post_init(post_init)
        .build()
    )

    setup_admin(application)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("mode", mode_command)) 
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)

    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        logger.info("Бот успішно виконав роботу")
    except Exception as e:
        logger.critical("Помилка запуску бота: %s", str(e), exc_info=True)
        raise
