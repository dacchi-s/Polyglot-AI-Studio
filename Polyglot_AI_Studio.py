import tkinter as tk
from tkinter import ttk, messagebox
import threading
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import os
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import sys
import difflib
import re
from abc import ABC, abstractmethod
import json
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file for API keys and settings
load_dotenv()

class SettingsManager:
    """
    Manages loading and saving of user settings, translation/proofread history, and autosave content.
    Stores data in ~/.ai_translation_studio directory.
    """
    def __init__(self, config_dir: Path = Path.home() / '.ai_translation_studio'):
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_dir / 'config.json'
        self.history_path = self.config_dir / 'history.json'
        self.autosave_path = self.config_dir / 'autosave.json'
        self.settings = self.load_settings()
        self.history = self.load_history()
    
    def load_settings(self) -> dict:
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'theme_mode': 'dark',
            'default_provider': 'OpenAI',
            'last_used_models': {
                'OpenAI': 'gpt-4.1-mini-2025-04-14',
                'Anthropic': 'claude-3-5-haiku-latest',
                'Google': 'gemini-2.5-flash'
            },
            'source_lang': 'Auto Detect',
            'target_lang': 'Japanese',
            'window_geometry': '1450x900',
            'proofread_style': 'Standard'
        }
    
    def save_settings(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def load_history(self) -> list:
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return []
    
    def save_history(self, history: list):
        try:
            history = history[-1000:]
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def save_autosave(self, content: dict):
        try:
            with open(self.autosave_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def load_autosave(self) -> Optional[dict]:
        if self.autosave_path.exists():
            try:
                with open(self.autosave_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

class APIError(Exception):
    """Custom exception for API errors."""
    pass

class LLMProvider(ABC):
    """
    Abstract base class for Large Language Model (LLM) API providers.
    Each provider must implement the call() method.
    """
    def __init__(self, client: any, model_name: str):
        if client is None:
            raise ValueError("API client is not configured.")
        self.client = client
        self.model_name = model_name

    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str) -> str:
        pass

class OpenAIProvider(LLMProvider):
    """
    Handles API calls to OpenAI's chat models.
    """
    def call(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            max_tokens=2500
        )
        return response.choices[0].message.content.strip()

class AnthropicProvider(LLMProvider):
    """
    Handles API calls to Anthropic's Claude models.
    """
    def call(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            temperature=0.4,
            max_tokens=2500
        )
        return response.content[0].text.strip()

class GoogleProvider(LLMProvider):
    """
    Handles API calls to Google's Gemini models.
    """
    def call(self, system_prompt: str, user_prompt: str) -> str:
        model = genai.GenerativeModel(self.model_name)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                max_output_tokens=2500
            )
        )
        return response.text.strip()

class ModernTranslationApp:
    """
    Main application class for the AI Translation & Writing Studio.
    Handles UI setup, user interaction, and API integration for translation and proofreading.
    """
    # Language, model, color, and prompt definitions for the app
    LANGUAGES = {
        "Auto Detect": "auto", "Japanese": "Japanese", "English": "English", "Chinese": "Chinese",
        "Korean": "Korean", "Spanish": "Spanish", "French": "French", "German": "German",
        "Italian": "Italian", "Portuguese": "Portuguese", "Russian": "Russian",
        "Arabic": "Arabic", "Hindi": "Hindi"
    }
    MODELS = {
        "OpenAI": {
            "gpt-4.5-preview-2025-02-27": "GPT-4.5 Preview (Highest Performance - $75/$150)",
            "gpt-4.1-2025-04-14": "GPT-4.1 (High Performance - $2/$8)",
            "gpt-4.1-mini-2025-04-14": "GPT-4.1 Mini (Balanced - $0.4/$1.6)",
            "gpt-4.1-nano-2025-04-14": "GPT-4.1 Nano (Fastest - $0.1/$0.4)",
            "gpt-4o-2024-08-06": "GPT-4o (Traditional High Performance - $2.5/$10)",
            "gpt-4o-mini-2024-07-18": "GPT-4o Mini (Traditional Fast - $0.15/$0.6)"
        },
        "Anthropic": {
            "claude-opus-4-20250514": "Claude Opus 4 (Highest Performance - $15/$75)",
            "claude-sonnet-4-20250514": "Claude Sonnet 4 (Balanced - $3/$15)",
            "claude-3-7-sonnet-latest": "Claude Sonnet 3.7 (High Performance - $3/$15)",
            "claude-3-5-haiku-latest": "Claude Haiku 3.5 (Fast - $0.8/$4)"
        },
        "Google": {
            "gemini-2.5-pro": "Gemini 2.5 Pro (Highest Performance - Multimodal - $1.25/$10)",
            "gemini-2.5-flash": "Gemini 2.5 Flash (Balanced - Cost-Effective - $0.3/$2.5)",
            "gemini-2.5-flash-lite-preview-06-17": "Gemini 2.5 Flash-Lite (Cheapest - Fastest - $0.1/$0.4)",
            "gemini-2.0-flash": "Gemini 2.0 Flash (Next Generation - Real-time - $0.1/$0.4)",
            "gemini-2.0-flash-lite": "Gemini 2.0 Flash-Lite (Low Latency - $0.075/$0.3)"
        }
    }
    COLORS_LIGHT = {
        'primary': '#2563eb', 'primary_hover': '#1d4ed8', 'secondary': '#f9fafb',
        'secondary_hover': '#f3f4f6', 'background': '#ffffff', 'surface': '#f8fafc', 
        'border': '#e5e7eb', 'text': '#1f2937', 'text_secondary': '#6b7280', 
        'success': '#10b981', 'error': '#ef4444', 'highlight_add': '#d1fae5', 
        'highlight_remove': '#fee2e2', 'highlight_change': '#fef3c7'
    }
    COLORS_DARK = {
        'primary': '#3b82f6', 'primary_hover': '#60a5fa', 'secondary': '#374151',
        'secondary_hover': '#4b5563', 'background': '#1f2937', 'surface': '#111827', 
        'border': '#374151', 'text': '#f9fafb', 'text_secondary': '#9ca3af', 
        'success': '#34d399', 'error': '#f87171', 'highlight_add': '#065f46', 
        'highlight_remove': '#991b1b', 'highlight_change': '#92400e'
    }
    PROOFREAD_STYLES = [
        "Standard", "Grammar Fix", "Natural",
        "Polish", "Concise", "Academic",
        "Paraphrase (Academic)", "Casual", "Speech", "Rewrite Boldly"
    ]
    PROOFREAD_INSTRUCTIONS = {
        "Standard": "Fix any grammar, spelling, punctuation errors and improve clarity and flow. Make it more polished and professional.",
        "Grammar Fix": "Only correct grammar, spelling, and punctuation mistakes. Do not change the style or wording otherwise.",
        "Natural": "Rewrite the text to sound more natural and fluent, as a native speaker would write it.",
        "Polish": "Polish the text to a high standard. Improve word choice, sentence structure, and overall elegance.",
        "Concise": "Rewrite the text to be as concise as possible. Remove redundant words, phrases, and sentences without losing the core meaning.",
        "Academic": "Rewrite the text in a formal, objective, and scholarly academic style. Use appropriate terminology and avoid slang or contractions.",
        "Paraphrase (Academic)": "Paraphrase the text using an academic tone. You must use different wording and sentence structure, while preserving the original meaning completely.",
        "Casual": "Rewrite the text in a casual, friendly, and conversational tone. You may use contractions and simpler language.",
        "Speech": "Adapt the text into a script for a speech. Make it engaging, clear, and easy to deliver orally. Use rhetorical devices where appropriate.",
        "Rewrite Boldly": "Completely rewrite the text to improve its quality, clarity, and impact. You have the creative freedom to restructure sentences and change wording significantly, but preserve the original core message."
    }
    PROMPTS = {
        "translate": {
            "system": "You are a professional translator. Provide accurate and natural translations.",
            "user_auto": "Translate the following text to {target_lang}. Only provide the translation, no explanations:\n\n{text}",
            "user_specified": "Translate the following {source_lang} text to {target_lang}. Only provide the translation, no explanations:\n\n{text}"
        },
        "proofread": {
            "system": "You are a world-class editor. You will be given text and an instruction on how to modify it. Adhere to the instruction precisely. Provide only the modified text in your response, without any preamble or explanation.",
            "user": "Instruction: {instruction}\n\nOriginal Text in {language}:\n\n---\n{text}"
        }
    }
    UI_TEXTS = {
        "translate_mode": "Text Translation", "proofread_mode": "Proofreading",
        "source_label": "Source", "target_label_translate": "Translation", "target_label_proofread": "Proofread Result",
        "action_btn_translate": "Translate", "action_btn_proofread": "Proofread",
        "status_translating": "Translating...", "status_proofreading": "Proofreading...",
        "status_success_translate": "Translation Complete", "status_success_proofread": "Proofreading Complete",
        "char_count": "{count} chars", "copied": "Copied"
    }

    def __init__(self, root):
        """
        Initialize the main window, load settings, set up UI and API clients.
        """
        self.root = root
        self.root.title("AI Translation & Writing")
        
        self.settings_manager = SettingsManager()
        
        self.root.geometry(self.settings_manager.settings.get('window_geometry', '1450x900'))
        self.root.minsize(1100, 600)

        self.theme_mode = self.settings_manager.settings.get('theme_mode', 'dark')
        self.colors = self.COLORS_DARK.copy() if self.theme_mode == 'dark' else self.COLORS_LIGHT.copy()

        self.setup_api_clients()
        self.current_mode = 'translate'
        self.translation_history = self.settings_manager.history
        
        self.last_used_models = self.settings_manager.settings.get('last_used_models', {
            'OpenAI': 'gpt-4.1-mini-2025-04-14',
            'Anthropic': 'claude-3-5-haiku-latest',
            'Google': 'gemini-2.5-flash'
        })
        self.current_provider = self.settings_manager.settings.get('default_provider', 'OpenAI')
        self.current_model = self.last_used_models.get(self.current_provider, list(self.MODELS[self.current_provider].keys())[0])

        self.font_family = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI"
        self.font_family_text = "SF Pro Text" if sys.platform == "darwin" else "Segoe UI"

        self.themed_widgets = []
        self.is_processing = False

        self.setup_ui()
        self.apply_theme()
        self.setup_keyboard_shortcuts()
        self.setup_autosave()
        self.create_context_menus()
        self.create_progress_overlay()
        
        self.load_autosaved_content()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """
        Save settings and history before closing the application.
        """
        self.settings_manager.settings['window_geometry'] = self.root.geometry()
        self.settings_manager.settings['theme_mode'] = self.theme_mode
        self.settings_manager.settings['default_provider'] = self.current_provider
        self.settings_manager.settings['last_used_models'] = self.last_used_models
        self.settings_manager.settings['source_lang'] = self.source_lang_var.get()
        self.settings_manager.settings['target_lang'] = self.target_lang_var.get()
        self.settings_manager.settings['proofread_style'] = self.proofread_style_var.get()
        self.settings_manager.save_settings()
        
        self.settings_manager.save_history(self.translation_history)
        self.autosave_content(is_closing=True)
        
        self.root.destroy()

    def setup_keyboard_shortcuts(self):
        """
        Register keyboard shortcuts for main actions (translate, copy, clear, switch mode, show history).
        """
        self.root.bind('<Command-Return>' if sys.platform == 'darwin' else '<Control-Return>', 
                       lambda e: self.perform_action() if self.action_btn['state'] == tk.NORMAL else None)
        self.root.bind('<Command-Shift-c>' if sys.platform == 'darwin' else '<Control-Shift-c>', 
                       lambda e: self.copy_translation())
        self.root.bind('<Command-l>' if sys.platform == 'darwin' else '<Control-l>', 
                       lambda e: self.clear_text())
        self.root.bind('<Command-t>' if sys.platform == 'darwin' else '<Control-t>', 
                       lambda e: self.switch_mode('translate' if self.current_mode == 'proofread' else 'proofread'))
        self.root.bind('<Command-h>' if sys.platform == 'darwin' else '<Control-h>', 
                       lambda e: self.show_history_window())

    def setup_autosave(self):
        """
        Set up periodic autosave of current content every 30 seconds.
        """
        self.root.after(30000, self.autosave_content)

    def autosave_content(self, is_closing=False):
        """
        Save the current source/target text and settings to autosave file.
        """
        try:
            content = {
                'source_text': self.source_text.get("1.0", tk.END).strip(),
                'target_text': self.target_text.get("1.0", tk.END).strip(),
                'mode': self.current_mode,
                'source_lang': self.source_lang_var.get(),
                'target_lang': self.target_lang_var.get(),
                'proofread_style': self.proofread_style_var.get() if hasattr(self, 'proofread_style_var') else '',
                'timestamp': datetime.now().isoformat()
            }
            self.settings_manager.save_autosave(content)
        except Exception:
            pass
        if not is_closing:
            self.root.after(30000, self.autosave_content)

    def load_autosaved_content(self):
        """
        Load autosaved content if available and prompt user to restore.
        """
        autosave = self.settings_manager.load_autosave()
        if autosave and 'timestamp' in autosave:
            try:
                timestamp = datetime.fromisoformat(autosave['timestamp'])
                if (datetime.now() - timestamp).days < 1:
                    if messagebox.askyesno("Auto Save Restore", 
                                         f"Previous work found.\n"
                                         f"({timestamp.strftime('%Y-%m-%d %H:%M')})\n\n"
                                         f"Restore?"):
                        self.source_text.insert("1.0", autosave.get('source_text', ''))
                        if autosave.get('target_text'):
                            self.target_text.config(state=tk.NORMAL)
                            self.target_text.insert("1.0", autosave['target_text'])
                            self.target_text.config(state=tk.DISABLED)
                        
                        if autosave.get('mode'):
                            self.switch_mode(autosave['mode'])
                        if autosave.get('source_lang'):
                            self.source_lang_var.set(autosave['source_lang'])
                        if autosave.get('target_lang'):
                            self.target_lang_var.set(autosave['target_lang'])
                        if autosave.get('proofread_style') and hasattr(self, 'proofread_style_var'):
                            self.proofread_style_var.set(autosave['proofread_style'])
            except Exception:
                pass

    def create_context_menus(self):
        """
        Create right-click context menus for source and target text areas.
        """
        self.source_context_menu = tk.Menu(self.root, tearoff=0)
        self.source_context_menu.add_command(label="Cut", command=lambda: self.source_text.event_generate("<<Cut>>"))
        self.source_context_menu.add_command(label="Copy", command=lambda: self.source_text.event_generate("<<Copy>>"))
        self.source_context_menu.add_command(label="Paste", command=lambda: self.source_text.event_generate("<<Paste>>"))
        self.source_context_menu.add_separator()
        self.source_context_menu.add_command(label="Select All", command=lambda: self.source_text.tag_add("sel", "1.0", "end"))
        
        self.source_text.bind("<Button-3>" if sys.platform == "win32" else "<Button-2>", 
                            lambda e: self.source_context_menu.post(e.x_root, e.y_root))
        
        self.target_context_menu = tk.Menu(self.root, tearoff=0)
        self.target_context_menu.add_command(label="Copy", command=lambda: self.copy_selected_text())
        self.target_context_menu.add_command(label="Select All", command=lambda: self.target_text.tag_add("sel", "1.0", "end"))
        
        self.target_text.bind("<Button-3>" if sys.platform == "win32" else "<Button-2>", 
                            lambda e: self.target_context_menu.post(e.x_root, e.y_root))

    def copy_selected_text(self):
        """
        Copy selected text from the target text area to clipboard.
        """
        try:
            selected_text = self.target_text.get("sel.first", "sel.last")
            if selected_text:
                self.root.clipboard_clear()
                self.root.clipboard_append(selected_text)
        except tk.TclError:
            pass

    def create_progress_overlay(self):
        """
        Create an overlay to show progress during API calls.
        """
        self.progress_overlay = tk.Frame(self.root, bg='black')
        self.progress_label = tk.Label(self.progress_overlay, text="Processing...", 
            font=(self.font_family, 16), fg='white', bg='black')
        self.progress_label.place(relx=0.5, rely=0.5, anchor='center')
        
        self.progress_dots = tk.Label(self.progress_overlay, text="",
            font=(self.font_family, 16), fg='white', bg='black')
        self.progress_dots.place(relx=0.5, rely=0.55, anchor='center')
        
    def show_progress(self, message: str = "Processing..."):
        """
        Display the progress overlay with a message.
        """
        self.is_processing = True
        self.progress_label.config(text=message)
        self.progress_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.progress_overlay.lift()
        self.animate_progress_dots()
        
    def animate_progress_dots(self, dots=""):
        """
        Animate dots on the progress overlay to indicate activity.
        """
        if self.is_processing:
            dots = dots + "." if len(dots) < 3 else ""
            self.progress_dots.config(text=dots)
            self.root.after(500, lambda: self.animate_progress_dots(dots))
        
    def hide_progress(self):
        """
        Hide the progress overlay.
        """
        self.is_processing = False
        self.progress_overlay.place_forget()

    def show_history_window(self):
        """
        Open a window displaying translation and proofreading history.
        """
        history_window = tk.Toplevel(self.root)
        history_window.title("History")
        history_window.geometry("900x600")
        history_window.configure(bg=self.colors['surface'])
        
        header = tk.Label(history_window, text="Translation & Proofreading History", font=(self.font_family, 18, 'bold'),
            bg=self.colors['surface'], fg=self.colors['text'])
        header.pack(pady=10)
        
        tree_frame = tk.Frame(history_window, bg=self.colors['surface'])
        tree_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        columns = ('Time', 'Mode', 'Model', 'Source', 'Result')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        tree.heading('Time', text='Time')
        tree.heading('Mode', text='Mode')
        tree.heading('Model', text='Model')
        tree.heading('Source', text='Source')
        tree.heading('Result', text='Result')
        
        tree.column('Time', width=150)
        tree.column('Mode', width=80)
        tree.column('Model', width=150)
        tree.column('Source', width=250)
        tree.column('Result', width=250)
        
        v_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        for item in reversed(self.translation_history[-500:]):
            timestamp = item.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            
            tree.insert('', 'end', values=(
                timestamp,
                item.get('mode', ''),
                item.get('model', '').split('/')[-1] if '/' in item.get('model', '') else item.get('model', ''),
                item['source'][:50] + '...' if len(item['source']) > 50 else item['source'],
                item['target'][:50] + '...' if len(item['target']) > 50 else item['target']
            ))
        
        def on_double_click(event):
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                values = item['values']
                for hist_item in self.translation_history:
                    if (hist_item.get('source', '').startswith(values[3][:30]) and 
                        hist_item.get('target', '').startswith(values[4][:30])):
                        detail_window = tk.Toplevel(history_window)
                        detail_window.title("Details")
                        detail_window.geometry("800x600")
                        detail_window.configure(bg=self.colors['surface'])
                        
                        tk.Label(detail_window, text="Source:", font=(self.font_family, 12, 'bold'),
                               bg=self.colors['surface'], fg=self.colors['text']).pack(anchor='w', padx=20, pady=(20, 5))
                        
                        source_text = tk.Text(detail_window, height=10, wrap=tk.WORD,
                                            bg=self.colors['background'], fg=self.colors['text'])
                        source_text.pack(fill='both', expand=True, padx=20, pady=(0, 10))
                        source_text.insert('1.0', hist_item['source'])
                        source_text.config(state='disabled')
                        
                        tk.Label(detail_window, text="Result:", font=(self.font_family, 12, 'bold'),
                               bg=self.colors['surface'], fg=self.colors['text']).pack(anchor='w', padx=20, pady=(10, 5))
                        
                        target_text = tk.Text(detail_window, height=10, wrap=tk.WORD,
                                            bg=self.colors['background'], fg=self.colors['text'])
                        target_text.pack(fill='both', expand=True, padx=20, pady=(0, 20))
                        target_text.insert('1.0', hist_item['target'])
                        target_text.config(state='disabled')
                        
                        break
        
        tree.bind('<Double-Button-1>', on_double_click)
        
        button_frame = tk.Frame(history_window, bg=self.colors['surface'])
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Clear History", command=lambda: self.clear_history(history_window),
            style="Secondary.TButton").pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Close", command=history_window.destroy,
            style="Secondary.TButton").pack(side='left', padx=5)

    def clear_history(self, window):
        """
        Clear all translation/proofread history after user confirmation.
        """
        if messagebox.askyesno("Confirm", "Are you sure you want to delete all history?"):
            self.translation_history.clear()
            self.settings_manager.save_history([])
            window.destroy()
            messagebox.showinfo("Complete", "History deleted.")

    def setup_api_clients(self):
        """
        Initialize API clients for OpenAI, Anthropic, and Google Gemini using environment variables.
        """
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") else None
        
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_configured = True
        else:
            self.gemini_configured = False
        
        if not self.openai_client and not self.anthropic_client and not self.gemini_configured:
            messagebox.showwarning("API Key Missing", "Please set at least one API key in your .env file.")

    def _get_provider(self) -> Optional[LLMProvider]:
        """
        Return the current LLMProvider instance based on selected provider and model.
        """
        try:
            if self.current_provider == "OpenAI":
                return OpenAIProvider(self.openai_client, self.current_model)
            elif self.current_provider == "Anthropic":
                return AnthropicProvider(self.anthropic_client, self.current_model)
            elif self.current_provider == "Google":
                if not self.gemini_configured: raise ValueError("Google API Key not configured.")
                return GoogleProvider(genai, self.current_model)
        except ValueError as e:
            messagebox.showerror("Error", f"{self.current_provider} API key not configured.")
            return None
        return None
        
    def setup_ui(self):
        """
        Set up the main UI layout and widgets.
        """
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.themed_widgets.append(self.main_container)
        
        self.create_header(self.main_container)
        self.create_mode_selector(self.main_container)
        self.create_language_bar(self.main_container)
        
        translation_container = tk.Frame(self.main_container)
        translation_container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.themed_widgets.append(translation_container)
        
        translation_container.grid_columnconfigure(0, weight=1)
        translation_container.grid_columnconfigure(1, weight=0)
        translation_container.grid_columnconfigure(2, weight=1)
        translation_container.grid_rowconfigure(0, weight=1)
        
        self.create_translation_areas(translation_container)
        self.create_control_bar(self.main_container)

    def toggle_theme(self):
        """
        Toggle between light and dark UI themes.
        """
        if self.theme_mode == 'light':
            self.theme_mode = 'dark'
            self.colors = self.COLORS_DARK.copy()
            self.theme_btn.config(text="☀️")
        else:
            self.theme_mode = 'light'
            self.colors = self.COLORS_LIGHT.copy()
            self.theme_btn.config(text="🌙")
        
        self.apply_theme()

    def apply_theme(self):
        """
        Apply the current theme colors and styles to all widgets.
        """
        self.setup_styles()

        self.root.config(bg=self.colors['surface'])
        for widget in self.themed_widgets:
            if isinstance(widget, (tk.Frame, tk.Label)):
                widget.config(bg=self.colors.get('surface', '#FFFFFF'))
        
        self.title_label.config(fg=self.colors['text'], bg=self.colors['surface'])
        self.provider_label.config(fg=self.colors['text_secondary'], bg=self.colors['surface'])

        self.lang_container.config(bg=self.colors['background'])
        self.lang_frame.config(bg=self.colors['background'])
        self.source_frame.config(bg=self.colors['background'])
        self.source_lang_label.config(fg=self.colors['text_secondary'], bg=self.colors['background'])
        self.swap_frame.config(bg=self.colors['background'])
        self.swap_btn.config(fg=self.colors['primary'], bg=self.colors['background'], activebackground=self.colors['background'], activeforeground=self.colors['primary_hover'])
        self.target_frame.config(bg=self.colors['background'])
        self.target_lang_label.config(fg=self.colors['text_secondary'], bg=self.colors['background'])
        
        self.source_text_area_frame.config(bg=self.colors['background'])
        self.source_header.config(bg=self.colors['background'])
        self.left_label.config(fg=self.colors['text'], bg=self.colors['background'])
        self.source_char_label.config(fg=self.colors['text_secondary'], bg=self.colors['background'])
        self.source_text.config(bg=self.colors['background'], fg=self.colors['text'], insertbackground=self.colors['primary'], selectbackground=self.colors['primary'])

        self.target_text_area_frame.config(bg=self.colors['background'])
        self.target_header.config(bg=self.colors['background'])
        self.right_label.config(fg=self.colors['text'], bg=self.colors['background'])
        self.target_char_label.config(fg=self.colors['text_secondary'], bg=self.colors['background'])
        self.target_text.config(bg=self.colors['background'], fg=self.colors['text'])
        self.target_text.tag_configure("add", background=self.colors['highlight_add'], foreground=self.colors['text'])
        self.target_text.tag_configure("remove", background=self.colors['highlight_remove'], foreground=self.colors['text'])
        self.target_text.tag_configure("change", background=self.colors['highlight_change'], foreground=self.colors['text'])

        self.status_label.config(fg=self.colors['text_secondary'], bg=self.colors['surface'])
        
        self.switch_mode(self.current_mode, force_update=True)
        
    def setup_styles(self):
        """
        Configure ttk styles for buttons, comboboxes, and other widgets.
        """
        style = ttk.Style()
        font_bold = (self.font_family, 12, 'bold')
        font_normal = (self.font_family, 11)

        style.configure("Modern.TButton", padding=(20, 10), relief="flat", background=self.colors['primary'],
                        foreground='white', borderwidth=0, focuscolor='none', font=font_bold)
        style.map("Modern.TButton", background=[('active', self.colors['primary_hover'])])
        
        style.configure("Secondary.TButton", padding=(15, 8), relief="flat", background=self.colors['surface'],
                        foreground=self.colors['text_secondary'], borderwidth=0, focuscolor='none', font=font_normal)
        style.map("Secondary.TButton", foreground=[('active', self.colors['text'])], background=[('active', self.colors['border'])])

        style.configure("Mode.TButton", padding=(15, 10), relief="flat", background=self.colors['background'],
                        foreground=self.colors['text'], borderwidth=0, focuscolor='none', font=(self.font_family, 12))
        style.map("Mode.TButton", background=[('active', self.colors['primary'])], foreground=[('active', 'white')])

        style.configure("ModeActive.TButton", padding=(15, 10), relief="flat", background=self.colors['primary'],
                        foreground='white', borderwidth=0, focuscolor='none', font=font_bold)
        
        style.configure("Modern.TCombobox", borderwidth=0, relief="flat", font=font_normal, padding=5)
        style.map('Modern.TCombobox', 
                  fieldbackground=[('readonly', self.colors['secondary'])],
                  foreground=[('readonly', 'white' if self.theme_mode == 'dark' else self.colors['text'])],
                  selectbackground=[('readonly', self.colors['secondary'])],
                  selectforeground=[('readonly', 'white' if self.theme_mode == 'dark' else self.colors['text'])])

        style.configure("Theme.TButton", padding=10, relief="flat", background=self.colors['surface'], 
                        foreground=self.colors['text'], font=(self.font_family, 14), borderwidth=0,
                        focuscolor=self.colors['surface'])
        style.map("Theme.TButton", background=[('active', self.colors['border'])])
        
    def create_header(self, parent):
        """
        Create the application header with title, theme switch, and model selection.
        """
        self.header_frame = tk.Frame(parent)
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        self.themed_widgets.append(self.header_frame)
        
        left_frame = tk.Frame(self.header_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.themed_widgets.append(left_frame)

        self.title_label = tk.Label(left_frame, text="AI Translation & Writing Studio 2025", font=(self.font_family, 24, 'bold'))
        self.title_label.pack(side=tk.LEFT)
        
        right_frame = tk.Frame(self.header_frame)
        right_frame.pack(side=tk.RIGHT)
        self.themed_widgets.append(right_frame)

        self.history_btn = ttk.Button(right_frame, text="📜", command=self.show_history_window, 
                                    style="Theme.TButton", width=3)
        self.history_btn.pack(side=tk.RIGHT, padx=(5, 0))

        self.theme_btn = ttk.Button(right_frame, text="☀️" if self.theme_mode == 'dark' else "🌙", 
                                  command=self.toggle_theme, style="Theme.TButton", width=3)
        self.theme_btn.pack(side=tk.RIGHT, padx=(5, 0))

        model_frame = tk.Frame(right_frame)
        model_frame.pack(side=tk.RIGHT)
        self.themed_widgets.append(model_frame)

        self.provider_label = tk.Label(model_frame, text="AI Model:")
        self.provider_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.provider_var = tk.StringVar(value=self.current_provider)
        provider_combo = ttk.Combobox(model_frame, textvariable=self.provider_var, values=list(self.MODELS.keys()), 
                                    state="readonly", width=10, style="Modern.TCombobox")
        provider_combo.pack(side=tk.LEFT, padx=(0, 10))
        provider_combo.bind('<<ComboboxSelected>>', self.on_provider_change)
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", 
                                      width=45, style="Modern.TCombobox")
        self.model_combo.pack(side=tk.LEFT)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)

        self.update_model_list()
        self.set_default_model_display()
        
    def on_provider_change(self, event=None):
        """
        Update model list and selection when provider is changed.
        """
        self.current_provider = self.provider_var.get()
        self.update_model_list()

        last_model = self.last_used_models.get(self.current_provider)
        available_models = self.MODELS.get(self.current_provider, {})

        if last_model and last_model in available_models:
            self.current_model = last_model
        else:
            model_keys = list(available_models.keys())
            if model_keys:
                self.current_model = model_keys[0]
            else:
                self.current_model = ""

        self.set_default_model_display()

    def on_model_change(self, event=None):
        """
        Update current model when model selection changes.
        Warn if an expensive model is selected.
        """
        selected_value = self.model_var.get()
        if not selected_value or "API key required" in selected_value:
            self.current_model = ""
            return
            
        self.current_model = selected_value.split(' ')[0]
        
        self.last_used_models[self.current_provider] = self.current_model
        
        if "gpt-4.5-preview" in self.current_model:
            messagebox.showwarning("Expensive Model Warning", "GPT-4.5 Preview is very expensive!")
        
    def update_model_list(self):
        """
        Update the list of available models for the selected provider.
        """
        provider = self.provider_var.get()
        models = self.MODELS.get(provider, {})
        
        display_list = [f"{key} ({desc})" for key, desc in models.items()]
        self.model_combo['values'] = display_list
        
        key_available = (provider == "OpenAI" and self.openai_client) or \
                        (provider == "Anthropic" and self.anthropic_client) or \
                        (provider == "Google" and self.gemini_configured)
                        
        if not key_available:
            self.model_combo.set(f"{provider} API key required")
            self.model_combo.config(state="disabled")
        else:
            self.model_combo.config(state="readonly")
            if not display_list:
                self.model_combo.set("No available models")

    def set_default_model_display(self):
        """
        Set the display text for the currently selected model.
        """
        provider = self.current_provider
        model_key = self.current_model
        models = self.MODELS.get(provider, {})

        if model_key in models:
            display_text = f"{model_key} ({models[model_key]})"
            self.model_var.set(display_text)
        elif models:
            first_key = list(models.keys())[0]
            self.current_model = first_key
            display_text = f"{first_key} ({models[first_key]})"
            self.model_var.set(display_text)

    def create_mode_selector(self, parent):
        """
        Create buttons to switch between translation and proofreading modes.
        """
        self.mode_frame = tk.Frame(parent, bd=0)
        self.mode_frame.pack(fill=tk.X, pady=(0, 10))
        self.themed_widgets.append(self.mode_frame)

        self.mode_inner_frame = tk.Frame(self.mode_frame)
        self.mode_inner_frame.pack(pady=5)
        self.themed_widgets.append(self.mode_inner_frame)
        
        self.translate_mode_btn = ttk.Button(self.mode_inner_frame, text=self.UI_TEXTS['translate_mode'], 
                                           command=lambda: self.switch_mode('translate'), style="ModeActive.TButton")
        self.translate_mode_btn.pack(side=tk.LEFT, padx=(10, 5))
        
        self.proofread_mode_btn = ttk.Button(self.mode_inner_frame, text=self.UI_TEXTS['proofread_mode'], 
                                           command=lambda: self.switch_mode('proofread'), style="Mode.TButton")
        self.proofread_mode_btn.pack(side=tk.LEFT, padx=(5, 10))
        
        saved_style = self.settings_manager.settings.get('proofread_style', self.PROOFREAD_STYLES[0])
        self.proofread_style_var = tk.StringVar(value=saved_style)
        self.proofread_style_combo = ttk.Combobox(self.mode_inner_frame, textvariable=self.proofread_style_var,
            values=self.PROOFREAD_STYLES, state="readonly", width=18, style="Modern.TCombobox")
        self.proofread_style_combo.bind('<<ComboboxSelected>>', self.on_proofread_style_change)
        
    def on_proofread_style_change(self, event=None):
        """
        (Placeholder) Handle changes to the selected proofreading style.
        """
        pass
        
    def switch_mode(self, mode, force_update=False):
        """
        Switch between translation and proofreading modes, updating UI accordingly.
        """
        if mode == self.current_mode and not force_update: return
        self.current_mode = mode
        
        is_translate = mode == 'translate'
        self.translate_mode_btn.configure(style="ModeActive.TButton" if is_translate else "Mode.TButton")
        self.proofread_mode_btn.configure(style="ModeActive.TButton" if not is_translate else "Mode.TButton")
        
        self.swap_btn.config(state=tk.NORMAL if is_translate else tk.DISABLED)
        self.target_lang_combo.config(state="readonly" if is_translate else tk.DISABLED)
        
        self.left_label.config(text=self.UI_TEXTS['source_label'])
        self.right_label.config(text=self.UI_TEXTS['target_label_translate'] if is_translate else self.UI_TEXTS['target_label_proofread'])
        self.action_btn.config(text=self.UI_TEXTS['action_btn_translate'] if is_translate else self.UI_TEXTS['action_btn_proofread'])
        
        if is_translate:
            self.proofread_style_combo.pack_forget()
        else:
            self.proofread_style_combo.pack(side=tk.LEFT, padx=(10, 5))
            if self.source_lang_var.get() != "Auto Detect":
                self.target_lang_var.set(self.source_lang_var.get())
        
        if not force_update:
            self.clear_text()
        
    def create_language_bar(self, parent):
        """
        Create the language selection bar for source and target languages.
        """
        self.lang_container = tk.Frame(parent, bd=0)
        self.lang_container.pack(fill=tk.X, pady=(0, 10))
        self.themed_widgets.append(self.lang_container)
        
        self.lang_frame = tk.Frame(self.lang_container)
        self.lang_frame.pack(pady=10, padx=20)
        self.themed_widgets.append(self.lang_frame)

        self.source_frame = tk.Frame(self.lang_frame)
        self.source_frame.pack(side=tk.LEFT)
        self.themed_widgets.append(self.source_frame)
        
        self.source_lang_label = tk.Label(self.source_frame, text="Language")
        self.source_lang_label.pack(anchor=tk.W, pady=(0, 5))
        
        saved_source = self.settings_manager.settings.get('source_lang', 'Auto Detect')
        self.source_lang_var = tk.StringVar(value=saved_source)
        self.source_lang_combo = ttk.Combobox(self.source_frame, textvariable=self.source_lang_var, 
                                            values=list(self.LANGUAGES.keys()), state="readonly", 
                                            width=18, style="Modern.TCombobox")
        self.source_lang_combo.pack()
        self.source_lang_combo.bind('<<ComboboxSelected>>', self.on_source_lang_change)
        
        self.swap_frame = tk.Frame(self.lang_frame)
        self.swap_frame.pack(side=tk.LEFT, padx=30)
        self.themed_widgets.append(self.swap_frame)
        
        self.swap_btn = tk.Button(self.swap_frame, text="⇄", command=self.swap_languages, 
                                font=(self.font_family, 20), bd=0, cursor="hand2")
        self.swap_btn.pack(pady=(20, 0))
        
        self.target_frame = tk.Frame(self.lang_frame)
        self.target_frame.pack(side=tk.LEFT)
        self.themed_widgets.append(self.target_frame)
        
        self.target_lang_label = tk.Label(self.target_frame, text="Target")
        self.target_lang_label.pack(anchor=tk.W, pady=(0, 5))
        
        saved_target = self.settings_manager.settings.get('target_lang', 'Japanese')
        self.target_lang_var = tk.StringVar(value=saved_target)
        self.target_lang_combo = ttk.Combobox(self.target_frame, textvariable=self.target_lang_var, 
                                            values=[lang for lang in self.LANGUAGES.keys() if lang != "Auto Detect"], 
                                            state="readonly", width=18, style="Modern.TCombobox")
        self.target_lang_combo.pack()
        self.target_lang_combo.bind('<<ComboboxSelected>>', self.on_target_lang_change)

    def on_source_lang_change(self, event=None):
        """
        Update target language if in proofreading mode and source language is changed.
        """
        if self.current_mode == 'proofread' and self.source_lang_var.get() != "Auto Detect":
            self.target_lang_var.set(self.source_lang_var.get())
        
    def on_target_lang_change(self, event=None):
        """
        (Placeholder) Handle changes to the target language selection.
        """
        pass
        
    def create_translation_areas(self, parent):
        """
        Create the text input/output areas for source and target text.
        """
        self.source_text_area_frame = tk.Frame(parent, bd=0)
        self.source_text_area_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.themed_widgets.append(self.source_text_area_frame)

        self.source_header = tk.Frame(self.source_text_area_frame)
        self.source_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        self.themed_widgets.append(self.source_header)
        
        self.left_label = tk.Label(self.source_header, text=self.UI_TEXTS['source_label'], 
                                 font=(self.font_family, 14, 'bold'))
        self.left_label.pack(side=tk.LEFT)
        
        self.source_char_label = tk.Label(self.source_header, text=self.UI_TEXTS['char_count'].format(count=0))
        self.source_char_label.pack(side=tk.RIGHT)
        
        text_frame_src = tk.Frame(self.source_text_area_frame)
        text_frame_src.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.source_text = tk.Text(text_frame_src, wrap=tk.WORD, font=(self.font_family_text, 14), 
                                 relief=tk.FLAT, bd=0, padx=15, pady=15, selectforeground='white')
        self.source_text.pack(fill=tk.BOTH, expand=True)
        self.source_text.bind('<KeyRelease>', self.on_source_change)

        self.target_text_area_frame = tk.Frame(parent, bd=0)
        self.target_text_area_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        self.themed_widgets.append(self.target_text_area_frame)
        
        self.target_header = tk.Frame(self.target_text_area_frame)
        self.target_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        self.themed_widgets.append(self.target_header)
        
        self.right_label = tk.Label(self.target_header, text=self.UI_TEXTS['target_label_translate'], 
                                  font=(self.font_family, 14, 'bold'))
        self.right_label.pack(side=tk.LEFT)
        
        self.target_char_label = tk.Label(self.target_header, text=self.UI_TEXTS['char_count'].format(count=0))
        self.target_char_label.pack(side=tk.RIGHT)
        
        text_frame_tgt = tk.Frame(self.target_text_area_frame)
        text_frame_tgt.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.target_text = tk.Text(text_frame_tgt, wrap=tk.WORD, font=(self.font_family_text, 14), 
                                 relief=tk.FLAT, bd=0, padx=15, pady=15, state=tk.DISABLED, cursor="arrow")
        self.target_text.pack(fill=tk.BOTH, expand=True)

    def create_control_bar(self, parent):
        """
        Create the control bar with clear, copy, and action buttons.
        """
        control_frame = tk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        self.themed_widgets.append(control_frame)
        
        left_frame = tk.Frame(control_frame)
        left_frame.pack(side=tk.LEFT)
        self.themed_widgets.append(left_frame)
        
        self.clear_btn = ttk.Button(left_frame, text="Clear", command=self.clear_text, style="Secondary.TButton")
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.copy_btn = ttk.Button(left_frame, text="Copy", command=self.copy_translation, style="Secondary.TButton")
        self.copy_btn.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(control_frame, text="")
        self.status_label.pack(side=tk.LEFT, padx=30)
        
        right_frame = tk.Frame(control_frame)
        right_frame.pack(side=tk.RIGHT)
        self.themed_widgets.append(right_frame)

        self.action_btn = ttk.Button(right_frame, text=self.UI_TEXTS['action_btn_translate'], 
                                   command=self.perform_action, style="Modern.TButton", state=tk.DISABLED)
        self.action_btn.pack()
        
    def on_source_change(self, event=None):
        """
        Handle changes in the source text area, update character count, and enable/disable action button.
        """
        text = self.source_text.get("1.0", tk.END).strip()
        self.source_char_label.config(text=self.UI_TEXTS['char_count'].format(count=len(text)))
        
        if text:
            self.action_btn.config(state=tk.NORMAL)
            if hasattr(self, '_process_timer'): self.root.after_cancel(self._process_timer)
            self._process_timer = self.root.after(1500, self.perform_action)
        else:
            self.action_btn.config(state=tk.DISABLED)
            self.clear_text(clear_source=False)
            
    def perform_action(self):
        """
        Perform translation or proofreading using the selected provider/model and update the result.
        """
        source_text = self.source_text.get("1.0", tk.END).strip()
        if not source_text or not self.current_model: return

        provider = self._get_provider()
        if not provider:
            self.status_label.config(text="API client is not configured", fg=self.colors['error'])
            return

        self.show_progress(self.UI_TEXTS['status_translating'] if self.current_mode == 'translate' else self.UI_TEXTS['status_proofreading'])
        
        self.action_btn.config(state=tk.DISABLED)
        system_prompt = ""
        user_prompt = ""
        is_proofread_task = False

        if self.current_mode == 'translate':
            is_proofread_task = False
            source_lang = self.LANGUAGES[self.source_lang_var.get()]
            target_lang = self.LANGUAGES[self.target_lang_var.get()]
            
            if source_lang == "auto":
                prompt_template = self.PROMPTS["translate"]["user_auto"]
                user_prompt = prompt_template.format(target_lang=target_lang, text=source_text)
            else:
                prompt_template = self.PROMPTS["translate"]["user_specified"]
                user_prompt = prompt_template.format(source_lang=source_lang, target_lang=target_lang, text=source_text)
            system_prompt = self.PROMPTS["translate"]["system"]

        else: # proofread
            is_proofread_task = True
            
            selected_style = self.proofread_style_var.get()
            instruction = self.PROOFREAD_INSTRUCTIONS.get(selected_style, self.PROOFREAD_INSTRUCTIONS["Standard"])
            language = self.source_lang_var.get() if self.source_lang_var.get() != "Auto Detect" else "its original language"
            
            user_prompt = self.PROMPTS["proofread"]["user"].format(
                instruction=instruction,
                language=language,
                text=source_text
            )
            system_prompt = self.PROMPTS["proofread"]["system"]

        thread = threading.Thread(target=self._call_api_and_update, args=(provider, system_prompt, user_prompt, is_proofread_task))
        thread.daemon = True
        thread.start()

    def _call_api_and_update(self, provider: LLMProvider, system_prompt: str, user_prompt: str, is_proofread: bool):
        """
        Call the selected LLM API in a background thread and update the UI with the result.
        """
        try:
            result_text = provider.call(system_prompt, user_prompt)
            self.root.after(0, self._update_result, result_text, True, is_proofread)
        except Exception as e:
            error_msg = self._get_user_friendly_error(e)
            self.root.after(0, self._update_result, error_msg, False, is_proofread)
            
    def _get_user_friendly_error(self, error: Exception) -> str:
        """
        Convert API/library errors to user-friendly error messages.
        """
        error_str = str(error).lower()
        
        if 'rate limit' in error_str:
            return "You have reached the API rate limit. Please wait and try again later."
        elif 'api key' in error_str or 'unauthorized' in error_str:
            return "API key is invalid. Please check your settings."
        elif 'timeout' in error_str:
            return "Request timed out. Please check your network connection."
        elif 'quota' in error_str:
            return "You have reached your API quota."
        elif 'model' in error_str and 'not found' in error_str:
            return "The selected model is not available. Please choose another model."
        else:
            return f"An error occurred: {str(error)}"
            
    def _update_result(self, text: str, success: bool, is_proofread: bool):
        """
        Update the target text area and status after translation/proofreading completes.
        Also saves the result to history.
        """
        self.hide_progress()
        
        self.target_text.config(state=tk.NORMAL)
        self.target_text.delete("1.0", tk.END)
        
        if success and is_proofread:
            original_text = self.source_text.get("1.0", tk.END).strip()
            self._show_differences(original_text, text)
        else:
            self.target_text.insert("1.0", text)
        
        self.target_text.config(state=tk.DISABLED)
        self.target_char_label.config(text=self.UI_TEXTS['char_count'].format(count=len(text)))
        
        if success:
            status_key = 'status_success_proofread' if is_proofread else 'status_success_translate'
            self.status_label.config(text=self.UI_TEXTS[status_key], fg=self.colors['success'])
            self.root.after(2000, lambda: self.status_label.config(text=""))
            
            history_entry = {
                'source': self.source_text.get("1.0", tk.END).strip(),
                'target': text,
                'source_lang': self.source_lang_var.get(),
                'target_lang': self.target_lang_var.get(),
                'mode': 'proofread' if is_proofread else 'translate',
                'model': f"{self.current_provider}/{self.current_model}",
                'timestamp': datetime.now().isoformat()
            }
            if is_proofread:
                history_entry['proofread_style'] = self.proofread_style_var.get()
                
            self.translation_history.append(history_entry)
            
            self.settings_manager.save_history(self.translation_history)
        else:
            self.status_label.config(text="An error occurred", fg=self.colors['error'])
            
        self.action_btn.config(state=tk.NORMAL)
        
    def _show_differences(self, original: str, corrected: str):
        """
        Highlight differences between original and proofread text using color tags.
        """
        original_words = re.findall(r'\S+|\s+', original)
        corrected_words = re.findall(r'\S+|\s+', corrected)
        diff = difflib.SequenceMatcher(None, original_words, corrected_words)
        
        for tag, i1, i2, j1, j2 in diff.get_opcodes():
            text = ''.join(corrected_words[j1:j2])
            if tag == 'equal':
                self.target_text.insert(tk.END, text)
            elif tag == 'replace':
                self.target_text.insert(tk.END, text, 'change')
            elif tag == 'insert':
                self.target_text.insert(tk.END, text, 'add')

    def swap_languages(self):
        """
        Swap the source and target languages and their respective text areas.
        """
        if self.source_lang_var.get() != "Auto Detect" and self.current_mode == 'translate':
            source_lang, target_lang = self.source_lang_var.get(), self.target_lang_var.get()
            self.source_lang_var.set(target_lang)
            self.target_lang_var.set(source_lang)
            
            source_text = self.source_text.get("1.0", tk.END).strip()
            target_text = self.target_text.get("1.0", tk.END).strip()
            
            self.source_text.delete("1.0", tk.END)
            self.source_text.insert("1.0", target_text)
            
            if target_text: self.perform_action()

    def clear_text(self, clear_source=True):
        """
        Clear the source and/or target text areas and reset status labels.
        """
        if clear_source: self.source_text.delete("1.0", tk.END)
        self.target_text.config(state=tk.NORMAL)
        self.target_text.delete("1.0", tk.END)
        self.target_text.config(state=tk.DISABLED)
        self.status_label.config(text="")
        if clear_source: self.action_btn.config(state=tk.DISABLED)
        if clear_source: self.source_char_label.config(text=self.UI_TEXTS['char_count'].format(count=0))
        self.target_char_label.config(text=self.UI_TEXTS['char_count'].format(count=0))
        
    def copy_translation(self):
        """
        Copy the translated/proofread text to the clipboard and show a status message.
        """
        translation = self.target_text.get("1.0", tk.END).strip()
        if translation:
            self.root.clipboard_clear()
            self.root.clipboard_append(translation)
            self.status_label.config(text=self.UI_TEXTS['copied'], fg=self.colors['success'])
            self.root.after(2000, lambda: self.status_label.config(text=""))

def main():
    """
    Entry point for the application. Sets up DPI awareness (Windows), creates the main window, and starts the Tkinter event loop.
    """
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass 

    root = tk.Tk()
    root.resizable(True, True) 
    
    if sys.platform == "darwin":
        try:
            root.tk.call("::tk::unsupported::MacWindowStyle", "style", root._w, "floating", "closeBox", "collapseBox", "resizable")
        except tk.TclError:
            pass
            
    app = ModernTranslationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
