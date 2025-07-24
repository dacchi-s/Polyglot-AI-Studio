# Polyglot AI Studio

Polyglot AI Studio is a cross-platform desktop application for AI-powered translation and advanced text proofreading. It provides a clean, efficient interface to access the world's leading language models from OpenAI, Anthropic, and Google, all in one place.

---

## ✨ Features

* **Multi-Provider Support**: Seamlessly switch between **OpenAI**, **Anthropic (Claude)**, and **Google (Gemini)** models to compare results and choose the best one for your needs.
* **Dual Modes**: 
    * **Translation Mode**: Fast and accurate text translation with automatic language detection.
    * **Proofreading Mode**: Go beyond simple grammar checks. Rewrite your text in various styles: make it more **concise**, sound more **natural**, adopt an **academic** tone, or even **paraphrase** complex sentences.
* **Modern UI**: A clean, intuitive interface with both **light and dark modes**. The UI is designed to be responsive and easy to use.
* **Difference Highlighting**: In Proofreading Mode, the app visually highlights additions, deletions, and changes, so you can easily see what was improved.
* **Persistent Settings**: The app remembers your last-used model for each provider, your preferred theme, language settings, and window size.
* **Full-Featured**: Includes an accessible translation **history**, **autosave** functionality to prevent data loss, keyboard **shortcuts**, and a convenient progress indicator for API calls.

---

## 🛠️ Setup & Installation

Follow these steps to get the application running on your local machine.

### **1. Prerequisites**

* Python 3.8 or higher
* An active internet connection
* Conda or Pip for package management

### **2. Clone the Repository**

First, clone this repository to your local machine:
```
git clone https://github.com/your-username/Polyglot-AI-Studio.git
cd Polyglot-AI-Studio
```

3. Set Up API Keys
The application requires API keys from the AI providers you wish to use.

Create a file named .env in the root directory of the project.

Add your API keys to this file. You only need to add the keys for the services you plan to use.

# .env file content
```
OPENAI_API_KEY="your_openai_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
GOOGLE_API_KEY="your_google_api_key_here"
```

4. Install Dependencies (Choose one option)
Option A: Using conda
This is the recommended method for managing environments.
Create the Conda Environment: Use the environment.yml file to create a new environment named polyglot-studio.

```
conda env create -f environment.yml
```

Activate the Environment:
```
conda activate polyglot-studio
```

Option B: Using pip
It is highly recommended to use a virtual environment.

Create a Virtual Environment:

```
python -m venv venv
```

Activate the Environment:

On macOS/Linux:

```
source venv/bin/activate
```

On Windows:

```
.\venv\Scripts\activate
```

Install Dependencies: Use the requirements.txt file to install the necessary packages.

```
pip install -r requirements.txt
```

5. Run the Application
Once the dependencies are installed and your environment is activated, you can run the application with the following command:

```
python Polyglot＿AI＿Studio.py
```

🚀 How to Use
Select a Provider and Model: Use the dropdown menus at the top right to choose your preferred AI provider (OpenAI, Anthropic, Google) and the specific model you want to use.

Choose a Mode:

Click "Text Translation" for standard translation tasks.

Click "Proofreading" for advanced text improvement. When in this mode, another dropdown will appear, allowing you to select a writing style (e.g., Academic, Concise, Paraphrase).

Enter Your Text: Type or paste your text into the left-hand panel.

Get Results: The result will automatically appear in the right-hand panel after a brief pause. You can also press Ctrl+Enter (or Cmd+Return on Mac) to trigger the action manually.

Toggle Theme: Use the 🌙 / ☀️ button in the header to switch between dark and light modes.

View History: Click the 📜 button to see a log of your recent activity.
