# START HERE: run this project with no experience

You will install two free programs (Python and VS Code), open this folder, paste three commands, and the app opens in your browser. It takes about 15 minutes. These steps are written for Windows. Mac notes are at the end.

## Step 1: Install Python
1. Go to **python.org/downloads** and click the big yellow **Download Python** button.
2. Open the downloaded file.
3. On the first screen, **tick the box "Add python.exe to PATH"** at the bottom. This is the step people miss.
4. Click **Install Now** and wait until it says "Setup was successful".

## Step 2: Install VS Code
1. Go to **code.visualstudio.com** and click **Download for Windows**.
2. Open the file and keep clicking **Next**, then **Install**.

## Step 3: Unzip the project
Right-click `ev-charge-predictor.zip`, choose **Extract All**, then **Extract**. Put the extracted folder somewhere easy, such as the Desktop.

## Step 4: Open the folder in VS Code
1. Open VS Code.
2. Click **File > Open Folder**, choose the `ev-charge-predictor` folder, click **Select Folder**.
3. If a box asks "Do you trust the authors?", click **Yes, I trust the authors**.
4. If a box suggests installing the **Python** extension, click **Install**.

## Step 5: Open the terminal
Click **Terminal > New Terminal** in the top menu. A black or blue panel opens at the bottom. This is where you paste commands. Press **Enter** after each one.

## Step 6: Install the libraries (one time only)
Paste this and press Enter. It downloads for a few minutes. Wait until you see a new empty line.

```
pip install -r requirements.txt
```

If it says `pip is not recognized`, use this instead:

```
py -m pip install -r requirements.txt
```

## Step 7: Start the app
Choose **one** of the two versions.

**Streamlit version (simplest):**
```
streamlit run streamlit_app.py
```
Your browser opens by itself. If Streamlit asks for an email, just press Enter.

**Flask version (website style):**
```
python app.py
```
Then open your browser and type **http://127.0.0.1:5000** in the address bar.

If it says `python is not recognized`, use `py app.py`, and for Streamlit use `py -m streamlit run streamlit_app.py`.

## Step 8: Stop the app
Click in the terminal and press **Ctrl + C**. To run it again later, repeat Steps 4, 5 and 7 only. Step 6 is a one-time job.

## If something goes wrong

| What you see | What to do |
|---|---|
| `python is not recognized` | Use `py` instead of `python`. If that fails too, reinstall Python and tick "Add python.exe to PATH". Then close VS Code and open it again. |
| `pip is not recognized` | Use `py -m pip install -r requirements.txt` |
| `No module named flask` (or `streamlit`, `sklearn`) | Step 6 did not finish. Run it again. |
| `can't open file ... app.py` | The terminal is in the wrong folder. Close it, make sure the folder is opened in VS Code (Step 4), and open a new terminal. |
| `Address already in use` | An older copy is still running. Press Ctrl + C in its terminal, or close VS Code and reopen it. |
| The page opens but looks plain | You need internet for the fonts. The app still works without them. |
| Windows blocks a script | Use the commands above without any virtual environment. You do not need one. |

## Mac
1. Install Python from python.org/downloads (open the `.pkg` file and click Continue).
2. Install VS Code from code.visualstudio.com and drag it to Applications.
3. Follow Steps 3 to 5 the same way.
4. In the terminal use `pip3 install -r requirements.txt`, then `python3 app.py` or `python3 -m streamlit run streamlit_app.py`.

## Using the app at your demo
- Move the sliders and pick a charger. Both models update straight away.
- The table shows which model is more accurate. The chart shows each prediction against the real value.
- Lower the Decision Tree depth to show how a simpler tree behaves.
- The data is simulated. Say so when you present it, or replace it with your real dataset (see `README.md`).
