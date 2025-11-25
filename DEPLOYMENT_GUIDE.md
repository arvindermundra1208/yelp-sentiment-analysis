# 🚀 Deployment Guide: Hosting Your Website on GitHub Pages

This guide provides step-by-step instructions to host your Yelp Sentiment Analysis website on GitHub for free.

## ✅ Prerequisites

1.  A **GitHub Account** (Sign up at [github.com](https://github.com) if you don't have one).
2.  Your project files ready:
    *   `index.html`
    *   `styles.css`
    *   `images/` folder (containing all project images)
    *   `report.pdf` (Make sure you have this file!)

---

## 🛠️ Step 1: Create a New Repository

1.  Log in to **GitHub**.
2.  Click the **+** icon in the top-right corner and select **New repository**.
3.  **Repository name**: Enter a name (e.g., `yelp-sentiment-analysis`).
4.  **Public/Private**: Select **Public** (Required for free GitHub Pages).
5.  **Initialize this repository**: Leave all checkboxes **unchecked** (No README, no .gitignore).
6.  Click **Create repository**.

---

## 📤 Step 2: Upload Your Files

Since you may not have Git installed or configured locally, we will use the **Web Interface** (easiest method).

1.  On your new repository page, look for the link that says:
    > *"…or create a new repository on the command line"*
    
    **IGNORE THAT**. Look below it for:
    > *"…or upload an existing file"*
    
    Click the **upload an existing file** link.

2.  **Drag and Drop Files**:
    *   Open your project folder on your computer.
    *   Select **`index.html`**, **`styles.css`**, and **`report.pdf`**.
    *   Drag them into the browser window.

3.  **Upload the Images Folder**:
    *   **Important**: You cannot just drag an empty folder.
    *   Drag the entire **`images`** folder from your computer into the browser window. GitHub should recognize it as a folder and upload the files inside it.
    *   *If dragging the folder doesn't work*:
        1.  Cancel and go back.
        2.  Click "Create new file".
        3.  Type `images/.keep` as the name (this creates the folder).
        4.  Commit new file.
        5.  Go back to "Upload files" and upload all images into that folder.
    *   **Simpler Method**: Just drag the `images` folder. It usually works!

4.  **Commit Changes**:
    *   In the "Commit changes" box at the bottom, type: `Initial deploy`.
    *   Click **Commit changes**.
    *   Wait for the files to process.

---

## ⚙️ Step 3: Enable GitHub Pages

1.  Click on the **Settings** tab (gear icon) at the top of your repository.
2.  In the left sidebar, scroll down and click on **Pages**.
3.  Under **Build and deployment** > **Source**, ensure **Deploy from a branch** is selected.
4.  Under **Branch**, click the dropdown menu (currently "None") and select **`main`** (or `master`).
5.  Click **Save**.

---

## 🌍 Step 4: Access Your Website

1.  Stay on the **Pages** settings page.
2.  Refresh the page after about 1-2 minutes.
3.  You will see a bar at the top saying:
    > *"Your site is live at..."*
4.  Click the link to view your website!

**Your URL will look like:** `https://[your-username].github.io/yelp-sentiment-analysis/`

---

## 🔍 Troubleshooting

*   **Images not showing?**
    *   Ensure your `images` folder is named exactly `images` (lowercase).
    *   Ensure the file names match exactly (e.g., `rating-distribution.png` vs `Rating-Distribution.png`). Linux/Web servers are case-sensitive!
*   **"404 Not Found"?**
    *   Wait a few more minutes. It takes time to build.
    *   Ensure your main file is named exactly `index.html` (lowercase).
*   **"View Full Report" button broken?**
    *   Ensure you uploaded `report.pdf` to the root directory (same place as `index.html`).
