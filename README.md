# DocuBot: AI-Powered Document Query Assistant

## Introduction

This manual provides instructions for setting up and using the AI Assistant program for document queries. The program utilizes AI models to answer user questions based on a collection of documents stored in a database.

## Requirements

- Python 3.x

## Installation

First, you can remove any example PDF inside the `documents` folder.

1. Clone the repository or download the program files from [GitHub](https://github.com/repository-link).

2. Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. Copy all your PDF files into the `documents` folder. Then execute the program:

   ```bash
   python chat-pdf.py
   ```

   If everything is correct, the program will download all the required models.

2. Once it finishes, the program will prompt you to ask a question.

3. Enter your query when prompted.

4. The program will search the document collection and provide relevant answers based on the query.

## Notes

- If you want to reset the database, remove the `database` directory, replace your PDF files in the `documents` directory, and execute the program again. The database will be recreated.
- Ensure that the document files are stored in the `documents` directory with the `.pdf` extension.
- Adjust the model name and parameters according to your requirements in the `chat-pdf.py` file.
- The program is designed to utilize GPU (`'cuda'`). If GPU is not available, modify `device_map` accordingly.
