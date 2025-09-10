import tkinter as tk
import nltk
from textblob import TextBlob
from newspaper import Article

def Sumarize():
    url = utext.get("1.0", tk.END).strip()
    article = Article(url)

    article.download()
    article.parse()
    article.nlp()

    # Enable editing
    title.config(state='normal')
    author.config(state='normal')
    publication.config(state='normal')
    summary.config(state='normal')
    sentiment.config(state='normal')

    # Clear old content
    title.delete("1.0", "end")
    author.delete("1.0", "end")
    publication.delete("1.0", "end")
    summary.delete("1.0", "end")
    sentiment.delete("1.0", "end")

    # Insert new content
    title.insert("1.0", article.title)
    author.insert("1.0", ", ".join(article.authors) if article.authors else "N/A")
    publication.insert("1.0", str(article.publish_date) if article.publish_date else "N/A")
    summary.insert("1.0", article.summary)

    analysis = TextBlob(article.text) 
    polarity = analysis.sentiment.polarity
    sentiment.insert(
        '1.0',
        f"Polarity: {polarity:.3f} | Sentiment: "
        f"{'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'}"
    )

    # Disable editing again
    title.config(state='disabled')
    author.config(state='disabled')
    publication.config(state='disabled')
    summary.config(state='disabled')
    sentiment.config(state='disabled')


root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("600x400")

tlabel = tk.Label(root, text="Sentiment Analysis of News Article", font=("Helvetica", 16))
tlabel.pack()

title = tk.Text(root, height=1, width=140)
title.config(state='disabled', bg='#dddddd')
title.pack()

alabel = tk.Label(root, text="Author", font=("Helvetica", 16))
alabel.pack()

author = tk.Text(root, height=1, width=140)
author.config(state='disabled', bg='#dddddd')
author.pack()

plabel = tk.Label(root, text="Publication Date", font=("Helvetica", 16))
plabel.pack()

publication = tk.Text(root, height=1, width=140)
publication.config(state='disabled', bg='#dddddd')
publication.pack()

slabel = tk.Label(root, text="Summary", font=("Helvetica", 16))
slabel.pack()

summary = tk.Text(root, height=20, width=140)
summary.config(state='disabled', bg='#dddddd')
summary.pack()

selabel = tk.Label(root, text="Sentiment Analysis", font=("Helvetica", 16))
selabel.pack()

sentiment = tk.Text(root, height=1, width=140)
sentiment.config(state='disabled', bg='#dddddd')
sentiment.pack()

ulabel = tk.Label(root, text="Url")
ulabel.pack()

utext = tk.Text(root, height=1, width=140)
utext.pack()

btn = tk.Button(root, text="Summarize", command=Sumarize)
btn.pack()

root.mainloop()
