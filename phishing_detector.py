#!/usr/bin/env python3
"""
Phishing Email Detection Model (Scikit-learn)
=============================================

Features
--------
* Trains on a CSV dataset of phishing / legitimate emails (or a built-in
  SYNTHETIC demo dataset if no CSV is given).
* Extracts text features (TF-IDF word + char n-grams) and hand-crafted
  URL / keyword / structure features.
* Compares several classifiers with cross-validation and picks the best.
* Classifies emails as "Phishing" or "Safe".
* Displays accuracy, precision/recall/F1, ROC-AUC and a confusion matrix
  (printed and saved as PNG).
* Saves the trained model and can classify new emails from the CLI.

Usage
-----
  # 1) Demo with synthetic data (for testing the pipeline only)
  python phishing_detector.py

  # 2) Train on a real dataset (CSV with a text column and a label column)
  python phishing_detector.py --data emails.csv --text-col "Email Text" --label-col "Email Type"

  # 3) Classify a new email using the saved model
  python phishing_detector.py --predict "Urgent! Verify your account at http://paypa1-secure.xyz/login"
  python phishing_detector.py --predict-file suspicious_email.txt

Labels accepted in the CSV (case-insensitive):
  phishing: phishing, phishing email, spam, malicious, 1, true, yes
  safe    : safe, safe email, legitimate, ham, benign, 0, false, no

Requirements:  pip install scikit-learn pandas numpy scipy matplotlib joblib

NOTE: Accuracy depends entirely on the dataset. The built-in synthetic
dataset is template-generated and EASY, so its scores say nothing about
real-world performance. Use a real corpus (e.g. Kaggle "Phishing Email
Dataset", Nazario, SpamAssassin, Enron for legitimate mail) for real results.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import warnings
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODEL_PATH = Path("phishing_model.joblib")
CM_PATH = Path("confusion_matrix.png")
LABELS = {0: "Safe", 1: "Phishing"}

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

URL_REGEX = re.compile(r"(https?://[^\s<>\"')]+|www\.[^\s<>\"')]+)", re.IGNORECASE)
IP_HOST_REGEX = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
HTML_TAG_REGEX = re.compile(r"<\s*(html|body|a\s|form|input|script|iframe|img)", re.I)
ANCHOR_REGEX = re.compile(
    r"<a\s[^>]*href=[\"']?([^\"' >]+)[\"']?[^>]*>(.*?)</a>", re.I | re.S
)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy",
}
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work",
    "support", "zip", "country", "loan", "rest", "icu", "buzz", "monster",
}
URGENCY_WORDS = [
    "urgent", "immediately", "act now", "asap", "within 24 hours", "expires",
    "final notice", "last chance", "suspended", "locked", "limited time",
    "action required", "right away", "deadline", "warning",
]
CREDENTIAL_WORDS = [
    "password", "verify", "confirm your", "login", "log in", "sign in",
    "credentials", "account", "username", "ssn", "social security", "pin",
    "otp", "security code", "bank", "credit card", "update your",
]
MONEY_WORDS = [
    "winner", "won", "lottery", "prize", "reward", "refund", "gift card",
    "free", "bonus", "inheritance", "million", "wire transfer", "bitcoin",
    "invoice", "payment", "claim",
]
GENERIC_GREETINGS = ["dear customer", "dear user", "dear member", "dear client",
                     "dear sir", "valued customer", "dear account holder"]
BRANDS = ["paypal", "amazon", "microsoft", "apple", "google", "netflix", "bank",
          "facebook", "instagram", "dhl", "fedex", "irs", "hdfc", "sbi", "icici"]


def extract_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?") for u in URL_REGEX.findall(text or "")]


def _host(url: str) -> str:
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _count_phrases(text_lower: str, phrases: list[str]) -> int:
    return sum(text_lower.count(p) for p in phrases)


FEATURE_NAMES = [
    "n_urls", "n_unique_domains", "has_ip_url", "has_shortener", "has_at_in_url",
    "has_suspicious_tld", "max_url_len", "max_subdomains", "n_hyphen_domain",
    "n_http_not_https", "max_digits_in_host", "url_has_login_kw", "brand_in_url_host",
    "n_urgency", "n_credential", "n_money", "generic_greeting",
    "n_exclaim", "n_question", "upper_ratio", "digit_ratio", "text_len",
    "n_words", "has_html", "anchor_mismatch", "n_punycode", "has_attachment_kw",
    "n_dollar_signs",
]


def extract_features(text: str) -> list[float]:
    """Return a numeric feature vector (same order as FEATURE_NAMES)."""
    text = text if isinstance(text, str) else ""
    low = text.lower()
    urls = extract_urls(text)
    hosts = [_host(u) for u in urls]
    hosts = [h for h in hosts if h]

    has_ip = any(IP_HOST_REGEX.match(h) for h in hosts)
    has_short = any(h in SHORTENERS or h.replace("www.", "") in SHORTENERS for h in hosts)
    has_at = any("@" in u.split("//", 1)[-1].split("/", 1)[0] for u in urls)
    sus_tld = any(h.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS for h in hosts if "." in h)
    max_len = max((len(u) for u in urls), default=0)
    max_sub = max((max(h.count(".") - 1, 0) for h in hosts), default=0)
    n_hyphen = sum(h.count("-") for h in hosts)
    n_http = sum(1 for u in urls if u.lower().startswith("http://"))
    max_digits = max((sum(c.isdigit() for c in h) for h in hosts), default=0)
    login_kw = sum(
        any(k in u.lower() for k in ("login", "verify", "secure", "account", "update", "signin", "confirm"))
        for u in urls
    )
    # brand name inside host but host is not the brand's real domain (e.g. paypal-secure.xyz)
    brand_in_host = 0
    for h in hosts:
        for b in BRANDS:
            if b in h and not re.search(rf"(^|\.){b}\.(com|in|co\.uk|org|net)$", h):
                brand_in_host += 1
                break

    # Anchor-text vs href mismatch: <a href="http://evil">http://paypal.com</a>
    mismatch = 0
    for href, label in ANCHOR_REGEX.findall(text):
        label_urls = extract_urls(label)
        if label_urls and _host(label_urls[0]) != _host(href):
            mismatch += 1

    letters = [c for c in text if c.isalpha()]
    upper_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    digit_ratio = (sum(c.isdigit() for c in text) / len(text)) if text else 0.0
    words = text.split()

    return [
        len(urls), len(set(hosts)), int(has_ip), int(has_short), int(has_at),
        int(sus_tld), max_len, max_sub, n_hyphen,
        n_http, max_digits, login_kw, brand_in_host,
        _count_phrases(low, URGENCY_WORDS), _count_phrases(low, CREDENTIAL_WORDS),
        _count_phrases(low, MONEY_WORDS), int(any(g in low for g in GENERIC_GREETINGS)),
        text.count("!"), text.count("?"), upper_ratio, digit_ratio, len(text),
        len(words), int(bool(HTML_TAG_REGEX.search(text))), mismatch,
        low.count("xn--"), int(any(k in low for k in ("attached", "attachment", ".exe", ".zip", ".scr", "enable macros"))),
        text.count("$"),
    ]


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer: raw text -> dense numeric feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X).fillna("").astype(str)
        arr = np.array([extract_features(t) for t in X], dtype=float)
        # compress heavy-tailed counts so scaling behaves
        return np.log1p(np.abs(arr)) * np.sign(arr)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)


def clean_for_tfidf(text: str) -> str:
    """Normalise text: replace URLs / emails / numbers with tokens (keeps signal, cuts noise)."""
    text = text if isinstance(text, str) else ""
    text = re.sub(r"<[^>]+>", " ", text)                      # strip HTML tags
    text = URL_REGEX.sub(lambda m: " URLTOKEN " + " ".join(re.split(r"[./\-_?=&:]+", _host(m.group(0)))) + " ", text)
    text = re.sub(r"\S+@\S+", " EMAILTOKEN ", text)
    text = re.sub(r"\d+", " 0 ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def build_preprocessor() -> ColumnTransformer:
    word_tfidf = TfidfVectorizer(
        preprocessor=clean_for_tfidf, ngram_range=(1, 2), min_df=2, max_df=0.95,
        sublinear_tf=True, max_features=30000, stop_words="english",
    )
    char_tfidf = TfidfVectorizer(
        preprocessor=clean_for_tfidf, analyzer="char_wb", ngram_range=(3, 5),
        min_df=3, sublinear_tf=True, max_features=30000,
    )
    hand = Pipeline([("feat", HandcraftedFeatures()), ("scale", StandardScaler())])
    return ColumnTransformer(
        [("word", word_tfidf, "text"), ("char", char_tfidf, "text"), ("hand", hand, "text")],
        sparse_threshold=0.3,
    )


def candidate_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
        "Linear SVM (calibrated)": CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced"), cv=3),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced_subsample",
            random_state=RANDOM_STATE),
    }


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

PHISH_LABELS = {"phishing", "phishing email", "phish", "spam", "malicious", "fraud", "1", "true", "yes"}
SAFE_LABELS = {"safe", "safe email", "legitimate", "legit", "ham", "benign", "normal", "0", "false", "no"}


def _normalise_label(v) -> int | None:
    s = str(v).strip().lower()
    if s in PHISH_LABELS:
        return 1
    if s in SAFE_LABELS:
        return 0
    try:
        f = float(s)
        return 1 if f == 1 else 0 if f == 0 else None
    except ValueError:
        return None


def load_csv(path: str, text_col: str | None, label_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding_errors="ignore")
    cols_lower = {c.lower(): c for c in df.columns}

    if text_col is None:
        for guess in ("email text", "text", "body", "email", "message", "content", "email_text"):
            if guess in cols_lower:
                text_col = cols_lower[guess]
                break
    if label_col is None:
        for guess in ("email type", "label", "class", "type", "category", "target", "is_phishing"):
            if guess in cols_lower:
                label_col = cols_lower[guess]
                break
    if text_col is None or label_col is None:
        sys.exit(f"Could not detect columns. Found: {list(df.columns)}. "
                 f"Pass --text-col and --label-col.")

    # Optional subject column gets prepended to the body
    subj = cols_lower.get("subject")
    body = df[text_col].fillna("").astype(str)
    if subj and subj != text_col:
        body = df[subj].fillna("").astype(str) + "\n" + body

    out = pd.DataFrame({"text": body, "label": df[label_col].map(_normalise_label)})
    before = len(out)
    out = out.dropna().astype({"label": int})
    out = out[out["text"].str.strip().str.len() > 0].drop_duplicates(subset="text")
    print(f"Loaded {len(out)} usable rows from {path} (dropped {before - len(out)} "
          f"empty / duplicate / unlabeled).")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synthetic demo data (TEMPLATE-GENERATED — for pipeline testing only)
# --------------------------------------------------------------------------- #

def make_synthetic_dataset(n_per_class: int = 800, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = random.Random(seed)
    brands = ["PayPal", "Amazon", "Microsoft", "Apple", "Netflix", "HDFC Bank", "SBI", "DHL"]
    names = ["Priya", "Arun", "Meena", "Karthik", "Divya", "Rahul", "Sneha", "Vikram", "Anita"]
    topics = ["project review", "lab schedule", "quarterly report", "team lunch", "workshop",
              "assignment submission", "meeting notes", "invoice #{n} for the services",
              "course registration", "travel itinerary", "code review"]
    tlds_bad = ["xyz", "top", "tk", "click", "link", "icu"]
    verbs = ["verify", "confirm", "update", "restore", "unlock", "secure"]
    # Shared real-looking links so URLs alone do not leak the label
    good_domains = ["github.com", "docs.google.com", "zoom.us", "college.edu", "company.com",
                    "linkedin.com", "stackoverflow.com"]

    def phish() -> str:
        b = rng.choice(brands)
        slug = b.lower().replace(" ", "")
        style = rng.randint(0, 5)
        # a minority of phishing emails use ordinary-looking links (harder cases)
        if rng.random() < 0.25:
            url = f"https://{rng.choice(good_domains)}/{rng.choice(verbs)}"
        else:
            url = rng.choice([
                f"http://{slug}-{rng.choice(verbs)}.{rng.choice(tlds_bad)}/login?id={rng.randint(1000, 99999)}",
                f"http://{rng.randint(11, 220)}.{rng.randint(1, 250)}.{rng.randint(1, 250)}.{rng.randint(1, 250)}/{slug}/signin",
                f"http://bit.ly/{rng.choice(['3xYz', 'Ab9Q', 'zK2m'])}{rng.randint(10, 99)}",
                f"http://secure.{slug}.account-{rng.choice(verbs)}.{rng.choice(tlds_bad)}/",
            ])
        if style == 0:
            return (f"Subject: URGENT: Your {b} account has been suspended\n\nDear Customer,\n\n"
                    f"We detected unusual activity. You must {rng.choice(verbs)} your account "
                    f"immediately or it will be locked within 24 hours.\nClick here: {url}\n\n{b} Security Team")
        if style == 1:
            return (f"Subject: Action Required - payment failed\n\nDear user, your {b} payment could not be "
                    f"processed. Update your credit card details and password now: {url}\nFailure to act will "
                    f"result in permanent suspension!!!")
        if style == 2:
            return (f"Subject: Congratulations! You are a WINNER\n\nYou won a ${rng.randint(5, 900)},000 prize "
                    f"in our lottery. Claim your reward before it expires: {url}\nSend your bank account number "
                    f"and SSN to receive the transfer.")
        if style == 3:
            return (f"Subject: Security alert\n\n<html><body><p>We noticed a login to your {b} account from a "
                    f"new device.</p><a href=\"{url}\">https://www.{slug}.com/security</a> "
                    f"to verify your identity and enter the OTP.</body></html>")
        if style == 4:
            return (f"Subject: Invoice attached\n\nHello,\nPlease find the attached invoice. Enable macros to "
                    f"view the document. If you do not recognise this charge, log in at {url} to confirm your "
                    f"password. Reply immediately.")
        return (f"Subject: Final notice from {b}\n\nYour account is limited. Confirm your login credentials, "
                f"security code and date of birth here {url} to avoid closure. This is your last chance.")

    def safe() -> str:
        n = rng.choice(names)
        t = rng.choice(topics).format(n=rng.randint(100, 999))
        style = rng.randint(0, 5)
        url = f"https://{rng.choice(good_domains)}/{rng.choice(['docs', 'meet', 'repo', 'login', 'account', 'update'])}/{rng.randint(1, 999)}"
        if style == 0:
            return f"Subject: {t}\n\nHi {n},\n\nPlease find the notes for the {t}. Let me know if you have questions.\nThanks,\n{rng.choice(names)}"
        if style == 1:
            return (f"Subject: Meeting tomorrow\n\nHi team,\n\nReminder about tomorrow's {t} at "
                    f"{rng.randint(9, 17)}:00. The agenda is at {url}.\nBest regards,\n{n}")
        if style == 2:
            return (f"Subject: Your order has shipped\n\nHello {n},\nYour order #{rng.randint(10000, 99999)} has shipped and "
                    f"will arrive in 3-5 days. Track it at {url}. Thank you for shopping with us.")
        if style == 3:
            return (f"Subject: Re: {t}\n\nThanks {n}, that works for me. I've updated the document, please review it "
                    f"and sign in to the shared folder if you want to comment: {url}")
        if style == 4:
            return (f"Subject: Account statement available\n\nHi {n}, your monthly statement is ready. You can log in "
                    f"to your account at {url} whenever convenient. No action is required.")
        return f"Subject: Lunch?\n\nHey {n}, are you free for lunch on Friday? We could try the new place near campus.\n- {rng.choice(names)}"

    rows = [(phish(), 1) for _ in range(n_per_class)] + [(safe(), 0) for _ in range(n_per_class)]
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #

def evaluate(model, X_test, y_test, title: str):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 64)
    print(f"TEST-SET RESULTS — {title}")
    print("=" * 64)
    print(f"Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}   (of emails flagged Phishing, how many were)")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}   (of real phishing emails, how many caught)")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    if hasattr(model, "predict_proba"):
        print(f"ROC-AUC  : {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=["Actual Safe", "Actual Phishing"],
                       columns=["Pred Safe", "Pred Phishing"]).to_string())
    print(f"\nTN={tn}  FP={fp} (safe mail wrongly flagged)  FN={fn} (phishing missed)  TP={tp}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"], digits=4))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=["Safe", "Phishing"]).plot(
            ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(f"Confusion Matrix — {title}\nAccuracy {acc * 100:.2f}%")
        fig.tight_layout()
        fig.savefig(CM_PATH, dpi=150)
        plt.close(fig)
        print(f"Confusion-matrix image saved to: {CM_PATH.resolve()}")
    except Exception as exc:  # plotting must never break training
        print(f"(Could not save confusion matrix image: {exc})")
    return acc


def top_features(model: Pipeline, k: int = 15) -> None:
    """Print most influential features if the final model is linear."""
    clf = model.named_steps["clf"]
    if not isinstance(clf, LogisticRegression):
        return
    try:
        names = model.named_steps["prep"].get_feature_names_out()
    except Exception:
        return
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    print("\nTop indicators of PHISHING:")
    for i in order[::-1][:k]:
        print(f"  +{coefs[i]:.3f}  {names[i]}")
    print("Top indicators of SAFE:")
    for i in order[:k]:
        print(f"  {coefs[i]:.3f}  {names[i]}")


def train(df: pd.DataFrame, test_size: float = 0.2, cv: int = 5):
    if df["label"].nunique() < 2:
        sys.exit("Dataset must contain both phishing and safe examples.")
    print(f"\nClass balance: Safe={int((df.label == 0).sum())}  Phishing={int((df.label == 1).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        df[["text"]], df["label"], test_size=test_size,
        stratify=df["label"], random_state=RANDOM_STATE)

    print("\nCross-validating candidate models (F1, train split only)...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores: dict[str, float] = {}
    for name, clf in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        s = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1", n_jobs=1)
        scores[name] = s.mean()
        print(f"  {name:<26} F1 = {s.mean():.4f} ± {s.std():.4f}")

    best = max(scores, key=scores.get)
    print(f"\nBest model: {best}")
    model = Pipeline([("prep", build_preprocessor()), ("clf", candidate_models()[best])])
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test, best)
    top_features(model)
    return model


def save_model(model) -> None:
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH.resolve()}")


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def explain(text: str) -> list[str]:
    """Human-readable red flags for a single email (rule-based, independent of the model)."""
    f = dict(zip(FEATURE_NAMES, extract_features(text)))
    flags = []
    if f["has_ip_url"]: flags.append("URL uses a raw IP address")
    if f["has_shortener"]: flags.append("URL shortener hides the real destination")
    if f["has_at_in_url"]: flags.append("'@' inside a URL (can disguise the real host)")
    if f["has_suspicious_tld"]: flags.append("link uses a high-risk top-level domain")
    if f["brand_in_url_host"]: flags.append("brand name used in a non-official domain")
    if f["n_http_not_https"]: flags.append("link is not HTTPS")
    if f["anchor_mismatch"]: flags.append("link text does not match the real link target")
    if f["max_subdomains"] >= 3: flags.append("unusually many subdomains in a URL")
    if f["n_urgency"]: flags.append(f"{int(f['n_urgency'])} urgency phrase(s)")
    if f["n_credential"] >= 2: flags.append("asks for credentials / account verification")
    if f["n_money"]: flags.append("money / prize / payment language")
    if f["generic_greeting"]: flags.append("generic greeting (e.g. 'Dear Customer')")
    if f["has_attachment_kw"]: flags.append("mentions attachments / macros")
    if f["n_punycode"]: flags.append("punycode (xn--) domain, possible lookalike")
    return flags


def predict_text(model, text: str) -> None:
    X = pd.DataFrame({"text": [text]})
    pred = int(model.predict(X)[0])
    print("\n" + "-" * 64)
    print(f"Prediction : {LABELS[pred].upper()}")
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
        print(f"Confidence : Phishing {p[1] * 100:.1f}%  |  Safe {p[0] * 100:.1f}%")
    urls = extract_urls(text)
    if urls:
        print("URLs found :", ", ".join(urls[:5]) + (" ..." if len(urls) > 5 else ""))
    flags = explain(text)
    if flags:
        print("Red flags  :")
        for fl in flags:
            print("   -", fl)
    print("-" * 64)
    print("This is a statistical prediction, not a guarantee. Verify suspicious mail through official channels.")


def interactive(model) -> None:
    print("\nPaste an email and finish with an empty line + Ctrl-D (Ctrl-Z then Enter on Windows). Ctrl-C to quit.")
    try:
        predict_text(model, sys.stdin.read())
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Phishing email detector (scikit-learn)")
    ap.add_argument("--data", help="CSV dataset path (omit to use synthetic demo data)")
    ap.add_argument("--text-col", help="Name of the email text column")
    ap.add_argument("--label-col", help="Name of the label column")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    ap.add_argument("--predict", help="Classify this email text with the saved model")
    ap.add_argument("--predict-file", help="Classify the email in this text file")
    ap.add_argument("--interactive", action="store_true", help="Read an email from stdin and classify it")
    ap.add_argument("--retrain", action="store_true", help="Force retraining even if a saved model exists")
    args = ap.parse_args()

    wants_prediction = args.predict or args.predict_file or args.interactive
    if wants_prediction and MODEL_PATH.exists() and not args.data and not args.retrain:
        model = joblib.load(MODEL_PATH)
    else:
        if args.data:
            df = load_csv(args.data, args.text_col, args.label_col)
        else:
            print("No --data given: using a SYNTHETIC template-generated dataset (demo only).")
            print("Scores on it do NOT reflect real-world accuracy.\n")
            df = make_synthetic_dataset()
        model = train(df, args.test_size, args.cv)
        save_model(model)

    if args.predict:
        predict_text(model, args.predict)
    if args.predict_file:
        predict_text(model, Path(args.predict_file).read_text(encoding="utf-8", errors="ignore"))
    if args.interactive:
        interactive(model)

    if not wants_prediction:
        print("\nTry:  python phishing_detector.py --predict \"Verify your account at http://paypa1-login.xyz\"")


if __name__ == "__main__":
    main()
