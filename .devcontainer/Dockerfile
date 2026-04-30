FROM python:3.11-slim

# -- Proxy d'entreprise --
# Valeurs par defaut adaptees au reseau SOMEI. Surchargeable via --build-arg
# ou via les build.args du devcontainer.json / docker-compose.yml.
ARG HTTP_PROXY=http://pxy-smi-user.somei.intranet:8080
ARG HTTPS_PROXY=http://pxy-smi-user.somei.intranet:8080
ARG NO_PROXY=localhost,127.0.0.1,.somei.intranet

# Variables d'environnement pour pip / apt / outils Python.
# On expose les deux casses (UPPER et lower) car certains outils (curl, requests)
# n'utilisent que la version minuscule.
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY:-localhost,127.0.0.1,.local} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY:-localhost,127.0.0.1,.local} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Configuration apt pour respecter le proxy si fourni.
RUN if [ -n "$HTTP_PROXY" ]; then \
        echo "Acquire::http::Proxy \"$HTTP_PROXY\";" > /etc/apt/apt.conf.d/00proxy && \
        echo "Acquire::https::Proxy \"${HTTPS_PROXY:-$HTTP_PROXY}\";" >> /etc/apt/apt.conf.d/00proxy ; \
    fi

# Dependances systeme minimales :
# - libgomp1 : runtime OpenMP requis par lightgbm/xgboost/scikit-learn
# - git/curl : utilitaires de dev
# - ca-certificates : certificats TLS
# - unzip : requis par Reflex pour installer Bun (POC reflex_poc/)
# On evite volontairement build-essential car (1) toutes nos deps Python
# ont des wheels pre-compilees pour cp311 linux, (2) le proxy SOMEI bloque
# certains paquets de la chaine gcc (libstdc++-14-dev en 403).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        git \
        curl \
        ca-certificates \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Installation des dependances Python (cachee tant que requirements.txt ne change pas).
# --only-binary=:all: force l'usage des wheels et echoue vite si un paquet
# tente une compilation source (ce qui necessiterait gcc).
WORKDIR /workspaces/MiniKu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --only-binary=:all: -r /tmp/requirements.txt

EXPOSE 8501

# Commande par defaut si on lance le conteneur en standalone (hors VS Code).
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
