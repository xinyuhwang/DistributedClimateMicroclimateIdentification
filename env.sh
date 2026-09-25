# Environment for local development of this project.
# Usage:  source env.sh
#
# Sets up the venv interpreter, a Spark-compatible JDK, and a CA bundle.
# Everything here is scoped to the shell that sources it — no system changes.

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# --- Java -------------------------------------------------------------------
# Spark 3.5 supports Java 8/11/17. The system default here is Java 23, which
# Spark rejects, so pin Temurin 17 explicitly.
export JAVA_HOME="/Users/haileewang/Library/Java/JavaVirtualMachines/temurin-17.0.13-1/Contents/Home"

# --- Python -----------------------------------------------------------------
# Spark launches workers via PYSPARK_PYTHON. Without this it picks up the
# system python3 (3.13) while the driver runs the venv (3.12), and every job
# dies with PYTHON_VERSION_MISMATCH.
export PYSPARK_PYTHON="$PROJECT_ROOT/venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PROJECT_ROOT/venv/bin/python"

# --- TLS --------------------------------------------------------------------
# The python.org build ships without a configured CA store, so NASA POWER's
# HTTPS endpoints fail cert verification. Point OpenSSL at certifi's bundle.
export SSL_CERT_FILE="$($PROJECT_ROOT/venv/bin/python -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# --- Convenience ------------------------------------------------------------
export PATH="$PROJECT_ROOT/venv/bin:$PATH"

echo "env ready:"
echo "  java    $("$JAVA_HOME/bin/java" -version 2>&1 | head -1 | sed 's/openjdk version //')"
echo "  python  $("$PYSPARK_PYTHON" --version 2>&1 | sed 's/Python //')"
echo "  spark   $("$PYSPARK_PYTHON" -c 'import pyspark; print(pyspark.__version__)' 2>/dev/null)"
