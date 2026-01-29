#!/bin/bash
#
# Generate TLS certificates for KETI GPU Webhook
#
# Usage: ./generate-certs.sh [namespace]
#

set -e

# Fix OpenSSL config path issue (for conda environments)
if [ ! -f "${OPENSSL_CONF:-/dev/null}" ]; then
    if [ -f "/etc/ssl/openssl.cnf" ]; then
        export OPENSSL_CONF="/etc/ssl/openssl.cnf"
    elif [ -f "/etc/pki/tls/openssl.cnf" ]; then
        export OPENSSL_CONF="/etc/pki/tls/openssl.cnf"
    fi
fi

NAMESPACE="${1:-edge-system}"
SERVICE="keti-gpu-webhook"
SECRET_NAME="keti-gpu-webhook-certs"
WEBHOOK_CONFIG="keti-gpu-webhook"

CERT_DIR=$(mktemp -d)
echo "Generating certificates in $CERT_DIR"

# Generate CA
openssl genrsa -out ${CERT_DIR}/ca.key 2048
openssl req -x509 -new -nodes -key ${CERT_DIR}/ca.key \
    -subj "/CN=KETI GPU Webhook CA" \
    -days 3650 -out ${CERT_DIR}/ca.crt

# Generate server key and CSR
openssl genrsa -out ${CERT_DIR}/server.key 2048

cat > ${CERT_DIR}/csr.conf <<EOF
[req]
req_extensions = v3_req
distinguished_name = req_distinguished_name
[req_distinguished_name]
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names
[alt_names]
DNS.1 = ${SERVICE}
DNS.2 = ${SERVICE}.${NAMESPACE}
DNS.3 = ${SERVICE}.${NAMESPACE}.svc
DNS.4 = ${SERVICE}.${NAMESPACE}.svc.cluster.local
EOF

openssl req -new -key ${CERT_DIR}/server.key \
    -subj "/CN=${SERVICE}.${NAMESPACE}.svc" \
    -out ${CERT_DIR}/server.csr \
    -config ${CERT_DIR}/csr.conf

# Sign server certificate with CA
openssl x509 -req -in ${CERT_DIR}/server.csr \
    -CA ${CERT_DIR}/ca.crt -CAkey ${CERT_DIR}/ca.key \
    -CAcreateserial -out ${CERT_DIR}/server.crt \
    -days 365 -extensions v3_req -extfile ${CERT_DIR}/csr.conf

echo ""
echo "Certificates generated successfully!"
echo ""

# Create or update Kubernetes secret
echo "Creating/updating secret ${SECRET_NAME} in namespace ${NAMESPACE}..."
kubectl create secret tls ${SECRET_NAME} \
    --cert=${CERT_DIR}/server.crt \
    --key=${CERT_DIR}/server.key \
    -n ${NAMESPACE} \
    --dry-run=client -o yaml | kubectl apply -f -

# Get CA bundle (base64 encoded)
CA_BUNDLE=$(cat ${CERT_DIR}/ca.crt | base64 | tr -d '\n')

echo ""
echo "Patching MutatingWebhookConfiguration with CA bundle..."
kubectl patch mutatingwebhookconfiguration ${WEBHOOK_CONFIG} \
    --type='json' \
    -p="[{'op': 'add', 'path': '/webhooks/0/clientConfig/caBundle', 'value': '${CA_BUNDLE}'}]" \
    2>/dev/null || echo "Note: Webhook configuration not found yet. Apply webhook-config.yaml first, then run this script again."

echo ""
echo "=========================================="
echo "  Certificate Setup Complete!            "
echo "=========================================="
echo ""
echo "CA Bundle (for manual configuration):"
echo "${CA_BUNDLE}"
echo ""

# Cleanup
rm -rf ${CERT_DIR}
