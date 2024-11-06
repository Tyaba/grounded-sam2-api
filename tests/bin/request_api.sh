#!/usr/bin/env bash
set -eu

# サービスのURLを取得
URL=$(gcloud run services describe $SERVICE_NAME \
  --region=us-central1 \
  --project=$GOOGLE_CLOUD_PROJECT \
  --format='value(status.url)')

# IDトークンを取得
TOKEN=$(gcloud auth print-identity-token)

# detectエンドポイントのテスト
detect_response=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "サンプルテキスト", "image": "base64エンコードされた画像データ"}' \
  "$URL/detect")

# segmentエンドポイントのテスト
segment_response=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "サンプルテキスト", "image": "base64エンコードされた画像データ"}' \
  "$URL/segment")

# レスポンスコードを確認
if [ "$detect_response" -eq 200 ] && [ "$segment_response" -eq 200 ]; then
  echo "両方のAPIテストが成功しました"
  echo "detect endpoint: $detect_response"
  echo "segment endpoint: $segment_response"
  exit 0
else
  echo "APIテストが失敗しました"
  echo "detect endpoint: $detect_response"
  echo "segment endpoint: $segment_response"
  exit 1
fi