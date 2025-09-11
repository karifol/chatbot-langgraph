curl -X 'POST' \
  'http://localhost:8080/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
        "messages": [
          {
            "role": "user",
            "content": "東京の天気を教えて"
          }
        ]
      }'

