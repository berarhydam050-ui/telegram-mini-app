export default async function handler(req, res) {
  // 1. Only allow POST requests from your Telegram Mini App
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  // 2. Fetch and sanitize environment variables
  const rawEndpointId = process.env.RUNPOD_ENDPOINT_ID || "";
  const rawApiKey = process.env.RUNPOD_API_KEY || "";

  // Strips accidental labels, newlines, or extra spaces
  const endpointId = rawEndpointId.replace("Endpoint_ID", "").trim();
  const apiKey = rawApiKey.trim();

  if (!endpointId) {
    return res.status(500).json({ error: "RUNPOD_ENDPOINT_ID is missing or invalid on Vercel" });
  }

  if (!apiKey) {
    return res.status(500).json({ error: "RUNPOD_API_KEY is missing on Vercel" });
  }

  try {
    const body = req.body || {};

    // 3. Extract inner data object if pre
  
