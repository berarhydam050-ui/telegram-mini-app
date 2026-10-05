export default async function handler(req, res) {
  // 1. Only allow POST requests from your Telegram Mini App
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  // 2. Fetch the credentials safely from Vercel's Environment Variables
  const endpointId = process.env.RUNPOD_ENDPOINT_ID;
  const apiKey = process.env.RUNPOD_API_KEY;

  if (!endpointId) {
    return res.status(500).json({ error: "RUNPOD_ENDPOINT_ID is missing on Vercel" });
  }
  if (!apiKey) {
    return res.status(500).json({ error: "RUNPOD_API_KEY is missing on Vercel" });
  }

  try {
    const body = req.body || {};
    
    // 3. Extract the inner data object if the body arrives pre-wrapped in "input"
    const input = body.input || body;

    // 4. Safe URL creation using simple string concatenation to avoid template errors
    const url = "https://runpod.ai" + endpointId + "/run";
    console.log("Sending request to RunPod:", url);

    // 5. Fire request using Vercel's native fetch utility
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": "Bearer " + apiKey,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ input: input })
    });

    // 6. Read and safely parse the network response stream
    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }

    console.log("RunPod HTTP status:", response.status);
    console.log("RunPod response:", data);

    // 7. Pass the RunPod gateway response back up to the frontend UI
    return res.status(response.status).json(data);

  } catch (error) {
    console.error("RunPod request error:", error);
    return res.status(500).json({ error: "Failed to call RunPod", details: error.message });
  }
}
