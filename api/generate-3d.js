export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

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
    
    // Safety check: extract payload correctly if already wrapped inside "input"
    const input = body.input || body;

    // FIX: Switched from /runsync to /run
    const url = `https://runpod.ai{endpointId}/run`;
    console.log("Sending request to RunPod:", url);

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ input: input })
    });

    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }

    console.log("RunPod HTTP status:", response.status);
    console.log("RunPod response:", data);

    return res.status(response.status).json(data);

  } catch (error) {
    console.error("RunPod request error:", error);
    return res.status(500).json({ error: "Failed to call RunPod", details: error.message });
  }
}
