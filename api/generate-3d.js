export default async function handler(req, res) {
  // 1. Configure standard CORS headers to allow your Telegram Mini App context
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

  // 2. Handle standard browser preflight options request checks
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Method Not Allowed' });
  }

  // 3. Extract your validated environment configuration tokens
  const endpointId = process.env.RUNPOD_ENDPOINT_ID;
  const apiKey = process.env.RUNPOD_API_KEY;

  if (!endpointId) {
    return res.status(500).json({ success: false, error: 'RUNPOD_ENDPOINT_ID environment variable is missing on Vercel' });
  }
  if (!apiKey) {
    return res.status(500).json({ success: false, error: 'RUNPOD_API_KEY environment variable is missing on Vercel' });
  }

  try {
    const body = req.body || {};
    const input = body.input || body;

    // 4. FIX: Fixed absolute URL construction structure 
    const url = "https://api.runpod.ai/v2/" + endpointId.toString().trim() + "/run";
    console.log("Sending request to RunPod:", url);

    // 5. Dispatch payload to your active serverless queue routing layer
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apiKey
      },
      body: JSON.stringify({ input: input })
    });

    const data = await response.json();

    console.log("RunPod response status:", response.status);
    console.log("RunPod response data:", JSON.stringify(data));

    if (response.ok && data.id) {
      return res.status(200).json({
        success: true,
        id: data.id,
        status: data.status
      });
    } else {
      return res.status(response.status).json({
        success: false,
        error: data.error || 'RunPod rejected the request payload infrastructure parameters'
      });
    }

  } catch (err) {
    console.error("Handler error:", err);
    return res.status(500).json({ success: false, error: err.message });
  }
}
