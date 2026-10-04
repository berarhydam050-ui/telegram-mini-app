export default async function handler(req, res) {
  // Support CORS headers for Telegram Web App
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
  );

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Method Not Allowed' });
  }

  try {
    const { image } = req.body || {};

    if (!image) {
      return res.status(400).json({ success: false, error: 'No image provided in request body' });
    }

    const apiKey = process.env.RUNPOD_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ success: false, error: 'RUNPOD_API_KEY environment variable is missing on Vercel' });
    }

    // Updated with your new RunPod Endpoint ID
    const runpodEndpoint = 'https://api.runpod.ai/v2/6iz1l1u676vw6o/run';

    const response = await fetch(runpodEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        input: {
          image: image,
          texture_resolution: 1024,
          remesh_option: 'none'
        }
      })
    });

    const data = await response.json();
    
    // Log the raw RunPod response to Vercel console for debugging
    console.log("RunPod response status:", response.status);
    console.log("RunPod raw response data:", JSON.stringify(data));

    if (response.ok && data.id) {
      return res.status(200).json({
        success: true,
        id: data.id,
        status: data.status
      });
    } else {
      return res.status(500).json({
        success: false,
        error: data.error || data.message || `RunPod rejected request with status ${response.status}`
      });
    }

  } catch (err) {
    console.error("Handler error:", err);
    return res.status(500).json({ success: false, error: err.message });
  }
}
