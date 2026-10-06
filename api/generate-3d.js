export default async function handler(req, res) {
  // CORS Headers for Telegram Mini App
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,POST');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Method Not Allowed' });
  }

  // Load and sanitize Vercel environment variables
  const rawEndpointId = process.env.RUNPOD_ENDPOINT_ID || '';
  const rawApiKey = process.env.RUNPOD_API_KEY || '';

  const endpointId = rawEndpointId.replace('Endpoint_ID', '').trim();
  const apiKey = rawApiKey.trim();

  if (!endpointId || !apiKey) {
    return res.status(500).json({ success: false, error: 'Missing RunPod credentials on Vercel' });
  }

  try {
    const body = req.body || {};
    const image = body.image || (body.input && body.input.image);

    if (!image) {
      return res.status(400).json({ success: false, error: 'No image provided in request body' });
    }

    // Exact RunPod API v2 Endpoint URL
    const url = `https://api.runpod.ai/v2/${endpointId}/run`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        input: {
          image: image,
          texture_resolution: body.texture_resolution || 1024,
          remesh_option: body.remesh_option || 'none'
        }
      })
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({ success: false, error: data.error || 'RunPod rejected the request' });
    }

    return res.status(200).json({
      success: true,
      id: data.id,
      status: data.status
    });

  } catch (error) {
    console.error('Generate 3D Handler Error:', error);
    return res.status(500).json({ success: false, error: error.message });
  }
}
