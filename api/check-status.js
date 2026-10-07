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

  const jobId = req.body.jobId || req.body;

  if (!jobId) {
    return res.status(400).json({ success: false, error: 'Missing jobId in request body' });
  }

  try {
    // Exact RunPod Status Check URL
    const url = `https://api.runpod.ai/v2/${endpointId}/status/${jobId}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });

    const data = await response.json();

    return res.status(200).json({
      success: true,
      status: data.status, // IN_QUEUE, IN_PROGRESS, COMPLETED, FAILED
      output: data.output || null,
      error: data.error || null
    });

  } catch (error) {
    console.error('Status Check Handler Error:', error);
    return res.status(500).json({ success: false, error: error.message });
  }
}
