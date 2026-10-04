export default async function handler(req, res) {
  // CORS Headers
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
    return res.status(405).json({ success: false, error: 'Method not allowed' });
  }

  const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
  const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID || 'ix8w90ssxkdyfs';

  if (!RUNPOD_API_KEY) {
    return res.status(500).json({ success: false, error: 'RUNPOD_API_KEY is missing in environment variables.' });
  }

  try {
    const { image } = req.body;

    if (!image) {
      return res.status(400).json({ success: false, error: 'No image provided' });
    }

    const runpodResponse = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${RUNPOD_API_KEY}`
      },
      body: JSON.stringify({ input: { image } })
    });

    const data = await runpodResponse.json();

    if (data.status === 'COMPLETED' && data.output?.model_mesh) {
      return res.status(200).json({
        success: true,
        model_url: data.output.model_mesh
      });
    } else {
      return res.status(500).json({
        success: false,
        error: data.output?.error || '3D mesh generation failed on GPU worker'
      });
    }
  } catch (err) {
    console.error('RunPod Error:', err);
    return res.status(500).json({ success: false, error: 'Failed to connect to RunPod worker' });
  }
  }
