const express = require('express');
const cors = require('cors');

const app = express();

// Enable CORS so your frontend index.html can call this server
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Safely pull secrets from environment variables
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID || 'ix8w90ssxkdyfs';

app.post('/api/generate-3d', async (req, res) => {
  try {
    const { image } = req.body; // Base64 image string sent from index.html

    if (!image) {
      return res.status(400).json({ error: "No image provided" });
    }

    // Call RunPod API securely using backend environment variables
    const runpodResponse = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${RUNPOD_API_KEY}`
      },
      body: JSON.stringify({
        input: { image: image }
      })
    });

    const data = await runpodResponse.json();

    if (data.status === "COMPLETED" && data.output?.model_mesh) {
      return res.json({
        success: true,
        model_url: data.output.model_mesh
      });
    } else {
      return res.status(500).json({
        error: data.output?.error || "3D mesh generation failed on GPU worker"
      });
    }

  } catch (err) {
    console.error("RunPod Execution Error:", err);
    res.status(500).json({ error: "Failed to connect to backend server or RunPod worker" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Backend live on port ${PORT}`));
