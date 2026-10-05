export default async function handler(req, res) {
  // CORS for Telegram Mini App
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader(
    "Access-Control-Allow-Methods",
    "GET,OPTIONS,PATCH,DELETE,POST,PUT"
  );
  res.setHeader(
    "Access-Control-Allow-Headers",
    "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version"
  );

  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  // Only POST is allowed
  if (req.method !== "POST") {
    return res.status(405).json({
      success: false,
      error: "Method Not Allowed"
    });
  }

  try {
    // Get image from request
    const { image } = req.body || {};

    if (!image) {
      return res.status(400).json({
        success: false,
        error: "No image provided in request body"
      });
    }

    // Get RunPod API key from Vercel Environment Variables
    const apiKey = process.env.RUNPOD_API_KEY;

    if (!apiKey) {
      return res.status(500).json({
        success: false,
        error: "RUNPOD_API_KEY environment variable is missing on Vercel"
      });
    }

    // Get RunPod Endpoint ID from Vercel Environment Variables
    const endpointId = process.env.RUNPOD_ENDPOINT_ID;

    if (!endpointId) {
      return res.status(500).json({
        success: false,
        error: "RUNPOD_ENDPOINT_ID environment variable is missing on Vercel"
      });
    }

    // RunPod Serverless endpoint
    const runpodEndpoint =
      `https://api.runpod.ai/v2/${endpointId}/run`;

    console.log("Sending request to RunPod:", runpodEndpoint);

    // Send generation request to RunPod
    const response = await fetch(runpodEndpoint, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },

      body: JSON.stringify({
        input: {
          image: image,
          texture_resolution: 1024,
          remesh_option: "none"
        }
      })
    });

    // Read RunPod response
    const data = await response.json();

    console.log("RunPod HTTP status:", response.status);
    console.log(
      "RunPod response:",
      JSON.stringify(data)
    );

    // RunPod rejected the request
    if (!response.ok) {
      return res.status(response.status).json({
        success: false,
        error:
          data.error ||
          data.message ||
          `RunPod returned HTTP ${response.status}`,
        details: data
      });
    }

    // Request successfully submitted
    return res.status(200).json({
      success: true,
      id: data.id || null,
      status: data.status || "IN_PROGRESS",
      message: "3D generation job submitted successfully"
    });

  } catch (error) {
    console.error("Handler Error:", error);

    return res.status(500).json({
      success: false,
      error: error.message || "Internal server error"
    });
  }
}
