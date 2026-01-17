// api/score.js
const Busboy = require("busboy");
const pdfParse = require("pdf-parse");

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.statusCode = 405;
    return res.end("Method Not Allowed");
  }

  const contentType = req.headers["content-type"] || "";
  if (!contentType.includes("multipart/form-data")) {
    res.statusCode = 400;
    return res.end("Expected multipart/form-data");
  }

  // ---- Parse multipart (fields + file buffer) ----
  const fields = {};
  let resumeBuf = null;
  let resumeName = null;

  try {
    await new Promise((resolve, reject) => {
      const bb = Busboy({
        headers: req.headers,
        limits: {
          files: 1,
          fileSize: 6 * 1024 * 1024 // 6MB
        }
      });

      bb.on("file", (name, file, info) => {
        const { filename, mimeType } = info;
        if (name !== "resume") {
          file.resume();
          return;
        }
        if (mimeType !== "application/pdf") {
          file.resume();
          return reject(new Error("Resume must be a PDF"));
        }
        resumeName = filename;

        const chunks = [];
        file.on("data", (d) => chunks.push(d));
        file.on("limit", () => reject(new Error("PDF too large")));
        file.on("end", () => {
          resumeBuf = Buffer.concat(chunks);
        });
      });

      bb.on("field", (name, val) => {
        fields[name] = val;
      });

      bb.on("error", reject);
      bb.on("finish", resolve);

      req.pipe(bb);
    });
  } catch (e) {
    res.statusCode = 400;
    return res.end(e.message || "Bad Request");
  }

  if (!resumeBuf) {
    res.statusCode = 400;
    return res.end("Missing resume PDF (field name: resume)");
  }

  // ---- Extract PDF text (v1) ----
  let resumeText = "";
  try {
    const parsed = await pdfParse(resumeBuf);
    resumeText = (parsed.text || "").trim();
  } catch (e) {
    res.statusCode = 400;
    return res.end("Could not read PDF text");
  }

  // Optional: light redaction of obvious PII before sending to the model
  // (You can disable if you want full fidelity.)
  resumeText = redactPII(resumeText);

  // Keep within a sane limit (resumes are short, but guard anyway)
  resumeText = clipText(resumeText, 22000);

  // ---- Call OpenAI (server-side) ----
  try {
    const OpenAI = (await import("openai")).default;
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    const schema = {
      name: "ResumeScore",
      schema: {
        type: "object",
        additionalProperties: false,
        properties: {
          resume_score: { type: "number", minimum: 0, maximum: 10 },
          subscores: {
            type: "object",
            additionalProperties: false,
            properties: {
              math_engagement: { type: "number", minimum: 0, maximum: 10 },
              research_exposition: { type: "number", minimum: 0, maximum: 10 },
              leadership_service: { type: "number", minimum: 0, maximum: 10 },
              initiative_trajectory: { type: "number", minimum: 0, maximum: 10 }
            },
            required: ["math_engagement", "research_exposition", "leadership_service", "initiative_trajectory"]
          },
          strengths: { type: "array", items: { type: "string" }, maxItems: 5 },
          risks_or_gaps: { type: "array", items: { type: "string" }, maxItems: 5 },
          confidence: { type: "string", enum: ["low", "medium", "high"] }
        },
        required: ["resume_score", "subscores", "strengths", "risks_or_gaps", "confidence"]
      }
    };

    const prompt = buildResumePrompt({
      resumeText,
      resumeName,
      context: {
        // You can pass any fields from the form here if you want:
        // e.g., fields.gpa, fields.calc_completed, fields.upper_courses, etc.
      }
    });

    // Responses API (recommended)  [oai_citation:3‡OpenAI Platform](https://platform.openai.com/docs/guides/migrate-to-responses)
    // Structured Outputs for strict JSON schema  [oai_citation:4‡OpenAI Platform](https://platform.openai.com/docs/guides/structured-outputs)
    const response = await client.responses.create({
      model: "gpt-4o-2024-08-06",
      input: prompt,
      text: {
        format: {
          type: "json_schema",
          strict: true,
          ...schema
        }
      }
    });

    const outText = response.output_text || "";
    const json = JSON.parse(outText);

    // Round to 1 decimal for display consistency
    json.resume_score = Math.round(json.resume_score * 10) / 10;
    for (const k of Object.keys(json.subscores || {})) {
      json.subscores[k] = Math.round(json.subscores[k] * 10) / 10;
    }

    res.setHeader("Content-Type", "application/json");
    return res.status(200).end(JSON.stringify(json));
  } catch (e) {
    res.statusCode = 500;
    return res.end(`Scoring failed: ${e.message || "Unknown error"}`);
  }
};

// ---------- helpers ----------
function clipText(s, maxChars) {
  if (!s) return "";
  if (s.length <= maxChars) return s;
  // Keep start + end (end often contains skills/keywords)
  const head = s.slice(0, Math.floor(maxChars * 0.75));
  const tail = s.slice(-Math.floor(maxChars * 0.25));
  return `${head}\n\n[... clipped ...]\n\n${tail}`;
}

function redactPII(s) {
  if (!s) return "";
  return s
    // emails
    .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, "[REDACTED_EMAIL]")
    // phone numbers (rough)
    .replace(/(\+?\d{1,2}\s*)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}/g, "[REDACTED_PHONE]")
    // addresses (very rough heuristic line-based)
    .replace(/^\s*\d{1,6}\s+.*(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln)\b.*$/gim, "[REDACTED_ADDRESS]");
}

function buildResumePrompt({ resumeText, resumeName }) {
  return [
    {
      role: "system",
      content:
        "You are an admissions scoring assistant for a Pi Mu Epsilon chapter. " +
        "Score the RESUME ONLY. Use the rubric. Be fair across backgrounds. " +
        "Do not reward prestige names; reward evidence of mathematical engagement, rigor, initiative, and contribution. " +
        "Do not penalize for formatting or unconventional paths. " +
        "Return ONLY valid JSON matching the provided schema."
    },
    {
      role: "user",
      content:
        `Resume filename: ${resumeName || "resume.pdf"}\n\n` +
        "Rubric (each 0–10):\n" +
        "- math_engagement: math coursework depth, problem-solving culture, TA/tutoring, seminars, competitions, reading groups\n" +
        "- research_exposition: research experience, papers/posters, expository writing, projects with mathematical substance\n" +
        "- leadership_service: organizing, outreach, mentoring, chapter/community involvement\n" +
        "- initiative_trajectory: self-started work, sustained commitment, upward trajectory\n\n" +
        "Overall resume_score (0–10) should reflect a holistic view.\n\n" +
        "RESUME TEXT:\n" +
        resumeText
    }
  ];
}