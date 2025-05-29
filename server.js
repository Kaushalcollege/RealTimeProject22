const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");
const requestIp = require("request-ip");
const useragent = require("useragent");
const cors = require("cors");

const app = express();

//Enable CORS for frontend
app.use(cors({
    origin: "http://localhost:63342",
    methods: "GET,POST",
    allowedHeaders: "Content-Type"
}));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

const userAgentMap = new Map();
const browserVersionMap = new Map();
const osVersionMap = new Map();
const loginAttempts = {}; // Store login attempts per IP

let userAgentCounter = 1;
let browserVersionCounter = 1;
let osVersionCounter = 1;

function convertIpToNumeric(ip) {
    if (ip.includes(":")) return 0; // Handle IPv6
    return ip.split(".").reduce((acc, octet) => (acc << 8) + parseInt(octet, 10), 0);
}

function getEncodedValue(map, key, counterVar) {
    if (!map.has(key)) {
        map.set(key, counterVar);
        return counterVar + 1; // Increment counter correctly
    }
    return map.get(key);
}

app.post("/collectData", async (req, res) => {
    try {
        const clientIp = requestIp.getClientIp(req) || "0.0.0.0";
        const agent = useragent.parse(req.headers["user-agent"]);
        const currentTime = new Date();

        // Track login attempts per IP
        if (!loginAttempts[clientIp]) {
            loginAttempts[clientIp] = { count: 0, lastAttempt: Date.now() };
        }
        loginAttempts[clientIp].count++;
        const timeSinceLastAttempt = Date.now() - loginAttempts[clientIp].lastAttempt;
        loginAttempts[clientIp].lastAttempt = Date.now();

        if (loginAttempts[clientIp].count > 5 && timeSinceLastAttempt < 5000) {
            console.log(`Bot detected from IP: ${clientIp}`);
            return res.status(403).json({ error: "Login Unsuccessful. Bot detected!" });
        }

        // Properly increment counters
        userAgentCounter = getEncodedValue(userAgentMap, agent.family, userAgentCounter);
        browserVersionCounter = getEncodedValue(browserVersionMap, `${agent.family} ${agent.major}`, browserVersionCounter);
        osVersionCounter = getEncodedValue(osVersionMap, `${agent.os.family} ${agent.os.major}`, osVersionCounter);

        // Prepare request data for Flask model
        const requestData = {
            "ASN": Math.floor(Math.random() * 100000),
            "Login Hour": currentTime.getHours(),
            "IP Address": convertIpToNumeric(clientIp),
            "User Agent String": userAgentCounter,
            "Browser Name and Version": browserVersionCounter,
            "OS Name and Version": osVersionCounter,
            "Country": "Unknown",
            "Device Type": agent.device.family || "Other"
        };

        console.log("Sending Data to Flask Model:", requestData);

        // Send request to Flask model
        const flaskResponse = await axios.post("http://127.0.0.1:5000/predict", requestData, {
            headers: { "Content-Type": "application/json" }
        });

        console.log("Flask Prediction:", flaskResponse.data);

        // If the model predicts account takeover, block the login
        if (flaskResponse.data["Is Account Takeover"] === 1) {
            return res.status(403).json({ error: "Login Unsuccessful. Suspicious activity detected!" });
        }

        res.json({ message: "Login successful!" });

    } catch (error) {
        console.error("Error:", error.message);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

// Start the server
app.listen(3001, () => {
    console.log("Server is running on http://localhost:3001");
});
