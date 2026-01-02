-- automain.lua
-- Return to MAIN when a non-main video reaches EOF.

local opts = {
    main = "",   -- path to main video (passed via --script-opts)
    loop = "inf"
}

require("mp.options").read_options(opts, "automain")

local function norm(p)
    if not p or p == "" then return "" end
    return (p:gsub("\\", "/"):lower())
end

local main_norm = norm(opts.main)

-- Track current file
local current_norm = ""

mp.register_event("file-loaded", function()
    current_norm = norm(mp.get_property("path"))
end)

-- Observe EOF state
mp.observe_property("eof-reached", "bool", function(_, v)
    if v ~= true then return end
    if main_norm == "" then return end

    -- Refresh current path 
    current_norm = norm(mp.get_property("path"))
    if current_norm == "" then return end

    -- If MAIN hit EOF, do nothing
    if current_norm == main_norm then return end

    -- Secondary hit EOF while keep-open -> go back to MAIN
    mp.commandv("loadfile", opts.main, "replace")
    mp.set_property("loop", opts.loop)
    mp.set_property_native("pause", false)
end)
