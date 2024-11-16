#!/usr/bin/env julia

using JuliaFormatter

# Get all files passed as arguments
files = ARGS

exit_code = 0

for file in files
    try
        if !isfile(file)
            println("Warning: File $file not found, skipping...")
            continue
        end
        
        println("Formatting $file...")
        formatted = format_file(file)
        
        if !formatted
            println("⚠️  Failed to format $file")
            exit_code = 1
        end
    catch e
        println("❌ Error formatting $file: $e")
        exit_code = 1
    end
end

exit(exit_code)