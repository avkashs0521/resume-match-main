from app.api.main import app
import uvicorn
import os

def main():
    """Main entry point for starting the server"""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
