# Internal imports
from raide import Raide, RaideMode
from frontend import VoiceChatFrontend

# External imports
import multiprocessing as mp
import time
from loguru import logger
import argparse


def main(mode: RaideMode, port: int):
    raide = Raide(mode=mode, port=port, play_audio=mode == RaideMode.STANDALONE)
    raide.run()

def run_web_frontend(frontend_port: int, websocket_port: int):
    VoiceChatFrontend().run(frontend_port=frontend_port, websocket_port=websocket_port)

if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="Raide AI Assistant")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run the application: 'cli' or 'web'")
    parser.add_argument("--frontend-port", type=int, default=7860, help="Port for the web frontend")
    parser.add_argument("--websocket-port", type=int, default=8765, help="Port for the WebSocket server")

    args = parser.parse_args()
 
    if args.mode == "cli":
        raide_mode = RaideMode.STANDALONE
    elif args.mode == "web":
        logger.info("Launching web interface...")
        mp.Process(target=run_web_frontend, args=(args.frontend_port, args.websocket_port)).start()
        raide_mode = RaideMode.WEBSOCKET
        time.sleep(2)
    elif args.mode == "websocket":
        logger.info("Launching WebSocket interface...")
        raide_mode = RaideMode.WEBSOCKET
    else:
        logger.error("Invalid mode specified. Use 'cli' or 'web'.")
        exit(1)

    main(mode=raide_mode, port=args.websocket_port)
