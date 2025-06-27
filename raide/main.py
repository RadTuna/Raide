# Internal imports
from raide import Raide, RaideMode
from frontend import VoiceChatFrontend

# External imports
import multiprocessing as mp
import time
from loguru import logger
import argparse


def main(mode: RaideMode):
    raide = Raide(mode=mode, play_audio=mode == RaideMode.STANDALONE)
    raide.run()

def run_web_frontend():
    VoiceChatFrontend().run()

if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="Raide AI Assistant")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run the application: 'cli' or 'web'")

    args = parser.parse_args()
 
    if args.mode == "cli":
        raide_mode = RaideMode.STANDALONE
    elif args.mode == "web":
        logger.info("Launching web interface...")
        mp.Process(target=run_web_frontend).start()
        raide_mode = RaideMode.WEBSOCKET
        time.sleep(2)
    elif args.mode == "websocket":
        logger.info("Launching WebSocket interface...")
        raide_mode = RaideMode.WEBSOCKET
    else:
        logger.error("Invalid mode specified. Use 'cli' or 'web'.")
        exit(1)

    main(mode=raide_mode)
