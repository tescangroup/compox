"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import os
import requests
import subprocess
import argparse
import platform

from loguru import logger

from compox.config.server_settings import get_server_settings


def parse_args():
    """
    Argument parser for the server.

    Returns
    -------
    object
        The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Path to server configuration yaml file. If None, use defaults specified in server_config.py.",
        default=None,
        required=False,
    )
    return parser.parse_args()


def get_minio(settings):
    if os.path.isfile(settings.storage.backend_settings.executable_path):
        logger.info("MinIO binary available.")
        return

    logger.info(
        f"Downloading MinIO binary into {settings.storage.backend_settings.executable_path}..."
    )

    os.makedirs(
        os.path.dirname(settings.storage.backend_settings.executable_path),
        exist_ok=True,
    )

    # if the operating system is posix
    if os.name == "posix":
        # download minio binary
        minio_url = "https://github.com/minio/minio/releases/download/RELEASE.2025-09-07T16-13-09Z/minio.linux-amd64.RELEASE.2025-09-07T16-13-09Z"
        if platform.system().lower() == "darwin":
            minio_url = f"https://github.com/minio/minio/releases/download/RELEASE.2025-09-07T16-13-09Z/minio.darwin-{platform.machine()}.RELEASE.2025-09-07T16-13-09Z"
        try:
            response = requests.get(minio_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error("Could not download the Minio binary due to: ", e)
            exit(1)
        # save the binary
        try:
            with open(
                settings.storage.backend_settings.executable_path, "wb"
            ) as file:
                file.write(response.content)
        except Exception as e:
            logger.error(
                "Could not write the downloaded Minio binary to disk due to: ",
                e,
            )
            exit(1)
        # set runnable permission
        try:
            subprocess.Popen(
                [
                    "chmod",
                    "+x",
                    settings.storage.backend_settings.executable_path,
                ]
            )
        except Exception as e:
            logger.error(
                "Could not set the Minio binary as runnable due to: ", e
            )
            exit(1)
    elif os.name == "nt":
        # download minio binary
        try:
            response = requests.get(
                "https://github.com/minio/minio/releases/download/RELEASE.2025-09-07T16-13-09Z/minio.windows-amd64.RELEASE.2025-09-07T16-13-09Z.exe"
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error("Could not download the Minio binary due to: ", e)
            exit(1)

        try:
            # save the binary
            with open(
                settings.storage.backend_settings.executable_path, "wb"
            ) as file:
                file.write(response.content)
        except Exception as e:
            logger.error(
                "Could not write the downloaded Minio binary to disk due to: ",
                e,
            )
            exit(1)
    else:
        logger.error("Operating system not supported")
        exit(1)


def main():
    """
    Main function to run the downloader.
    """
    args = parse_args()
    settings = get_server_settings(args.config)

    get_minio(settings)
    os.makedirs(settings.storage.backend_settings.storage_path, exist_ok=True)


if __name__ == "__main__":
    main()
