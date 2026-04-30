import sys
from PySide6.QtWidgets import QApplication

from core.config_manager import ConfigManager
from core.db import DatabaseManager
from core.paths import resource_path
from ui.login_window import LoginWindow
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    try:
        with resource_path("assets/style.qss").open("r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        pass

    config = ConfigManager()
    db = DatabaseManager(config.get("database_path", "database/hospital_client.db"))
    db.initialize()

    windows = {}

    def open_main(role: str, hospital_name: str, hospital_id: str, display_name: str):
        config.set("user_role", role)
        config.set("hospital_name", hospital_name)
        config.set("hospital_id", hospital_id)
        config.set("display_name", display_name)

        windows["main"] = MainWindow(config=config, db=db)
        windows["main"].showMaximized()
        windows["login"].close()

    windows["login"] = LoginWindow(on_login=open_main, db=db)
    windows["login"].resize(420, 400)
    windows["login"].show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
