import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from retrain import retrain_model

class NewDataHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.xlsx', '.csv')):
            print(f"New data file detected: {event.src_path}")
            # Wait a moment to ensure file is fully written
            time.sleep(5)
            retrain_model()

def start_monitoring(path_to_watch):
    event_handler = NewDataHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=False)
    observer.start()
    print(f"Monitoring {path_to_watch} for new data files...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    start_monitoring('data/new_data')