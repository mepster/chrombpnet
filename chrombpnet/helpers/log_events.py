from subprocess import run
import pandas as pd
import os, time
from csv import writer

class EventLog():
  def __init__(self, *args, **kwargs):
    print("el", args, kwargs)
    self.filename = kwargs['filename']
    with open(self.filename, 'w') as f_object:
      writer_object = writer(f_object)
      writer_object.writerow(['timestamp', 'event_name'])
      
  def write_event(self, event_name):
    t = pd.Timestamp.now()
    row = [t, event_name]
    print(row)
    with open(self.filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(row)

if __name__ == "__main__":
  el = EventLog(filename="el.log")
  for i in range(5):
    el.write_event(event_name=f"event {i}")
    time.sleep(1)
