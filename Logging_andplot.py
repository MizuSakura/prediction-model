import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self):
        self.df = pd.DataFrame()
        self.current_path = Path(os.getcwd())

    def add_data_log(self, columns_name, data_list):  
        if len(columns_name) != len(set(columns_name)):
            raise ValueError("columns_name contains duplicate columns, please check your input")

        def safe_len(x):
            if hasattr(x, '__len__'):
                return len(x)
            return 1

        max_len = max(safe_len(col_data) for col_data in data_list)

        padded_data = []
        for col_data in data_list:
            if not hasattr(col_data, '__len__'):
                col_data = [col_data] * max_len
            elif len(col_data) < max_len:
                col_data = col_data + [None] * (max_len - len(col_data))
            else:
                col_data = col_data[:max_len]
            padded_data.append(col_data)

        new_data = {col: data for col, data in zip(columns_name, padded_data)}

        if self.df.empty:
            self.df = pd.DataFrame(new_data)
        else:
            extra_cols = [col for col in self.df.columns if col not in columns_name]
            for col in extra_cols:
                new_data[col] = [None] * max_len
            
            all_cols = list(self.df.columns) + [col for col in columns_name if col not in self.df.columns]
            new_rows = pd.DataFrame(new_data)
            new_rows = new_rows.reindex(columns=all_cols, fill_value=None)
            if not self.df.empty:
                new_rows = new_rows.astype(self.df.dtypes.to_dict())
            self.df = pd.concat([self.df, new_rows], ignore_index=True)

    def clear_data(self):
        self.df = pd.DataFrame()
        print("Data cleared")

    def show_data(self):
        print(self.df)

    def result_column(self, column_name=None):
        if self.df.empty:
            print("No data to display")
            return None
        if column_name is None:
            return None
        elif column_name in self.df.columns:
            data = self.df[column_name]
            data = data.to_numpy()
            return data
        else:
            print(f'column: {column_name} does not exist in the data')
            return

    def save_to_csv(self, file_name, folder_name=None, path_name=None):
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        if path_name is not None:
            path_to_save = Path(path_name)
        else:
            path_to_save = self.current_path
        if folder_name is None:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            folder_to_save = path_to_save / today
        else:
            folder_to_save = path_to_save / folder_name
        print(f'path file : {folder_to_save}')
        folder_to_save.mkdir(parents=True, exist_ok=True)
        path_file = folder_to_save / file_name
        self.df.to_csv(path_file, index=False)
        print(f"Data saved to {path_file}")

    def load_csv(self, path_file):
        path = Path(path_file)
        if path.suffix.lower() == ".csv":
            try:
                self.df = pd.read_csv(path)
            except Exception as e:
                print("File not found. Please check the file path and file name.")
        else:
            raise ValueError("CSV file not found. Please check the file extension ")
    
    def append_from_csv_folder(self, folder_path):

        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            print(f"Invalid folder: {folder}")
            return

        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            print("No CSV files found in the folder.")
            return

        combined_df = pd.DataFrame()
        for file in csv_files:
            try:
                temp_df = pd.read_csv(file)
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
            except Exception as e:
                print(f"Error loading {file.name}: {e}")

        if self.df.empty:
            self.df = combined_df
        else:
            self.df = pd.concat([self.df, combined_df], ignore_index=True)

        print(f"Appended {len(csv_files)} files from {folder_path}")

    def check_column_condition(self, column_name =None, traget_value = True,min_count=0,start=None,end=None):

        if self.df.empty:
            print("No data available to check.")

        if column_name not in self.df.columns:
            print(f"Column '{column_name}' not found.")

        data_slice = self.df[column_name]
        if start is not None or end is not None:
            data_slice = data_slice[start:end]
        
        count = (data_slice == traget_value).sum()

        return count >= min_count

