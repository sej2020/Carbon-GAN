"""
Dataset class for the emissions datasets used in this project.

Typical usage example:
```python
>>> AUS_QLD_dataset_solar = CarbonDataset("AUS_QLD", "solar")
>>> print(AUS_QLD_dataset_solar[0])
```
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class CarbonDataset(Dataset):
    """
    Attributes:
        region: The region from which the data is reported
        elec_source: The type of fuel/method for electricity generation
        full_metadata: The metadata accompanying the emissions data: comprising time and weather data
        metadata: The metadata accompanying the emissions data: comprising time and weather data, perhaps truncated
        full_data: The emissions data by region and electricity source
        data: The emissions data by region and electricity source, perhaps truncated
        metadata_scaler: The standard scaler object for the metadata
        data_scaler: The standard scaler object for the data
    """
    def __init__(self, region: str, elec_source: str):
        """
        Initializes the CarbonDataset class. Performs normalization on the metadata and data.

        Args:
            region: The region from which the data is reported
            elec_source: The type of fuel/method for electricity generation
        """
        self.region = region
        self.elec_source = elec_source
        full_data = pd.read_csv(f"data/{region}/{region}_2019_clean.csv")
        elec_source_data = full_data[elec_source].values
        metadata = self._preprocess_metadata(region)
        self.metadata_scaler = StandardScaler()
        self.data_scaler = StandardScaler()
        self.full_metadata = torch.tensor(self.metadata_scaler.fit_transform(metadata), dtype=torch.float32)
        self.full_data = torch.tensor(self.data_scaler.fit_transform(elec_source_data.reshape(-1, 1)), dtype=torch.float32)
        self.metadata = self.full_metadata
        self.data = self.full_data


    def _preprocess_metadata(self, region):
        """
        Preprocesses datetime information and assembles the weather and datetime data.

        Args:
            region: The region from which the data is reported
        
        Returns:
            The weather and time data in the form of a numpy array
        """
        weather_data = pd.read_csv(f"data/{region}/{region}_aggregated_weather_data.csv")
        weather_data["day"] = weather_data["datetime"].apply(lambda x: int(x.split(" ")[0].split("-")[2]))
        weather_data["month"] = weather_data["datetime"].apply(lambda x: int(x.split(" ")[0].split("-")[1]))
        weather_data["year"] = weather_data["datetime"].apply(lambda x: int(x.split(" ")[0].split("-")[0])-2019)
        weather_data["hour"] = weather_data["datetime"].apply(lambda x: int(x.split(" ")[1].split(":")[0]))
        weather_data.drop(columns=["datetime"], inplace=True)
        wd_cols = weather_data.columns.to_list()
        wd_cols = wd_cols[-4:] + wd_cols[:-4]
        weather_data = weather_data[wd_cols]
        return weather_data.values
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.metadata[idx], self.data[idx]
    

    def _roll(self, idx):
        """
        Args:
            idx: The number of elements to remove from the front of the dataset
        
        Returns:
            A new dataset with the first idx elements removed
        """
        self.metadata = self.metadata[idx:]
        self.data = self.data[idx:]


    def _unroll(self):
        """
        Resets the meta data and data to their original (unrolled) state
        """
        self.metadata = self.full_metadata
        self.data = self.full_data
    
    
if __name__ == '__main__':
    AUS_QLD_dataset_solar = CarbonDataset("AUS_QLD", "solar")
    print(AUS_QLD_dataset_solar[0])