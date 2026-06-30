from pathlib import Path
from astropy.table import Table

import pandas as pd
import numpy as np

from Spec7DT.core import FitResultSet


class CatalogueManager():
    def __init__(self):
        pass
    
    def find_results_file(self, pattern: Path, recent: int=-1) -> Path:
        """Return the most recent FITS file matching the given glob pattern."""
        files = sorted(pattern.parent.glob(pattern.name))
        if not files:
            raise FileNotFoundError(f"No files match {pattern}")
        return files[recent]

    def load_results(self, fits_path: Path, supply_path: Path = None) -> pd.DataFrame:
        """Read the CIGALE FITS, decode IDs, and split into 'id' + integer 'idx_num'."""
        tbl = Table.read(fits_path, format='fits')
        units = {col: tbl[col].unit for col in tbl.colnames}
        df  = tbl.to_pandas()
        df['id'] = df['id'].str.decode('utf-8')

        if supply_path is not None:
            supply_df = Table.read(supply_path, format='fits').to_pandas()
            supply_df['id'] = supply_df['id'].str.decode('utf-8')
            
            df = df.set_index('id')
            supply_df = supply_df.set_index('id')
            
            df.update(supply_df)
            df = df.reset_index()
        
        totals = df[df['id'] == 'Total']  # get total value
        df.drop(df[df['id'] == 'Total'].index, axis=0, inplace=True)
        df[['id', 'idx_num']] = df['id'].str.rsplit('_', n=1, expand=True)
        df['idx_num'] = df['idx_num'].astype(int)
        return df, totals, units

    def build_rslts_dict(
        self,
        df: pd.DataFrame,
        targets: list[str],
        im_size: int
    ) -> dict[str, pd.DataFrame]:
        """
        For each target ID, create a DataFrame indexed 0..im_size**2-1,
        with missing rows filled with zeros.
        """
        full_idx = np.arange(im_size**2)
        return {
            targ: (
                df.loc[df['id'] == targ]
                .set_index('idx_num')
                .reindex(full_idx, fill_value=0)
            )
            for targ in targets
        }
    

    def load(self, results_file: str=None, im_size: int=200, supply_file: str=None, pattern: str | Path = None):
        if results_file is None:
            if pattern is None:
                raise ValueError("results_file or pattern must be provided.")
            results_file = self.find_results_file(Path(pattern), recent=-2)
            
        results_file = Path(results_file) if isinstance(results_file, str) else results_file
        date_tag = results_file.stem.split('_')[-1]

        supply_path = Path(supply_file) if supply_file else None

        rslt_df, total, units = self.load_results(results_file, supply_path=supply_path)
        targets = rslt_df['id'].unique().tolist()

        rslts = self.build_rslts_dict(rslt_df, targets, im_size)

        return results_file, date_tag, rslts, total, units

    def load_fit_result_set(
        self,
        results_file: str | Path,
        im_size: int = 200,
        supply_file: str | Path = None,
        tool: str = "cigale",
    ) -> FitResultSet:
        """Load fitting results into the shared Spec7DT FitResultSet contract."""
        source_file, date_tag, results, total, units = self.load(results_file, im_size, supply_file)
        return FitResultSet(
            results=results,
            totals=total,
            units=units,
            source_file=source_file,
            tool=tool,
            image_size=im_size,
            date_tag=date_tag,
        )


class CigaleResultAdapter:
    """Adapter for CIGALE output tables."""

    def __init__(self, manager: CatalogueManager = None):
        self.manager = manager or CatalogueManager()

    def load(self, results_file: str | Path, im_size: int = 200, supply_file: str | Path = None) -> FitResultSet:
        return self.manager.load_fit_result_set(results_file, im_size=im_size, supply_file=supply_file, tool="cigale")
