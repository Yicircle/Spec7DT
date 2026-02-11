import subprocess
import os
import numpy as np
from typing import Union
from importlib import resources
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS

from ..utils.utility import useful_functions


class Register:
    @classmethod
    def save_all_progress(cls, image_data, header, error_data, galaxy_name, observatory, band, image_set):
        file_list = list(useful_functions.extract_values_recursive(image_set._files, galaxy_name))
        
        repre_path = Path(file_list[0]).parent
        output_path = repre_path / "SWarp"
        
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        current_file_path = Path(image_set._files[galaxy_name][observatory][band])
        current_filename = current_file_path.name
        
        fits.writeto(output_path / str("unpro_" + str(current_filename)), image_data, header, overwrite=True)
        
        if error_data is not None:
            fits.writeto(output_path / str("unpro_" + str(current_filename.replace(".fits", "_err.fits"))), error_data, header, overwrite=True)
        
    
    @classmethod
    def swarp_register(cls, image_data, header, error_data, galaxy_name, observatory, band, image_set):
        file_list = list(useful_functions.extract_values_recursive(image_set._files, galaxy_name))
        
        repre_path = Path(file_list[0]).parent
        output_path = repre_path / "SWarp"
        
        if image_set._files[galaxy_name][observatory][band] == file_list[0]:
            saved_files = sorted([f for f in output_path.glob("*.fits") 
                                if not f.name.endswith("_err.fits") 
                                and "weight" not in f.name 
                                and "coadd" not in f.name
                                and ".resamp." not in f.name
                                and "unpro_" in f.name])
            
            list_file_path = output_path / f"{galaxy_name}.list"
            with open(list_file_path, "w") as f:
                for saved_file in saved_files:
                    f.write(str(saved_file) + "\n")
                    err_file = saved_file.parent / saved_file.name.replace(".fits", "_err.fits")
                    if err_file.exists():
                        f.write(str(err_file) + "\n")
                        
            file_list_name = str(list_file_path)
            
            center_coord = useful_functions.get_sky_loc(galaxy_name)
            
            cls.swarp(cls, input_list=file_list_name, output=output_path,
                    center=f"{center_coord.ra.value},{center_coord.dec.value}",
                    dump_dir=output_path,
                    resample_dir=output_path,
                    log_file=output_path / "log.log")

            for file in os.listdir(output_path):
                if file.endswith(".fits") and '.resamp.' in file:
                    if "weight" in file:
                        os.remove(output_path / file)
                    else:
                        os.rename(output_path / file, output_path / file.replace('.resamp.','.').replace('unpro_', ''))
        
        file_name = Path(image_set._files[galaxy_name][observatory][band]).name
        registered_file = output_path / file_name
        registered_data = fits.getdata(registered_file).astype(np.float32)
        
        registered_header = fits.getheader(registered_file)
        
        header_clean = header.copy()
        wcs_keys = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
                    'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2',
                    'CUNIT1', 'CUNIT2', 'RADESYS', 'LONPOLE', 'LATPOLE']
        
        for key in wcs_keys:
            if key in header_clean:
                del header_clean[key]
        
        for key in list(header_clean.keys()):
            if key.startswith('PV'):
                del header_clean[key]
        
        updated_header = useful_functions.update_header(header_clean, registered_header)
        
        image_set.update_data(registered_data, galaxy_name, observatory, band)
        image_set.update_header(updated_header, galaxy_name, observatory, band)
        
        registered_error_file = str(registered_file).replace(".fits", "_err.fits")
        if os.path.isfile(registered_error_file):
            registered_error = fits.getdata(registered_error_file).astype(np.float32)
            image_set.update_error(registered_error, galaxy_name, observatory, band)
            
            
    @classmethod
    def trim(cls, image_data, header, error_data, galaxy_name, observatory, band, image_set, trim_size):
        center_coord = useful_functions.get_sky_loc(galaxy_name)
        
        trim_image, trim_header, trim_error = cls.trim_sky(cls, image=image_data, header=header, error=error_data, skycoord=center_coord, size=trim_size)
        
        
        image_set.update_data(trim_image, galaxy_name, observatory, band)
        image_set.update_header(trim_header, galaxy_name, observatory, band)
        image_set.update_error(trim_error, galaxy_name, observatory, band)
    
    def swarp(
        self,
        input_list,
        output=None,
        center=None,
        dump_dir=None,
        resample_dir=None,
        log_file=None,
    ):
        """input is a list of filenames"""
                
        def add_suffix(filename: str | list[str], suffix: str | list[str]) -> str | list[str]:
            """
            Add a suffix to the filename before the extension.
            Both filename and suffix can be strings or lists of strings.

            Args:
                filename (str | list[str]): The original filename(s).
                suffix (str | list[str]): The suffix to add.

            Returns:
                str | list[str]: The modified filename(s) with the suffix added.
            """
            if isinstance(suffix, list):
                if len(suffix) == 1:
                    suffix = suffix[0]
                else:
                    assert len(filename) == len(suffix), "Filename and suffix must have the same length"
                    return [add_suffix(f, s) for f, s in zip(filename, suffix)]

            if isinstance(suffix, str):
                return _add_suffix(filename, suffix)

            raise ValueError(f"Invalid suffix type: {type(suffix)}")


        def _add_suffix(filename: str | list[str], suffix: str) -> str | list[str]:
            if isinstance(filename, list):
                return [add_suffix(f, suffix) for f in filename]
            stem, ext = os.path.splitext(filename)
            suffix = suffix if suffix.startswith("_") else f"_{suffix}"
            return f"{stem}{suffix}{ext}"

        if isinstance(input_list, list):
            input_list = ",".join(input_list)
        elif isinstance(input_list, str):  # assume file input
            input_list = f"@{input_list}"
        else:
            raise ValueError("Input must be a list or a string")

        if not center:
            raise ValueError("Deprojection center undefined")

        dump_dir = dump_dir or os.path.join(os.path.dirname(input_list), "tmp_swarp")
        log_file = log_file or os.path.join(dump_dir, "swarp.log")
        resample_dir = resample_dir or os.path.join(dump_dir, "resamp")
        os.makedirs(resample_dir, exist_ok=True)
        comim = output or os.path.join(dump_dir, "coadd.fits")
        # weightim = swap_ext(comim, "weight.fits")
        weightim = add_suffix(comim, "weight")

        # 	SWarp
        config_file_resource = resources.files('Spec7DT.reference.configs').joinpath('default.swarp')
        with resources.as_file(config_file_resource) as config_path:
            config_file = str(config_path)

        swarpcom = [
            "swarp", input_list,
            "-c", f"{config_file}",
            "-IMAGEOUT_NAME", f"{comim}",
            "-RESAMPLE_DIR", f"{resample_dir}"
        ]  

        swarpcom = " ".join(swarpcom)

        process = subprocess.Popen(
            swarpcom,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        with open(log_file, "w") as f:
            f.write(swarpcom + "\n" * 3)
            f.flush()
            for line in process.stdout:
                f.write(line)
                f.flush()

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"SWarp failed with return code {process.returncode}. See log: {log_file}")

        return swarpcom
    

    def trim_sky(self,
                 image,
                 header,
                 error,
                 skycoord: Union[tuple, SkyCoord],
                 size: tuple
                 ):

        wcs = WCS(header)

        skycoord_obj = SkyCoord(ra=skycoord[0]*u.deg, dec=skycoord[1]*u.deg) if isinstance(skycoord, tuple) else skycoord
        pixel_position = skycoord_obj.to_pixel(wcs=wcs)

        cut = Cutout2D(image, pixel_position, size, wcs=wcs, mode='partial', fill_value=0.0)
        cut_error = Cutout2D(error, pixel_position, size, wcs=wcs, mode='partial', fill_value=0.0) if error is not None else None

        new_header = header.copy()
        wcs_keys = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
                    'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2',
                    'CUNIT1', 'CUNIT2', 'RADESYS', 'LONPOLE', 'LATPOLE']
        
        for key in wcs_keys:
            if key in new_header:
                del new_header[key]
        
        for key in list(new_header.keys()):
            if key.startswith('PV'):
                del new_header[key]
        
        new_header.update(cut.wcs.to_header())

        return np.asarray(cut.data, order="C"), new_header, np.asarray(cut_error.data, order="C")