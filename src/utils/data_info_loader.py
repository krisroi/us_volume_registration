import pandas as pd


class GetDatasetInformation():
    """Reads dataset information from a .txt file and returns the information as lists
        Args:
            filepath (string): filepath to the .txt file
            filename (string): filename of the .txt file
    """

    def __init__(self, filepath, filename):
        super(GetDatasetInformation, self).__init__()

        self.filepath = filepath
        self.filename = filename

        self.fix_files, self.mov_files, self.fix_vols, self.mov_vols = self.load_dataset()

    def load_dataset(self):
        """ Reads the dataset information, pulls out the usable datasets
            and returns them together with corresponding volumes.
        """

        data = pd.read_csv('{}{}'.format(self.filepath, self.filename))
        data = data.loc[lambda df: data.usable == 'y', :]  # Extract only usable datasets (y: yes)
        group = data.group
        patient = data.patient
        ref_filename = data.ref_filename
        mov_filename = data.mov_filename
        ref_vol_frame_no = data.ref_vol_frame_no
        mov_vol_frame_no = data.mov_vol_frame_no

        # Initializing empty list-holders
        fix_files = []
        mov_files = []
        fix_vols = []
        mov_vols = []

        for _, pat_idx in enumerate((patient.index)):
            fix_files.append('{}/{}/{}'.format(group[pat_idx], patient[pat_idx], ref_filename[pat_idx]))
            mov_files.append('{}/{}/{}'.format(group[pat_idx], patient[pat_idx], mov_filename[pat_idx]))
            fix_vols.append('{:02}'.format(ref_vol_frame_no[pat_idx]))
            mov_vols.append('{:02}'.format(mov_vol_frame_no[pat_idx]))

        return fix_files, mov_files, fix_vols, mov_vols
