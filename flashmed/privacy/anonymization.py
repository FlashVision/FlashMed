"""DICOM de-identification and anonymization utilities."""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flashmed.registry import PRIVACY_METHODS


HIPAA_IDENTIFIERS = {
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
    "ReferringPhysicianName", "InstitutionName", "InstitutionAddress",
    "StationName", "PhysiciansOfRecord", "PerformingPhysicianName",
    "OperatorsName", "OtherPatientIDs", "OtherPatientNames",
    "PatientBirthName", "PatientMotherBirthName", "MedicalRecordLocator",
    "PatientTelephoneNumbers", "ResponsiblePerson", "RequestingPhysician",
}

SAFE_PRIVATE_TAGS_TO_REMOVE = {
    "StudyDescription", "SeriesDescription", "ImageComments",
    "AdditionalPatientHistory", "PatientComments",
}


@PRIVACY_METHODS.register("anonymization")
class DicomAnonymizer:
    """DICOM de-identification following HIPAA Safe Harbor guidelines.

    Removes or pseudonymizes Protected Health Information (PHI) from DICOM files
    while preserving medically relevant metadata for analysis.

    Args:
        method: "remove" to blank PHI fields, "pseudonymize" to replace with hashed values
        preserve_dates: Keep study dates (shifted by random offset)
        date_offset_days: Fixed offset for date shifting (random if None)
        keep_descriptors: Preserve non-PHI descriptive fields (Modality, BodyPart, etc.)
        salt: Salt for pseudonymization hashing
    """

    def __init__(
        self,
        method: str = "pseudonymize",
        preserve_dates: bool = True,
        date_offset_days: Optional[int] = None,
        keep_descriptors: bool = True,
        salt: str = "flashmed_anon",
    ):
        self.method = method
        self.preserve_dates = preserve_dates
        self.date_offset_days = date_offset_days or hash(salt) % 365
        self.keep_descriptors = keep_descriptors
        self.salt = salt

    def anonymize_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Anonymize a single DICOM file.

        Args:
            input_path: Path to original DICOM file
            output_path: Output path (overwrites input if None)

        Returns:
            Path to anonymized file
        """
        import pydicom

        ds = pydicom.dcmread(input_path)
        ds = self._anonymize_dataset(ds)

        out = output_path or input_path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(out)
        return out

    def anonymize_directory(self, input_dir: str, output_dir: str) -> List[str]:
        """Anonymize all DICOM files in a directory.

        Args:
            input_dir: Source directory
            output_dir: Destination directory

        Returns:
            List of output file paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results = []

        for dcm_file in input_path.rglob("*.dcm"):
            relative = dcm_file.relative_to(input_path)
            out_file = output_path / relative
            out_file.parent.mkdir(parents=True, exist_ok=True)
            self.anonymize_file(str(dcm_file), str(out_file))
            results.append(str(out_file))

        print(f"[Anonymization] Processed {len(results)} DICOM files")
        return results

    def _anonymize_dataset(self, ds) -> object:
        """Apply anonymization rules to a DICOM dataset."""
        for tag_name in HIPAA_IDENTIFIERS:
            if hasattr(ds, tag_name):
                if self.method == "pseudonymize":
                    original = str(getattr(ds, tag_name))
                    pseudonym = self._pseudonymize(original)
                    setattr(ds, tag_name, pseudonym)
                else:
                    setattr(ds, tag_name, "")

        if not self.keep_descriptors:
            for tag_name in SAFE_PRIVATE_TAGS_TO_REMOVE:
                if hasattr(ds, tag_name):
                    setattr(ds, tag_name, "")

        if self.preserve_dates:
            self._shift_dates(ds)
        else:
            date_fields = ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"]
            for field in date_fields:
                if hasattr(ds, field):
                    setattr(ds, field, "")

        ds.remove_private_tags()

        if hasattr(ds, "DeidentificationMethod"):
            ds.DeidentificationMethod = "FlashMed Anonymizer (HIPAA Safe Harbor)"

        return ds

    def _pseudonymize(self, value: str) -> str:
        """Generate a deterministic pseudonym from original value."""
        h = hashlib.sha256(f"{self.salt}:{value}".encode()).hexdigest()[:12]
        return f"ANON_{h.upper()}"

    def _shift_dates(self, ds):
        """Shift all dates by a fixed offset."""
        from datetime import timedelta

        date_fields = ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "PatientBirthDate"]
        for field in date_fields:
            if hasattr(ds, field):
                date_str = str(getattr(ds, field))
                if date_str and len(date_str) == 8:
                    try:
                        dt = datetime.strptime(date_str, "%Y%m%d")
                        shifted = dt + timedelta(days=self.date_offset_days)
                        setattr(ds, field, shifted.strftime("%Y%m%d"))
                    except ValueError:
                        setattr(ds, field, "")

    def verify_anonymization(self, path: str) -> Dict[str, bool]:
        """Verify that a DICOM file has been properly anonymized.

        Returns:
            Dict of checks and their pass/fail status
        """
        import pydicom
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        results = {}

        for tag_name in HIPAA_IDENTIFIERS:
            if hasattr(ds, tag_name):
                value = str(getattr(ds, tag_name))
                is_clean = value == "" or value.startswith("ANON_")
                results[tag_name] = is_clean
            else:
                results[tag_name] = True

        results["private_tags_removed"] = len(ds.private_creators()) == 0 if hasattr(ds, "private_creators") else True
        return results
