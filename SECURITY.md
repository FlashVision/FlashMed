# Security Policy

## Medical Data Handling

FlashMed processes sensitive medical imaging data. We take security and privacy seriously.

### HIPAA Compliance

- FlashMed includes DICOM de-identification tools following HIPAA Safe Harbor guidelines
- Never commit real patient data to version control
- Use the `DicomAnonymizer` before sharing any DICOM files
- Federated learning enables model training without sharing raw patient data

### Data Protection

- All DICOM files are excluded from git tracking by default (`.gitignore`)
- The anonymization module removes 18+ HIPAA identifiers
- Date shifting preserves temporal relationships while protecting identity
- Pseudonymization uses salted SHA-256 hashing

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email: security@flashvision.dev
3. Include a description of the vulnerability and steps to reproduce
4. Allow reasonable time for a fix before public disclosure

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | Yes                |

## Security Best Practices

When using FlashMed:

1. Keep dependencies updated (`pip install --upgrade flashmed`)
2. Run anonymization before sharing any medical data
3. Use differential privacy for training on sensitive data
4. Enable federated learning for multi-site collaborations
5. Review model outputs before clinical use — AI is a decision support tool
