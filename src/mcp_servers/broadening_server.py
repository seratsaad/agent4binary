#!/usr/bin/env python3
"""
MCP Server for Line Broadening Calculations

Provides tools for thermal, rotational and instrumental (LSF) broadening.

Adapted for agent4binary (line-broadening tools for SB2 disentangling). Original
from Agents_Summer_School/notebooks/mcp_servers/broadening_server.py; the
instrumental_broadening (APOGEE LSF) tool is added for this project.
"""

import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Broadening Tools", log_level="WARNING")
C_KMS = 299792.458


@mcp.tool()
def thermal_broadening(
    wavelength_angstroms: float,
    temperature_kelvin: float,
    atomic_mass_amu: float
) -> dict:
    """
    Calculate thermal (Doppler) broadening of a spectral line.
    
    Atoms in a hot gas move randomly due to thermal motion, broadening
    spectral lines. Hotter gas = faster atoms = broader lines.
    Heavier atoms move slower at the same temperature = narrower lines.
    
    Args:
        wavelength_angstroms: Line wavelength in Angstroms
        temperature_kelvin: Gas temperature in Kelvin
        atomic_mass_amu: Atomic mass in amu (H=1, Fe=55.845, etc.)
        
    Returns:
        Thermal broadening FWHM in Angstroms and km/s
    """
    if wavelength_angstroms <= 0 or temperature_kelvin <= 0 or atomic_mass_amu <= 0:
        return {"error": "All parameters must be positive"}
    
    k_B = 1.380649e-23  # Boltzmann constant
    amu_kg = 1.66054e-27
    mass_kg = atomic_mass_amu * amu_kg
    
    sigma_v = np.sqrt(2 * k_B * temperature_kelvin / mass_kg)
    fwhm_kms = 2.355 * sigma_v / 1000
    fwhm_angstroms = wavelength_angstroms * fwhm_kms / C_KMS
    
    return {
        "fwhm_angstroms": round(float(fwhm_angstroms), 4),
        "fwhm_kms": round(float(fwhm_kms), 3),
        "sigma_kms": round(float(fwhm_kms / 2.355), 3)
    }


@mcp.tool()
def rotational_broadening(
    wavelength_angstroms: float,
    vsini_kms: float
) -> dict:
    """
    Estimate line broadening from stellar rotation.
    
    One stellar limb moves toward us (blueshift), the other away (redshift),
    broadening spectral lines. We measure 'v sin i'—rotation speed times
    sin(inclination)—since we can't separate these from spectra alone.
    
    Args:
        wavelength_angstroms: Line wavelength in Angstroms
        vsini_kms: Projected rotational velocity in km/s
                  (Sun ≈ 2, typical F star ≈ 20-50, hot stars > 200)
        
    Returns:
        Rotational broadening estimate and rotation classification
    """
    if wavelength_angstroms <= 0:
        return {"error": "Wavelength must be positive"}
    if vsini_kms < 0:
        return {"error": "v sin i must be non-negative"}
    
    delta_lambda = wavelength_angstroms * vsini_kms / C_KMS
    fwhm = 1.8 * delta_lambda
    
    if vsini_kms < 5:
        rot_class = "Slow rotator (Sun-like)"
    elif vsini_kms < 50:
        rot_class = "Moderate rotator"
    elif vsini_kms < 150:
        rot_class = "Fast rotator"
    else:
        rot_class = "Very fast rotator"
    
    return {
        "fwhm_angstroms": round(float(fwhm), 4),
        "delta_lambda_angstroms": round(float(delta_lambda), 4),
        "rotation_class": rot_class
    }


@mcp.tool()
def instrumental_broadening(
    wavelength_angstroms: float,
    resolving_power: float = 22500.0,
) -> dict:
    """
    Estimate the instrumental (LSF) line broadening of a spectrograph.

    A spectrograph smears intrinsically sharp lines by its line-spread function,
    set by the resolving power R = lambda / dlambda. APOGEE's nominal R ~ 22500.
    This sets the floor on any measurable broadening and must be deconvolved
    before interpreting rotational/thermal widths.

    Args:
        wavelength_angstroms: Line wavelength in Angstroms.
        resolving_power: Spectral resolving power R (APOGEE ~= 22500).

    Returns:
        Instrumental FWHM in Angstroms and km/s, plus the velocity-space sigma.
    """
    if wavelength_angstroms <= 0:
        return {"error": "Wavelength must be positive"}
    if resolving_power <= 0:
        return {"error": "Resolving power must be positive"}

    fwhm_angstroms = wavelength_angstroms / resolving_power
    fwhm_kms = C_KMS / resolving_power

    return {
        "fwhm_angstroms": round(float(fwhm_angstroms), 4),
        "fwhm_kms": round(float(fwhm_kms), 3),
        "sigma_kms": round(float(fwhm_kms / 2.355), 3),
        "resolving_power": float(resolving_power),
    }


if __name__ == "__main__":
    mcp.run()
