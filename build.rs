// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/aeroflyway.ico");
    println!("cargo:rerun-if-changed=assets/aeroflyway_icon_64.png");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let ico_path = manifest_dir.join("assets").join("aeroflyway.ico");
    if !ico_path.exists() {
        println!(
            "cargo:warning=Missing icon at {}; skipping PE resource embedding",
            ico_path.display()
        );
        return;
    }

    let ico_rc = ico_path.to_string_lossy().replace('\\', "/");
    let rc_content = format!(
        r#"1 ICON "{ico_rc}"

1 VERSIONINFO
FILEVERSION 0,6,0,0
PRODUCTVERSION 0,6,0,0
FILEFLAGSMASK 0x3fL
FILEFLAGS 0x0L
FILEOS 0x40004L
FILETYPE 0x1L
FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904B0"
        BEGIN
            VALUE "CompanyName", "CRV Science Network"
            VALUE "FileDescription", "AeroFlyway Avian Migration, NEXRAD Radar & Bioacoustic Workstation"
            VALUE "FileVersion", "0.6.0.0"
            VALUE "InternalName", "aeroflyway"
            VALUE "LegalCopyright", "Copyright CRV Science Network (GPL-3.0-or-later)"
            VALUE "OriginalFilename", "aeroflyway.exe"
            VALUE "ProductName", "AeroFlyway"
            VALUE "ProductVersion", "0.6.0.0"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x0409, 1200
    END
END
"#
    );

    let rc_path = out_dir.join("aeroflyway.rc");
    if let Err(err) = fs::write(&rc_path, rc_content) {
        println!("cargo:warning=Failed writing {}: {err}", rc_path.display());
        return;
    }

    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_env == "gnu" {
        let obj_path = out_dir.join("aeroflyway_res.o");
        for tool in ["windres", "x86_64-w64-mingw32-windres"] {
            if let Ok(status) = Command::new(tool)
                .arg("--input")
                .arg(&rc_path)
                .arg("--output")
                .arg(&obj_path)
                .arg("--output-format=coff")
                .status()
            {
                if status.success() {
                    println!("cargo:rustc-link-arg={}", obj_path.display());
                    return;
                }
            }
        }
        println!("cargo:warning=windres failed or not found; PE resources skipped");
    } else if target_env == "msvc" {
        let res_path = out_dir.join("aeroflyway_res.res");
        if let Some(rc_exe) = find_msvc_rc() {
            if let Ok(status) = Command::new(&rc_exe)
                .arg("/nologo")
                .arg("/fo")
                .arg(&res_path)
                .arg(&rc_path)
                .status()
            {
                if status.success() {
                    println!("cargo:rustc-link-arg={}", res_path.display());
                    return;
                }
            }
        }
        println!("cargo:warning=rc.exe failed or not found; PE resources skipped");
    }
}

fn find_msvc_rc() -> Option<PathBuf> {
    if let Ok(status) = Command::new("rc.exe").arg("/?").output() {
        if status.status.success() || !status.stdout.is_empty() || !status.stderr.is_empty() {
            return Some(PathBuf::from("rc.exe"));
        }
    }
    let arch_dir = match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("x86") => "x86",
        Ok("aarch64") => "arm64",
        _ => "x64",
    };
    if let Ok(sdk_dir) = env::var("WindowsSdkDir") {
        if let Some(found) = search_rc_under_bin(&PathBuf::from(sdk_dir).join("bin"), arch_dir) {
            return Some(found);
        }
    }
    let kit_bin = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    search_rc_under_bin(kit_bin, arch_dir)
}

fn search_rc_under_bin(bin_dir: &Path, arch_dir: &str) -> Option<PathBuf> {
    let direct = bin_dir.join(arch_dir).join("rc.exe");
    if direct.is_file() {
        return Some(direct);
    }
    let entries = fs::read_dir(bin_dir).ok()?;
    let mut versions: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|entry| entry.path()))
        .filter(|p| p.is_dir())
        .collect();
    versions.sort();
    for ver in versions.into_iter().rev() {
        let candidate = ver.join(arch_dir).join("rc.exe");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
