// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use sbi_rt::Physical;
use sync::{Mutex, Once};

use crate::print::{Console, ConsoleDriver};

/// Driver for an SBI based console.
pub struct SbiConsole {
    buffer: Mutex<&'static mut [u8]>,
}

static SBI_CONSOLE: Once<SbiConsole> = Once::new();

impl SbiConsole {
    /// Sets the SBI debug console as the system console. Uses `console_buffer` for buffering console
    /// output.
    pub fn set_as_console(console_buffer: &'static mut [u8]) {
        let console = SbiConsole {
            buffer: Mutex::new(console_buffer),
        };
        SBI_CONSOLE.call_once(|| console);
        Console::set_driver(SBI_CONSOLE.get().unwrap());
    }
}

impl ConsoleDriver for SbiConsole {
    /// Write an entire byte sequence to the SBI console.
    fn write_bytes(&self, bytes: &[u8]) {
        let mut buffer = self.buffer.lock();
        for chunk in bytes.chunks(buffer.len()) {
            let (dest, _) = buffer.split_at_mut(chunk.len());
            dest.copy_from_slice(chunk);
            // Ignore errors as there isn't currently a way to report them if the console doesn't work.
            let _ = sbi_rt::console_write(Physical::new(dest.len(), dest.as_ptr() as usize, 0));
        }
    }
}

/// Driver for the legacy SBI console from v0.1 of the SBI spec.
pub struct SbiConsoleV01 {}

static SBI_CONSOLE_V01: SbiConsoleV01 = SbiConsoleV01 {};

impl SbiConsoleV01 {
    /// Sets the SBI legacy console as the system console.
    pub fn set_as_console() {
        Console::set_driver(&SBI_CONSOLE_V01);
    }
}

impl ConsoleDriver for SbiConsoleV01 {
    /// Write an entire byte sequence to the SBI console.
    fn write_bytes(&self, bytes: &[u8]) {
        for &b in bytes {
            let _ = sbi_rt::legacy::console_putchar(b as usize);
        }
    }
}
