//! SBI return encoding shared by the dispatch layer and the library modules.

/// Two-register SBI return value (`a0` = error, `a1` = value).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SbiRet {
    pub error: usize,
    pub value: usize,
}

impl SbiRet {
    pub const fn success(value: usize) -> Self {
        Self { error: 0, value }
    }
    pub const fn failed() -> Self {
        Self {
            error: (-1isize) as usize,
            value: 0,
        }
    }
    pub const fn not_supported() -> Self {
        Self {
            error: (-2isize) as usize,
            value: 0,
        }
    }
    pub const fn invalid_param() -> Self {
        Self {
            error: (-3isize) as usize,
            value: 0,
        }
    }
    pub const fn invalid_address() -> Self {
        Self {
            error: (-5isize) as usize,
            value: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SbiRet;

    #[test]
    fn success_carries_the_value() {
        let ret = SbiRet::success(5);
        assert_eq!(ret.error, 0);
        assert_eq!(ret.value, 5);
    }

    #[test]
    fn error_constructors_use_the_sbi_error_codes() {
        assert_eq!(SbiRet::failed().error, (-1isize) as usize);
        assert_eq!(SbiRet::not_supported().error, (-2isize) as usize);
        assert_eq!(SbiRet::invalid_param().error, (-3isize) as usize);
        assert_eq!(SbiRet::invalid_address().error, (-5isize) as usize);
        for ret in [
            SbiRet::failed(),
            SbiRet::not_supported(),
            SbiRet::invalid_param(),
            SbiRet::invalid_address(),
        ] {
            assert_eq!(ret.value, 0);
        }
    }
}
