OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
    . = 0x80400000;

    .text : ALIGN(4K) {
        *(.text.entry)
        *(.text .text.*)
    }

    .rodata : ALIGN(4K) {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    }

    .data : ALIGN(4K) {
        *(.data .data.*)
        *(.sdata .sdata.*)
    }

    .bss (NOLOAD) : ALIGN(4K) {
        sbss = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        ebss = .;
    }

    .stack (NOLOAD) : ALIGN(4K) {
        . += 0x10000;
        boot_stack_top = .;
    }

    /DISCARD/ : {
        *(.eh_frame)
    }
}
