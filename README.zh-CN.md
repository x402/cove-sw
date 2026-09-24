# cove-sw — RISC-V CoVE 软件栈(RDSM + TSM)

[English](README.md) | [简体中文](README.zh-CN.md)

`cove-sw` 是面向实现 Smmtt(Supervisor Domains Access Protection)的 RISC-V 平台的
CoVE(Confidential VM Extension)软件栈:包含 M-mode 的 **RDSM**(Root Domain
Security Manager)固件层与 HS-mode 的 **TSM**(TEE Security Manager)内核。二者
之间的 SBI 协议是本栈私有的,因此在同一仓库中统一演进。

## 功能概览

- **RDSM**(M-mode,链接进 SBI 固件):编程机器级内存保护表(`mmpt` CSR / MPT),
  并在非机密宿主域(SDID=0)与机密域(SDID=1)之间执行 TEECALL/TEERET 域切换。
- **TSM**(HS-mode,机密域):管理 TVM 生命周期、将物理页转换为机密内存、构建
  G-stage 页表并运行 vCPU 上下文。
- **test-host**(HS-mode,宿主域):端到端驱动 CoVE 宿主扩展的测试 VMM;
  **test-guest** 是运行在 TVM 内的 VS-mode 负载。

全栈在 NEMU(Smmtt v0.49 模型,`smmtt` 分支)上,通过与 RustSBI Prototyper
固件(`rdsm` cargo feature 编译期链接 RDSM 层)完成端到端验证。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `crates/rdsm-abi` | RDSM↔TSM 私有 ABI 唯一定义:EID/FID 常量、`cove-payload` 头、平台信息类型 |
| `crates/rdsm` | RDSM 基础库(`no_std`,除 rdsm-abi 外零依赖):MPT 树、CSR 编解码、域状态、trap-safe 探测 |
| `crates/rdsm-mech` | RDSM 机制层——全部 `unsafe` 硬件访问集中于此(对外 API 全安全):裸汇编 CSR 保存/恢复、`mmpt`/中断域编程、镜像加载、per-hart 存储 |
| `crates/rdsm-policy` | RDSM 策略层——零 `unsafe` 的时序编排:启动初始化、私有 SBI 分发(RDSM/SUPD/COVH/COVI)、TEECALL/TEERET |
| `tsm` | HS-mode TSM 内核 |
| `test-host` | HS-mode 测试 VMM(宿主域) |
| `test-guest` | VS-mode 测试负载(TVM 工作负载) |
| `xtask` | `cargo xtask pack`:将 TSM @ 0x80400000 与 test-host @ 0x80800000 打包为 `cove-payload.bin` |
| `vendor/` | vendored 依赖(`data-model`、`riscv-page-tables` 等,来自 Rivos `salus` 系列) |
| `tests/e2e` | 端到端回归脚本 |
| `docs/rdsm-tsm-protocol.md` | RDSM↔TSM 接口与启动协议规范 |

## 全栈源码布局

固件构建使用 path 依赖,三个仓库需同级检出:

```
<root>/
├── NEMU/      github.com/x402/NEMU,分支 smmtt      — Smmtt v0.49 硬件模型
├── rustsbi/   github.com/x402/rustsbi              — Prototyper 固件(rdsm feature)
└── cove-sw/   本仓库
```

## 构建

工具链 nightly 固定于 `rust-toolchain.toml`;目标 `riscv64gc-unknown-none-elf`。

```bash
cargo xtask pack
# → target/riscv64gc-unknown-none-elf/release/cove-payload.bin
cargo check -p tsm -p test-host -p test-guest --target riscv64gc-unknown-none-elf
```

## 端到端回归

```bash
tests/e2e/run_e2e.sh
```

脚本打包 payload、构建带 `rdsm` feature 的 Prototyper 固件、启动 NEMU,并按序
校验 17 个串行 marker,覆盖完整 TEECALL/TEERET 生命周期:RDSM 初始化 → TSM
就绪 → 宿主扩展发现 → 机密页转换 → TVM 创建/终结/运行 → guest 退出 → TVM
拆除 → 页回收。

## 规范基线

| 规范 | 版本 |
|---|---|
| Smmtt(Supervisor Domains Access Protection) | v0.49 draft |
| CoVE ABI | v0.7 |
| RISC-V SBI | v3.0 |

## 状态

实验性研究项目,持续开发中。全局状态按多 hart 安全标准实现(per-hart 化);
当前运行时验证范围为 NEMU 单 hart。
