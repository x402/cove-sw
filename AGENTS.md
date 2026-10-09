# AGENTS.md — cove-sw（RDSM + TSM CoVE 软件栈）

> 开发纪律（规范依据、跨仓库依赖方向、提交规范）见顶层 `/home/x402/CoVE/AGENTS.md`。

## 仓库定位（2026-09-11 重构定稿）

RDSM（M-mode 固件层）与 TSM（HS-mode 机密域内核）运行在不同特权级，但二者之间的
SBI 协议是私有的、不受其他组件影响，因此合并为本仓库统一演进。本仓库由原 `tsm`
仓库演化而来（保留全部 git 历史），并吸收了原 rustsbi fork 中的 rdsm 代码。

## 目录结构

- `crates/rdsm-abi` — RDSM↔TSM 私有 ABI 的唯一定义（EID/FID 常量、cove-payload 头、
  `RdsmPlatformInfo`）。tsm/test-host/xtask/固件侧一律依赖此 crate，**禁止再复制常量**。
- `crates/rdsm` — RDSM 基础库（MPT 树、CSR 编解码、域/中断域管理、trap-safe 探测），
  no_std、除 rdsm-abi 外零依赖。
- `crates/rdsm-mech` — 机制层（unsafe 集中）：裸 CSR 读写、per-hart 保存区
  （`DomainCsrs`/`Thcs`、`PerHartCell`）、MPT 页池、mmpt/msdcfg 编程与域切换原语。
- `crates/rdsm-policy` — 策略层（`#![forbid(unsafe_code)]`）：RDSM 初始化与 CoVE
  payload 加载、私有 SBI 分发（RDSM/SUPD/COVH/COVI）、TEECALL/TEERET 事务、MSDEI
  咨询钩子；经 `InitEnv` 注入固件环境并 path 依赖 `../rustsbi/firmware/runtime`，
  由 `../rustsbi` 的 `rdsm` cargo feature 编译期链接进 Prototyper。
- `tsm/`、`test-host/`、`test-guest/` — HS-mode TSM 内核与测试 VMM/负载。
- `xtask/` — `cargo xtask pack` 打包 cove-payload.bin（TSM@0x80400000, Host@0x80800000）。
- `vendor/` — vendored 依赖（data-model、riscv-page-tables 等，来自 Rivos salus 系）。
- `tests/e2e/run_e2e.sh` — 一键端到端回归（17 个串行 marker 顺序校验）。

## 构建与验证

```bash
cd /home/x402/CoVE/cove-sw
cargo xtask pack                 # 打包 cove-payload.bin
cargo check -p tsm -p test-host -p test-guest --target riscv64gc-unknown-none-elf
tests/e2e/run_e2e.sh             # 全链路回归（需要 ../rustsbi 与 ../NEMU 就位）
cargo test -p rdsm-abi           # ABI 布局单测（宿主机）
```

## 提交规范

- DCO：`Signed-off-by: x402 <akr01@qq.com>`；Conventional Commits（英文）。
- GitHub 远程：`x402/cove-sw`（public）。
