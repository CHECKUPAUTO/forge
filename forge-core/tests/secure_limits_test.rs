use std::process::Command;
use std::time::Duration;
use forge_core::isolation::run_with_secure_limits;

fn echo_cmd(msg: &str) -> Command {
    let mut cmd = Command::new("echo");
    cmd.arg(msg);
    cmd
}

fn sleep_cmd(secs: &str) -> Command {
    let mut cmd = Command::new("sleep");
    cmd.arg(secs);
    cmd
}

fn dd_cmd() -> Command {
    let mut cmd = Command::new("dd");
    cmd.arg("if=/dev/zero")
       .arg("of=/tmp/forge_limits_test.bin")
       .arg("bs=1M")
       .arg("count=2");
    cmd
}

#[test]
fn test_secure_limits_allows_small_process() {
    let result = run_with_secure_limits(
        echo_cmd("hello-secure"),
        Duration::from_secs(5),
        512 * 1024 * 1024,
        10 * 1024 * 1024,
    );
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("hello-secure"));
}

#[test]
fn test_secure_limits_kills_slow_process() {
    let result = run_with_secure_limits(
        sleep_cmd("10"),
        Duration::from_millis(100),
        512 * 1024 * 1024,
        10 * 1024 * 1024,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Timeout") || err.contains("timeout"));
}

#[test]
fn test_secure_limits_file_size_limit() {
    let result = run_with_secure_limits(
        dd_cmd(),
        Duration::from_secs(5),
        512 * 1024 * 1024,
        1 * 1024 * 1024, // 1 MB file size limit
    );
    let _ = std::fs::remove_file("/tmp/forge_limits_test.bin");
    assert!(result.is_err());
}
