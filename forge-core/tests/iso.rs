use forge_core::isolation::run_with_timeout;
use forge_core::ForgeError;
use std::process::Command;
use std::time::Duration;

#[test]
fn kills_long_running_subprocess() {
    let mut cmd = Command::new("sleep");
    cmd.arg("30");
    let err = run_with_timeout(cmd, Duration::from_millis(150)).unwrap_err();
    match err {
        ForgeError::Evaluation(m) => assert!(m.contains("Timeout"), "got: {m}"),
        other => panic!("expected Evaluation timeout, got {other:?}"),
    }
}

#[test]
fn captures_success_output() {
    let mut cmd = Command::new("echo");
    cmd.arg("hello-forge");
    let out = run_with_timeout(cmd, Duration::from_secs(2)).expect("echo doit reussir");
    assert!(out.contains("hello-forge"), "stdout: {out}");
}
