"""
Validation harness for candidate task instances.

This script validates candidate PRs by:
1. Building a container at the base_commit
2. Applying test_patch and running tests (baseline - tests should fail)
3. Applying gold patch and running tests again (tests should pass)
4. Computing FAIL_TO_PASS and PASS_TO_PASS from the diff

Usage:
    python -m swebench.harness.run_validation \
        --instances_path data/tasks/rich-task-instances.jsonl \
        --log_dir logs/validation \
        --output_path data/tasks/rich-validated.jsonl \
        --max_workers 4
"""

from __future__ import annotations

import docker
import json
import platform
import traceback

if platform.system() == "Linux":
    import resource

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tqdm.auto import tqdm
from typing import Any, Optional

from swebench.harness.constants import (
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    END_TEST_OUTPUT,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
    UTF8,
)
from swebench.harness.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    remove_image,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.create_scripts import (
    make_repo_script_list,
    make_env_script_list,
)
from swebench.harness.test_spec.test_spec import TestSpec
from swebench.versioning.get_versions import get_version


HEREDOC_DELIMITER = "EOF_114329324912"


@dataclass
class ValidationSpec:
    """Spec for validating a candidate instance."""

    instance_id: str
    repo: str
    version: str
    base_commit: str
    patch: str  # Gold patch
    test_patch: str
    repo_script_list: list[str]
    env_script_list: list[str]
    arch: str
    language: str
    docker_specs: dict
    namespace: Optional[str] = None
    base_image_tag: str = "latest"
    env_image_tag: str = "latest"
    instance_image_tag: str = "latest"

    @property
    def setup_env_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.env_script_list)
            + "\n"
        )

    @property
    def install_repo_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.repo_script_list)
            + "\n"
        )

    @property
    def base_image_key(self):
        return f"sweb.base.{MAP_REPO_TO_EXT[self.repo]}.{self.arch}:{self.base_image_tag}"

    @property
    def env_image_key(self):
        import hashlib

        hash_key = str(self.env_script_list)
        hash_object = hashlib.sha256()
        hash_object.update(hash_key.encode("utf-8"))
        hash_value = hash_object.hexdigest()[:22]
        return f"sweb.env.{MAP_REPO_TO_EXT[self.repo]}.{self.arch}.{hash_value}:{self.env_image_tag}"

    @property
    def instance_image_key(self):
        # Use mini-swe-agent compatible naming: swebench/sweb.eval.x86_64.textualize_1776_rich-XXX
        # Docker doesn't allow double underscore, so replace __ with _1776_
        docker_safe_id = self.instance_id.replace("__", "_1776_").lower()
        prefix = f"{self.namespace}/" if self.namespace else "swebench/"
        return f"{prefix}sweb.eval.{self.arch}.{docker_safe_id}:{self.instance_image_tag}"

    @property
    def is_remote_image(self):
        return self.namespace is not None

    def get_instance_container_name(self, run_id=None):
        if not run_id:
            return f"sweb.eval.{self.instance_id}"
        return f"sweb.eval.{self.instance_id.lower()}.{run_id}"

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        raise ValueError(f"Invalid architecture: {self.arch}")

    # Properties needed by build_container
    @property
    def base_dockerfile(self):
        from swebench.harness.dockerfiles import get_dockerfile_base
        from swebench.harness.constants import DEFAULT_DOCKER_SPECS

        return get_dockerfile_base(
            self.platform,
            self.arch,
            self.language,
            **{**DEFAULT_DOCKER_SPECS, **self.docker_specs},
        )

    @property
    def env_dockerfile(self):
        from swebench.harness.dockerfiles import get_dockerfile_env
        from swebench.harness.constants import DEFAULT_DOCKER_SPECS

        return get_dockerfile_env(
            self.platform,
            self.arch,
            self.language,
            self.base_image_key,
            **{**DEFAULT_DOCKER_SPECS, **self.docker_specs},
        )

    @property
    def instance_dockerfile(self):
        from swebench.harness.dockerfiles import get_dockerfile_instance

        return get_dockerfile_instance(self.platform, self.language, self.env_image_key)

    # Dummy properties for TestSpec compatibility
    FAIL_TO_PASS: list[str] = None
    PASS_TO_PASS: list[str] = None

    @property
    def eval_script(self):
        return ""  # We build this dynamically during validation

    @property
    def eval_script_list(self):
        return []


def make_validation_spec(
    instance: dict,
    version: str,
    arch: str = "x86_64",
) -> ValidationSpec:
    """Create a ValidationSpec from a candidate instance."""
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    docker_specs = specs.get("docker_specs", {})
    env_name = "testbed"
    repo_directory = f"/{env_name}"

    repo_script_list = make_repo_script_list(
        specs, repo, repo_directory, base_commit, env_name
    )
    env_script_list = make_env_script_list(instance, specs, env_name)

    return ValidationSpec(
        instance_id=instance[KEY_INSTANCE_ID],
        repo=repo,
        version=version,
        base_commit=base_commit,
        patch=instance["patch"],
        test_patch=instance["test_patch"],
        repo_script_list=repo_script_list,
        env_script_list=env_script_list,
        arch=arch,
        language=MAP_REPO_TO_EXT[repo],
        docker_specs=docker_specs,
    )


def make_test_script(
    instance: dict,
    version: str,
    apply_gold_patch: bool = False,
) -> str:
    """
    Create a test script for validation.

    Args:
        instance: The candidate instance
        version: The version of the repo
        apply_gold_patch: Whether to apply the gold patch before running tests
    """
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    test_patch = instance["test_patch"]
    gold_patch = instance["patch"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    # Create a copy of instance with version added (needed by get_test_cmds)
    instance_with_version = {**instance, "version": version}

    env_name = "testbed"
    repo_directory = f"/{env_name}"

    # Get test files from test_patch
    from swebench.harness.test_spec.utils import get_modified_files, get_test_cmds

    test_files = get_modified_files(test_patch)
    reset_tests_command = (
        f"git checkout {base_commit} {' '.join(test_files)}"
        if test_files
        else 'echo "No test files to reset"'
    )

    # Build command list
    commands = [
        "#!/bin/bash",
        "set -uxo pipefail",  # Don't exit on error, we need to capture test results
        f"cd {repo_directory}",
        f"git config --global --add safe.directory {repo_directory}",
    ]

    # For Python repos, activate conda
    if MAP_REPO_TO_EXT[repo] == "py":
        commands += [
            "source /opt/miniconda3/bin/activate",
            f"conda activate {env_name}",
        ]

    # If there are eval_commands in specs, add them
    if "eval_commands" in specs:
        commands += specs["eval_commands"]

    # Optionally re-install (some repos need this)
    if "install" in specs:
        commands.append(specs["install"])

    # Reset test files to base state
    commands.append(reset_tests_command)

    # Apply test patch
    apply_test_patch = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    commands.append(apply_test_patch)

    # Optionally apply gold patch
    if apply_gold_patch:
        apply_gold = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{gold_patch}\n{HEREDOC_DELIMITER}"
        commands.append(apply_gold)

    # Run tests
    test_command = get_test_command(instance_with_version, specs)
    commands += [
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
    ]

    # Reset test files after done
    commands.append(reset_tests_command)

    return "\n".join(commands) + "\n"


def get_test_command(instance: dict, specs: dict) -> str:
    """Get the test command for an instance."""
    from swebench.harness.test_spec.python import get_test_directives

    test_cmd = specs.get("test_cmd", "pytest -rA")
    # Get test files from the test_patch
    test_directives = get_test_directives(instance)
    if test_directives:
        return " ".join([test_cmd] + test_directives)
    return test_cmd


def parse_test_output(
    log_content: str,
    repo: str,
    version: str,
) -> dict[str, str]:
    """Parse test output to get test status map."""
    if START_TEST_OUTPUT not in log_content or END_TEST_OUTPUT not in log_content:
        return {}

    test_content = log_content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    log_parser = MAP_REPO_TO_PARSER[repo]

    # Create a minimal test_spec-like object for the parser
    class MinimalSpec:
        pass

    spec = MinimalSpec()
    spec.repo = repo
    spec.version = version
    spec.FAIL_TO_PASS = []
    spec.PASS_TO_PASS = []

    status_map = log_parser(test_content, spec)
    if not status_map:
        # Fallback: try parsing entire content
        status_map = log_parser(log_content, spec)

    return status_map


def compute_f2p_p2p(
    baseline_status: dict[str, str],
    gold_status: dict[str, str],
) -> tuple[list[str], list[str]]:
    """
    Compute FAIL_TO_PASS and PASS_TO_PASS from baseline and gold test results.

    FAIL_TO_PASS: Tests that failed in baseline but pass with gold patch
    PASS_TO_PASS: Tests that passed in both baseline and with gold patch

    Returns:
        (fail_to_pass, pass_to_pass)
    """
    from swebench.harness.constants import TestStatus

    fail_to_pass = []
    pass_to_pass = []

    # All tests we know about
    all_tests = set(baseline_status.keys()) | set(gold_status.keys())

    for test in all_tests:
        baseline_result = baseline_status.get(test)
        gold_result = gold_status.get(test)

        baseline_passed = baseline_result in [
            TestStatus.PASSED.value,
            TestStatus.XFAIL.value,
        ]
        gold_passed = gold_result in [TestStatus.PASSED.value, TestStatus.XFAIL.value]

        if not baseline_passed and gold_passed:
            fail_to_pass.append(test)
        elif baseline_passed and gold_passed:
            pass_to_pass.append(test)

    return fail_to_pass, pass_to_pass


def run_tests_in_container(
    container,
    script: str,
    log_dir: Path,
    script_name: str,
    timeout: int,
    logger,
) -> str:
    """Run a test script in a container and return the output."""
    script_path = log_dir / f"{script_name}.sh"
    script_path.write_text(script)
    copy_to_container(container, script_path, PurePosixPath(f"/{script_name}.sh"))

    output, timed_out, runtime = exec_run_with_timeout(
        container, f"/bin/bash /{script_name}.sh", timeout
    )

    output_path = log_dir / f"{script_name}_output.txt"
    output_path.write_text(output)
    logger.info(f"{script_name} completed in {runtime:.2f}s, output saved to {output_path}")

    if timed_out:
        logger.warning(f"{script_name} timed out after {timeout}s")

    return output


def validate_instance(
    instance: dict,
    client: docker.DockerClient,
    log_dir: Path,
    run_id: str,
    timeout: int,
    force_rebuild: bool = False,
    version: str | None = None,
) -> dict | None:
    """
    Validate a single candidate instance.

    Returns the validated instance with FAIL_TO_PASS and PASS_TO_PASS populated,
    or None if validation fails.
    """
    instance_id = instance[KEY_INSTANCE_ID]
    instance_log_dir = log_dir / instance_id
    instance_log_dir.mkdir(parents=True, exist_ok=True)

    log_file = instance_log_dir / "validation.log"
    logger = setup_logger(instance_id, log_file)

    container = None
    repo = instance["repo"]

    try:
        # Use provided version or determine it
        if version is None:
            logger.info(f"Determining version for {instance_id}...")
            version = get_version(instance)
        if version is None:
            logger.error(f"Could not determine version for {instance_id}")
            return None
        logger.info(f"Version: {version}")

        # Create validation spec
        val_spec = make_validation_spec(instance, version)

        # Build container
        logger.info(f"Building container for {instance_id}...")
        container = build_container(val_spec, client, run_id, logger, False, force_rebuild)
        container.start()
        logger.info(f"Container started: {container.id}")

        # Run baseline tests (without gold patch)
        logger.info("Running baseline tests (without gold patch)...")
        baseline_script = make_test_script(instance, version, apply_gold_patch=False)
        baseline_output = run_tests_in_container(
            container, baseline_script, instance_log_dir, "baseline", timeout, logger
        )
        baseline_status = parse_test_output(baseline_output, repo, version)
        logger.info(f"Baseline: {len(baseline_status)} tests parsed")

        if not baseline_status:
            logger.error("No test results from baseline run")
            return None

        # Run tests with gold patch
        logger.info("Running tests with gold patch...")
        gold_script = make_test_script(instance, version, apply_gold_patch=True)
        gold_output = run_tests_in_container(
            container, gold_script, instance_log_dir, "gold", timeout, logger
        )
        gold_status = parse_test_output(gold_output, repo, version)
        logger.info(f"Gold: {len(gold_status)} tests parsed")

        if not gold_status:
            logger.error("No test results from gold run")
            return None

        # Compute FAIL_TO_PASS and PASS_TO_PASS
        fail_to_pass, pass_to_pass = compute_f2p_p2p(baseline_status, gold_status)
        logger.info(f"FAIL_TO_PASS: {len(fail_to_pass)}, PASS_TO_PASS: {len(pass_to_pass)}")

        # Validation criteria: must have at least one FAIL_TO_PASS test
        if not fail_to_pass:
            logger.warning(f"No FAIL_TO_PASS tests found for {instance_id}, skipping")
            return None

        # Create validated instance
        validated = {
            **instance,
            "version": version,
            "FAIL_TO_PASS": json.dumps(fail_to_pass),
            "PASS_TO_PASS": json.dumps(pass_to_pass),
        }

        # Save result
        result_path = instance_log_dir / "validated.json"
        result_path.write_text(json.dumps(validated, indent=2))
        logger.info(f"Validation successful! Result saved to {result_path}")

        return validated

    except BuildImageError as e:
        logger.error(f"Build error: {e}")
        return None
    except Exception as e:
        logger.error(f"Validation failed: {e}\n{traceback.format_exc()}")
        return None
    finally:
        cleanup_container(client, container, logger)
        close_logger(logger)


def load_instances(path: str) -> list[dict]:
    """Load instances from a JSONL file."""
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def save_instances(instances: list[dict], path: str):
    """Save instances to a JSONL file."""
    with open(path, "w") as f:
        for instance in instances:
            f.write(json.dumps(instance) + "\n")


def determine_versions(instances: list[dict]) -> dict[str, str]:
    """Determine versions for all instances."""
    versions = {}
    print(f"Determining versions for {len(instances)} instances...")
    for instance in tqdm(instances, desc="Versioning"):
        instance_id = instance[KEY_INSTANCE_ID]
        try:
            version = get_version(instance)
            if version:
                versions[instance_id] = version
            else:
                print(f"  Warning: Could not determine version for {instance_id}")
        except Exception as e:
            print(f"  Error getting version for {instance_id}: {e}")
    return versions


def build_validation_images(
    client: docker.DockerClient,
    instances: list[dict],
    versions: dict[str, str],
    force_rebuild: bool,
    max_workers: int,
):
    """Build Docker images for validation."""
    from swebench.harness.docker_build import (
        build_image,
        BASE_IMAGE_BUILD_DIR,
        ENV_IMAGE_BUILD_DIR,
    )
    from swebench.harness.docker_utils import remove_image

    # Create validation specs for instances with known versions
    val_specs = []
    for instance in instances:
        instance_id = instance[KEY_INSTANCE_ID]
        if instance_id not in versions:
            continue
        version = versions[instance_id]
        repo = instance["repo"]
        if repo not in MAP_REPO_VERSION_TO_SPECS:
            continue
        if version not in MAP_REPO_VERSION_TO_SPECS[repo]:
            continue
        try:
            val_spec = make_validation_spec(instance, version)
            val_specs.append(val_spec)
        except Exception as e:
            print(f"  Error creating spec for {instance_id}: {e}")

    if not val_specs:
        print("No valid specs to build images for")
        return

    print(f"Building images for {len(val_specs)} instances...")

    # Get unique base images to build
    base_images = {
        spec.base_image_key: (spec.base_dockerfile, spec.platform)
        for spec in val_specs
    }
    print(f"  Base images needed: {len(base_images)}")

    # Build base images
    for image_name, (dockerfile, platform) in base_images.items():
        try:
            client.images.get(image_name)
            if force_rebuild:
                remove_image(client, image_name, "quiet")
            else:
                print(f"  Base image {image_name} already exists, skipping.")
                continue
        except docker.errors.ImageNotFound:
            pass
        print(f"  Building base image: {image_name}")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        )

    # Get unique env images to build
    env_images = {
        spec.env_image_key: (spec.env_dockerfile, spec.platform, spec.setup_env_script)
        for spec in val_specs
    }
    print(f"  Env images needed: {len(env_images)}")

    # Build env images
    for image_name, (dockerfile, platform, setup_script) in env_images.items():
        try:
            client.images.get(image_name)
            if force_rebuild:
                remove_image(client, image_name, "quiet")
            else:
                print(f"  Env image {image_name} already exists, skipping.")
                continue
        except docker.errors.ImageNotFound:
            pass
        print(f"  Building env image: {image_name}")
        build_image(
            image_name=image_name,
            setup_scripts={"setup_env.sh": setup_script},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=ENV_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        )

    print("Images built successfully.")


def main(
    instances_path: str,
    log_dir: str,
    output_path: str,
    max_workers: int = 4,
    timeout: int = 1800,
    force_rebuild: bool = False,
    open_file_limit: int = 4096,
    run_id: str = "validation",
    instance_ids: list[str] | None = None,
):
    """Run validation for candidate instances."""
    # Set open file limit on Linux
    if platform.system() == "Linux":
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    # Load instances
    instances = load_instances(instances_path)
    print(f"Loaded {len(instances)} candidate instances from {instances_path}")

    # Filter by instance_ids if provided
    if instance_ids:
        instances = [i for i in instances if i[KEY_INSTANCE_ID] in instance_ids]
        print(f"Filtered to {len(instances)} instances")

    # Setup
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    client = docker.from_env()

    # First, determine versions for all instances
    versions = determine_versions(instances)
    print(f"Determined versions for {len(versions)}/{len(instances)} instances")

    # Filter to instances with valid versions and specs
    valid_instances = []
    for instance in instances:
        instance_id = instance[KEY_INSTANCE_ID]
        if instance_id not in versions:
            print(f"  Skipping {instance_id}: no version")
            continue
        version = versions[instance_id]
        repo = instance["repo"]
        if repo not in MAP_REPO_VERSION_TO_SPECS:
            print(f"  Skipping {instance_id}: repo {repo} not in specs")
            continue
        if version not in MAP_REPO_VERSION_TO_SPECS[repo]:
            print(f"  Skipping {instance_id}: version {version} not in specs for {repo}")
            continue
        valid_instances.append(instance)

    print(f"Valid instances to validate: {len(valid_instances)}")

    if not valid_instances:
        print("No valid instances to validate")
        return

    # Build env images
    print("Building environment images...")
    build_validation_images(client, valid_instances, versions, force_rebuild, max_workers)

    # Validate instances
    validated = []
    failed = []

    print(f"Validating {len(valid_instances)} instances with {max_workers} workers...")

    # Pass versions to validate_instance
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                validate_instance,
                instance,
                client,
                log_dir,
                run_id,
                timeout,
                force_rebuild,
                versions.get(instance[KEY_INSTANCE_ID]),
            ): instance
            for instance in valid_instances
        }

        pbar = tqdm(total=len(futures), desc="Validation")
        for future in as_completed(futures):
            instance = futures[future]
            instance_id = instance[KEY_INSTANCE_ID]
            try:
                result = future.result()
                if result:
                    validated.append(result)
                    pbar.set_postfix({"valid": len(validated), "failed": len(failed)})
                else:
                    failed.append(instance_id)
            except Exception as e:
                print(f"Error validating {instance_id}: {e}")
                failed.append(instance_id)
            pbar.update()
        pbar.close()

    # Save results
    print(f"\nValidation complete: {len(validated)} valid, {len(failed)} failed")
    if validated:
        save_instances(validated, output_path)
        print(f"Saved {len(validated)} validated instances to {output_path}")

    if failed:
        failed_path = Path(output_path).with_suffix(".failed.json")
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"Failed instance IDs saved to {failed_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Validate candidate task instances",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--instances_path",
        type=str,
        required=True,
        help="Path to candidate instances JSONL file",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/validation",
        help="Directory for validation logs",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for validated instances output",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for each test run",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild Docker images",
    )
    parser.add_argument(
        "--open_file_limit",
        type=int,
        default=4096,
        help="Open file limit",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="validation",
        help="Run ID for this validation run",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Specific instance IDs to validate (optional)",
    )

    args = parser.parse_args()
    main(**vars(args))

