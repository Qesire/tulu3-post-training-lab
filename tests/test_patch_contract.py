from scripts.setup.apply_lecture6_open_instruct_patch import NEW_INIT, NEW_URL, OLD_URL, patch_text


def test_lecture6_patch_rewrites_exact_three_urls_and_guards_tracker():
    text = f'''x={OLD_URL}\n{OLD_URL}\n        wandb_tracker = accelerator.get_tracker("wandb")\n        maybe_update_beaker_description(wandb_url={OLD_URL})\n'''
    patched, status = patch_text(text)
    assert status == "patched"
    assert OLD_URL not in patched
    assert patched.count(NEW_URL) == 3
    assert NEW_INIT in patched


def test_lecture6_patch_is_idempotent():
    text = f'''x={NEW_URL}\ny={NEW_URL}\n{NEW_INIT}\n'''
    patched, status = patch_text(text)
    assert status == "already_patched"
    assert patched == text
