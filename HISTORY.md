Changelog
=========


(unreleased)
------------
- Feat(video_ql/VERSION): increment. [AdityaNG]
- Feat(model): moondream added. [AdityaNG]
- Feat(models): gemini support. [AdityaNG]


1.4.0 (2025-07-16)
------------------
- Release: version 1.4.0 🚀 [AdityaNG]
- Video_ql/VERSION increment. [AdityaNG]
- Fix(video_ql/models.py): default config to match requirements.
  [AdityaNG]
- Feat(pricing): model pricing estimation support. [AdityaNG]


1.3.0 (2025-05-18)
------------------
- Release: version 1.3.0 🚀 [AdityaNG]
- Feat(video_ql): query_proposer - video_ql/cli.py: updated to use NLP
  query proposer - README.md: docs for query_proposer -
  requirements.txt: rich added for CLI UX - tests/test_utils.py: timing
  adjusted - video_ql/base.py: helper calls to query_proposer -
  video_ql/query_proposer.py: linting - video_ql/yaml_analysis.py: moved
  original CLI tool here - video_ql/VERSION: incrementy. [AdityaNG]
- Feat(video_ql): query_proposer - video_ql/query_proposer.py: query
  proposer takes the context as input and produces the list of possible
  queries as input - tests/test_query_proposer.py: test cases -
  README.md: updated with usage instructions - tests/test_utils.py:
  timing relaxed - video_ql/cli.py: fixed, removed unintended exit -
  VERSION: increment. [AdityaNG]


1.2.0 (2025-05-17)
------------------
- Release: version 1.2.0 🚀 [AdityaNG]
- Feat(video_ql/base.py): parallel processing - video_ql/base.py: cache
  thread safety - video_ql/cli.py: take advantage of multithreading for
  parallel API calls - video_ql/utils.py - tests/: updated test case -
  video_ql/VERSION: incremented. [AdityaNG]

  Measured speedup
  100 threads
  real    0m29.947s
  user    1m27.712s
  sys     0m14.065s

  1 thread
  real    6m33.628s
  user    0m28.482s
  sys     0m9.564s


1.1.0 (2025-05-17)
------------------
- Release: version 1.1.0 🚀 [AdityaNG]
- Feat(video_ql): single frame analysis, fast frame count - setup.py:
  package desc change - tests/test_single_frame.py: single frame test
  cases - README.md: single frame usage example -
  video_ql/single_frame.py: `SingleFrameAnalyzer` - video_ql/utils.py:
  faster `get_length_of_video`, `video_hash` implementations -
  tests/test_utils.py: time based tests for the above - VERSION:
  increment. [AdityaNG]


1.0.0 (2025-05-09)
------------------
- Release: version 1.0.0 🚀 [AdityaNG]
- Test(tests/test_video_ql.py): coverage to 48. [AdityaNG]
- Test(tests/test_query.py): converage to 54. [AdityaNG]
- Test(tests/test_utils.py): converage to 48. [AdityaNG]
- Test(tests/test_visualization.py): coverage to 40. [AdityaNG]
- Ci(.github/workflows/main.yml): codecov. [AdityaNG]
- Merge pull request #4 from
  AdityaNG/dependabot/github_actions/actions/setup-python-5. [Aditya]

  Bump actions/setup-python from 4 to 5
- Bump actions/setup-python from 4 to 5. [dependabot[bot]]

  Bumps [actions/setup-python](https://github.com/actions/setup-python) from 4 to 5.
  - [Release notes](https://github.com/actions/setup-python/releases)
  - [Commits](https://github.com/actions/setup-python/compare/v4...v5)

  ---
  updated-dependencies:
  - dependency-name: actions/setup-python
    dependency-version: '5'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...
- Merge pull request #5 from
  AdityaNG/dependabot/github_actions/codecov/codecov-action-5. [Aditya]

  Bump codecov/codecov-action from 3 to 5
- Bump codecov/codecov-action from 3 to 5. [dependabot[bot]]

  Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 3 to 5.
  - [Release notes](https://github.com/codecov/codecov-action/releases)
  - [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/codecov/codecov-action/compare/v3...v5)

  ---
  updated-dependencies:
  - dependency-name: codecov/codecov-action
    dependency-version: '5'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...
- Merge pull request #3 from
  AdityaNG/dependabot/github_actions/stefanzweifel/git-auto-commit-
  action-5. [Aditya]

  Bump stefanzweifel/git-auto-commit-action from 4 to 5
- Bump stefanzweifel/git-auto-commit-action from 4 to 5.
  [dependabot[bot]]

  Bumps [stefanzweifel/git-auto-commit-action](https://github.com/stefanzweifel/git-auto-commit-action) from 4 to 5.
  - [Release notes](https://github.com/stefanzweifel/git-auto-commit-action/releases)
  - [Changelog](https://github.com/stefanzweifel/git-auto-commit-action/blob/master/CHANGELOG.md)
  - [Commits](https://github.com/stefanzweifel/git-auto-commit-action/compare/v4...v5)

  ---
  updated-dependencies:
  - dependency-name: stefanzweifel/git-auto-commit-action
    dependency-version: '5'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...
- Merge pull request #2 from
  AdityaNG/dependabot/github_actions/softprops/action-gh-release-2.
  [Aditya]

  Bump softprops/action-gh-release from 1 to 2
- Bump softprops/action-gh-release from 1 to 2. [dependabot[bot]]

  Bumps [softprops/action-gh-release](https://github.com/softprops/action-gh-release) from 1 to 2.
  - [Release notes](https://github.com/softprops/action-gh-release/releases)
  - [Changelog](https://github.com/softprops/action-gh-release/blob/master/CHANGELOG.md)
  - [Commits](https://github.com/softprops/action-gh-release/compare/v1...v2)

  ---
  updated-dependencies:
  - dependency-name: softprops/action-gh-release
    dependency-version: '2'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...
- Merge pull request #1 from
  AdityaNG/dependabot/github_actions/actions/checkout-4. [Aditya]

  Bump actions/checkout from 3 to 4
- Bump actions/checkout from 3 to 4. [dependabot[bot]]

  Bumps [actions/checkout](https://github.com/actions/checkout) from 3 to 4.
  - [Release notes](https://github.com/actions/checkout/releases)
  - [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/actions/checkout/compare/v3...v4)

  ---
  updated-dependencies:
  - dependency-name: actions/checkout
    dependency-version: '4'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...
- Update LICENSE. [Aditya]
- Feat(video_ql/models.py): query config is now a pydantic model.
  [AdityaNG]
- Docs(assets/video_ql_forklift_demo.gif): capitalization. [AdityaNG]
- Fix(requirements.txt): minimal. [AdityaNG]
- Docs(assets/video_ql_forklift_demo.gif): demo. [AdityaNG]
- Refactor(video_ql/base): making the base class have less
  responisibility. [AdityaNG]
- Refactor(video_ql/visualization.py): vis in seperate file. [AdityaNG]
- Feat(video_ql/query.py): query logic in seperate file. [AdityaNG]
- Fmt(video_ql): linting passing. [AdityaNG]
- Feat(video_ql): anthropic model support, disable cache option, minor
  bug fixes. [AdityaNG]
- Init. [AdityaNG]
- ✅ Ready to clone and code. [AdityaNG]
- Initial commit. [Aditya]


