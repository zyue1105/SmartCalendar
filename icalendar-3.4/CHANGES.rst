
Changelog
=========

3.4 (2013-04-24)
----------------

- Switch to unicode internally. This should fix all en/decoding errors.
  [thet]

- Support for non-ascii parameter values. Fixes #88.
  [warvariuc]

- Added functions to transform chars in string with '\\' + any of r'\,;:' chars
  into '%{:02X}' form to avoid splitting on chars escaped with '\\'.
  [warvariuc]

- Allow seconds in vUTCOffset properties. Fixes #55.
  [thet]

- Let ``Component.decode`` better handle vRecur and vDDDLists properties.
  Fixes #70.
  [thet]

- Don't let ``Component.add`` re-encode already encoded values. This simplifies
  the API, since there is no need explicitly pass ``encode=False``. Fixes #82.
  [thet]

- Rename tzinfo_from_dt to tzid_from_dt, which is what it does.
  [thet]

- More support for dateutil parsed tzinfo objects. Fixes #89.
  [leo-naeka]

- Remove python-dateutil version fix at all. Current python-dateutil has Py3
  and Py2 compatibility.
  [thet]

- Declare the required python-dateutil dependency in setup.py. Fixes #90.
  [kleink]

- Raise test coverage.
  [thet]

- Remove interfaces module, as it is unused.
  [thet]

- Remove ``test_doctests.py``, test suite already created properly in
  ``test_icalendar.py``.
  [rnix]

- Transformed doctests into unittests, Test fixes and cleanup.
  [warvariuc]


3.3 (2013-02-08)
----------------

- Drop support for Python < 2.6.
  [thet]

- Allow vGeo to be instantiated with list and not only tuples of geo
  coordinates. Fixes #83.
  [thet]

- Don't force to pass a list to vDDDLists and allow setting individual RDATE
  and EXDATE values without having to wrap them in a list.
  [thet]

- Fix encoding function to allow setting RDATE and EXDATE values and not to
  have bypass encoding with an icalendar property.
  [thet]

- Allow setting of timezone for vDDDLists and support timezone properties for
  RDATE and EXDATE component properties.
  [thet]

- Move setting of TZID properties to vDDDTypes, where it belongs to.
  [thet]

- Use @staticmethod decorator instead of wrapper function.
  [warvariuc, thet]

- Extend quoting of parameter values to all of those characters: ",;: ’'".
  This fixes an outlook incompatibility with some characters. Fixes: #79,
  Fixes: #81.
  [warvariuc]

- Define VTIMETZONE subcomponents STANDARD and DAYLIGHT for RFC5545 compliance.
  [thet]


3.2 (2012-11-27)
----------------

- Documentation file layout restructuring.
  [thet]

- Fix time support. vTime events can be instantiated with a datetime.time
  object, and do not inherit from datetime.time itself.
  [rdunklau]

- Correctly handle tzinfo objects parsed with dateutil. Fixes #77.
  [warvariuc, thet]

- Text values are escaped correclty. Fixes #74.
  [warvariuc]

- Returned old folding algorithm, as the current implementation fails in some
  cases. Fixes #72, Fixes #73.
  [warvariuc]

- Supports to_ical() on date/time properties for dates prior to 1900.
  [cdevienne]


3.1 (2012-09-05)
----------------

- Make sure parameters to certain properties propagate to the ical output.
  [kanarip]

- Re-include doctests.
  [rnix]

- Ensure correct datatype at instance creation time in ``prop.vCalAddress``
  and ``prop.vText``.
  [rnix]

- Apply TZID parameter to datetimes parsed from RECURRENCE-ID
  [dbstovall]

- Localize datetimes for timezones to avoid DST transition errors.
  [dbstovall]

- Allow UTC-OFFSET property value data types in seconds, which follows RFC5545
  specification.
  [nikolaeff]

- Remove utctz and normalized_timezone methods to simplify the codebase. The
  methods were too tiny to be useful and just used at one place.
  [thet]

- When using Component.add() to add icalendar properties, force a value
  conversion to UTC for CREATED, DTSTART and LAST-MODIFIED. The RFC expects UTC
  for those properties.
  [thet]

- Removed last occurrences of old API (from_string).
  [Rembane]

- Add 'recursive' argument to property_items() to switch recursive listing.
  For example when parsing a text/calendar text including multiple components
  (e.g. a VCALENDAR with 5 VEVENTs), the previous situation required us to look
  over all properties in VEVENTs even if we just want the properties under the
  VCALENDAR component (VERSION, PRODID, CALSCALE, METHOD).
  [dmikurube]

- All unit tests fixed.
  [mikaelfrykholm]


3.0.1b2 (2012-03-01)
--------------------

- For all TZID parameters in DATE-TIME properties, use timezone identifiers
  (e.g. Europe/Vienna) instead of timezone names (e.g. CET), as required by
  RFC5545. Timezone names are used together with timezone identifiers in the
  Timezone components.
  [thet]

- Timezone parsing, issues and test fixes.
  [mikaelfrykholm, garbas, tgecho]

- Since we use pytz for timezones, also use UTC tzinfo object from the pytz
  library instead of own implementation.
  [thet]


3.0.1b1 (2012-02-24)
--------------------

- Update Release information.
  [thet]


3.0
---

- Add API for proper Timezone support. Allow creating ical DATE-TIME strings
  with timezone information from Python datetimes with pytz based timezone
  information and vice versa.
  [thet]

- Unify API to only use to_ical and from_ical and remove string casting as a
  requirement for Python 3 compatibility:
  New: to_ical.
  Old: ical, string, as_string and string casting via __str__ and str.
  New: from_ical.
  Old: from_string.
  [thet]


2.2 (2011-08-24)
----------------

- migration to https://github.com/collective/icalendar using svn2git preserving
  tags, branches and authors.
  [garbas]

- using tox for testing on python 2.4, 2.5, 2.6, 2.6.
  [garbas]

- fixed tests so they pass also under python 2.7.
  [garbas]

- running tests on https://jenkins.plone.org/job/icalendar (only 2.6 for now)
  with some other metrics (pylint, clonedigger, coverage).
  [garbas]

- review and merge changes from https://github.com/cozi/icalendar fork.
  [garbas]

- created sphinx documentation and started documenting development and goals.
  [garbas]

- hook out github repository to http://readthedocs.org service so sphinx
  documentation is generated on each commit (for master). Documentation can be
  visible on: http://readthedocs.org/docs/icalendar/en/latest/
  [garbas]


2.1 (2009-12-14)
----------------

- Fix deprecation warnings about ``object.__init__`` taking no parameters.

- Set the VALUE parameter correctly for date values.

- Long binary data would be base64 encoded with newlines, which made the
  iCalendar files incorrect. (This still needs testing).

- Correctly handle content lines which include newlines.


2.0.1 (2008-07-11)
------------------

- Made the tests run under Python 2.5+

- Renamed the UTC class to Utc, so it would not clash with the UTC object,
  since that rendered the UTC object unpicklable.


2.0 (2008-07-11)
----------------

- EXDATE and RDATE now returns a vDDDLists object, which contains a list
  of vDDDTypes objects. This is do that EXDATE and RDATE can contain
  lists of dates, as per RFC.

  ***Note!***: This change is incompatible with earlier behavior, so if you
  handle EXDATE and RDATE you will need to update your code.

- When createing a vDuration of -5 hours (which in itself is nonsensical),
  the ical output of that was -P1DT19H, which is correct, but ugly. Now
  it's '-PT5H', which is prettier.


1.2 (2006-11-25)
----------------

- Fixed a string index out of range error in the new folding code.


1.1 (2006-11-23)
----------------

- Fixed a bug in caselessdicts popitem. (thanks to Michael Smith
  <msmith@fluendo.com>)

- The RFC 2445 was a bit unclear on how to handle line folding when it
  happened to be in the middle of a UTF-8 character. This has been clarified
  in the following discussion:
  http://lists.osafoundation.org/pipermail/ietf-calsify/2006-August/001126.html
  And this is now implemented in iCalendar. It will not fold in the middle of
  a UTF-8 character, but may fold in the middle of a UTF-8 composing character
  sequence.


1.0 (2006-08-03)
----------------

- make get_inline and set_inline support non ascii codes.

- Added support for creating a python egg distribution.


0.11 (2005-11-08)
-----------------

- Changed component .from_string to use types_factory instead of hardcoding
  entries to 'inline'

- Changed UTC tzinfo to a singleton so the same one is used everywhere

- Made the parser more strict by using regular expressions for key name,
  param name and quoted/unquoted safe char as per the RFC

- Added some tests from the schooltool icalendar parser for better coverage

- Be more forgiving on the regex for folding lines

- Allow for multiple top-level components on .from_string

- Fix vWeekdays, wasn't accepting relative param (eg: -3SA vs -SA)

- vDDDTypes didn't accept negative period (eg: -P30M)

- 'N' is also acceptable as newline on content lines, per RFC


0.10 (2005-04-28)
-----------------

- moved code to codespeak.net subversion.

- reorganized package structure so that source code is under 'src' directory.
  Non-package files remain in distribution root.

- redid doc/.py files as doc/.txt, using more modern doctest. Before they
  were .py files with big docstrings.

- added test.py testrunner, and tests/test_icalendar.py that picks up all
  doctests in source code and doc directory, and runs them, when typing::

    python2.3 test.py

- renamed iCalendar to lower case package name, lowercased, de-pluralized and
  shorted module names, which are mostly implementation detail.

- changed tests so they generate .ics files in a temp directory, not in the
  structure itself.
