// Function: ctor_304
// Address: 0x4ff800
//
int ctor_304()
{
  int v0; // edx

  qword_4FC63E0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC63EC &= 0xF000u;
  qword_4FC63F0 = 0;
  qword_4FC6428 = (__int64)qword_4FA01C0;
  qword_4FC63F8 = 0;
  qword_4FC6400 = 0;
  qword_4FC6408 = 0;
  dword_4FC63E8 = v0;
  qword_4FC6438 = (__int64)&unk_4FC6458;
  qword_4FC6440 = (__int64)&unk_4FC6458;
  qword_4FC6410 = 0;
  qword_4FC6418 = 0;
  qword_4FC6488 = (__int64)&unk_49E74A8;
  qword_4FC6420 = 0;
  qword_4FC6430 = 0;
  qword_4FC63E0 = (__int64)&unk_49EEAF0;
  qword_4FC6448 = 4;
  dword_4FC6450 = 0;
  qword_4FC6498 = (__int64)&unk_49EEE10;
  byte_4FC6478 = 0;
  dword_4FC6480 = 0;
  byte_4FC6494 = 1;
  dword_4FC6490 = 0;
  sub_16B8280(&qword_4FC63E0, "align-all-functions", 19);
  qword_4FC6410 = 37;
  qword_4FC6408 = (__int64)"Force the alignment of all functions.";
  dword_4FC6480 = 0;
  byte_4FC6494 = 1;
  dword_4FC6490 = 0;
  LOBYTE(word_4FC63EC) = word_4FC63EC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC63E0);
  return __cxa_atexit(sub_12EDE60, &qword_4FC63E0, &qword_4A427C0);
}
