// Function: ctor_228
// Address: 0x4e9fe0
//
int ctor_228()
{
  int v0; // edx

  qword_4FB4E60 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB4E6C &= 0xF000u;
  qword_4FB4E70 = 0;
  qword_4FB4EA8 = (__int64)qword_4FA01C0;
  qword_4FB4E78 = 0;
  qword_4FB4E80 = 0;
  qword_4FB4E88 = 0;
  dword_4FB4E68 = v0;
  qword_4FB4EB8 = (__int64)&unk_4FB4ED8;
  qword_4FB4EC0 = (__int64)&unk_4FB4ED8;
  qword_4FB4E90 = 0;
  qword_4FB4E98 = 0;
  qword_4FB4F08 = (__int64)&unk_49E74A8;
  qword_4FB4EA0 = 0;
  qword_4FB4EB0 = 0;
  qword_4FB4E60 = (__int64)&unk_49EEAF0;
  qword_4FB4EC8 = 4;
  dword_4FB4ED0 = 0;
  qword_4FB4F18 = (__int64)&unk_49EEE10;
  byte_4FB4EF8 = 0;
  dword_4FB4F00 = 0;
  byte_4FB4F14 = 1;
  dword_4FB4F10 = 0;
  sub_16B8280(&qword_4FB4E60, "callsite-splitting-duplication-threshold", 40);
  qword_4FB4E90 = 82;
  dword_4FB4F00 = 5;
  byte_4FB4F14 = 1;
  dword_4FB4F10 = 5;
  LOBYTE(word_4FB4E6C) = word_4FB4E6C & 0x9F | 0x20;
  qword_4FB4E88 = (__int64)"Only allow instructions before a call, if their cost is below DuplicationThreshold";
  sub_16B88A0(&qword_4FB4E60);
  return __cxa_atexit(sub_12EDE60, &qword_4FB4E60, &qword_4A427C0);
}
