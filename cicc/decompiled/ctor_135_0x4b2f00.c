// Function: ctor_135
// Address: 0x4b2f00
//
int ctor_135()
{
  int v0; // edx

  qword_4F9D4C0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9D4CC &= 0xF000u;
  qword_4F9D4D0 = 0;
  qword_4F9D508 = (__int64)&unk_4FA01C0;
  qword_4F9D4D8 = 0;
  qword_4F9D4E0 = 0;
  qword_4F9D4E8 = 0;
  dword_4F9D4C8 = v0;
  qword_4F9D518 = (__int64)&unk_4F9D538;
  qword_4F9D520 = (__int64)&unk_4F9D538;
  qword_4F9D4F0 = 0;
  qword_4F9D4F8 = 0;
  qword_4F9D568 = (__int64)&unk_49E74E8;
  word_4F9D570 = 256;
  qword_4F9D500 = 0;
  qword_4F9D510 = 0;
  qword_4F9D4C0 = (__int64)&unk_49EEC70;
  qword_4F9D528 = 4;
  byte_4F9D558 = 0;
  qword_4F9D578 = (__int64)&unk_49EEDB0;
  dword_4F9D530 = 0;
  byte_4F9D560 = 0;
  sub_16B8280(&qword_4F9D4C0, "enable-tbaa", 11);
  word_4F9D570 = 257;
  byte_4F9D560 = 1;
  LOBYTE(word_4F9D4CC) = word_4F9D4CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4F9D4C0);
  return __cxa_atexit(sub_12EDEC0, &qword_4F9D4C0, &qword_4A427C0);
}
