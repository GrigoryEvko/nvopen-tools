// Function: ctor_132
// Address: 0x4b0000
//
int ctor_132()
{
  int v0; // edx

  qword_4F9B620 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9B62C &= 0xF000u;
  qword_4F9B630 = 0;
  qword_4F9B668 = (__int64)&unk_4FA01C0;
  qword_4F9B638 = 0;
  qword_4F9B640 = 0;
  qword_4F9B648 = 0;
  dword_4F9B628 = v0;
  qword_4F9B678 = (__int64)&unk_4F9B698;
  qword_4F9B680 = (__int64)&unk_4F9B698;
  qword_4F9B650 = 0;
  qword_4F9B658 = 0;
  qword_4F9B6C8 = (__int64)&unk_49E74E8;
  word_4F9B6D0 = 256;
  qword_4F9B660 = 0;
  qword_4F9B670 = 0;
  qword_4F9B620 = (__int64)&unk_49EEC70;
  qword_4F9B688 = 4;
  byte_4F9B6B8 = 0;
  qword_4F9B6D8 = (__int64)&unk_49EEDB0;
  dword_4F9B690 = 0;
  byte_4F9B6C0 = 0;
  sub_16B8280(&qword_4F9B620, "enable-scoped-noalias", 21);
  word_4F9B6D0 = 257;
  byte_4F9B6C0 = 1;
  LOBYTE(word_4F9B62C) = word_4F9B62C & 0x9F | 0x20;
  sub_16B88A0(&qword_4F9B620);
  return __cxa_atexit(sub_12EDEC0, &qword_4F9B620, &qword_4A427C0);
}
