// Function: ctor_354
// Address: 0x50c540
//
int ctor_354()
{
  int v0; // edx

  qword_4FCFB40 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCFB4C &= 0xF000u;
  qword_4FCFB50 = 0;
  qword_4FCFB88 = (__int64)qword_4FA01C0;
  qword_4FCFB58 = 0;
  qword_4FCFB60 = 0;
  qword_4FCFB68 = 0;
  dword_4FCFB48 = v0;
  qword_4FCFB98 = (__int64)&unk_4FCFBB8;
  qword_4FCFBA0 = (__int64)&unk_4FCFBB8;
  qword_4FCFB70 = 0;
  qword_4FCFB78 = 0;
  qword_4FCFBE8 = (__int64)&unk_49E74E8;
  word_4FCFBF0 = 256;
  qword_4FCFB80 = 0;
  qword_4FCFB90 = 0;
  qword_4FCFB40 = (__int64)&unk_49EEC70;
  qword_4FCFBA8 = 4;
  byte_4FCFBD8 = 0;
  qword_4FCFBF8 = (__int64)&unk_49EEDB0;
  dword_4FCFBB0 = 0;
  byte_4FCFBE0 = 0;
  sub_16B8280(&qword_4FCFB40, "safe-stack-coloring", 19);
  qword_4FCFB68 = (__int64)"enable safe stack coloring";
  word_4FCFBF0 = 256;
  byte_4FCFBE0 = 0;
  qword_4FCFB70 = 26;
  LOBYTE(word_4FCFB4C) = word_4FCFB4C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FCFB40);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCFB40, &qword_4A427C0);
}
