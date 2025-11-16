// Function: ctor_121
// Address: 0x4acd40
//
int ctor_121()
{
  int v0; // eax
  int v1; // eax

  qword_4F99040 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9904C &= 0xF000u;
  qword_4F99050 = 0;
  qword_4F99058 = 0;
  qword_4F99060 = 0;
  qword_4F99068 = 0;
  qword_4F99070 = 0;
  dword_4F99048 = v0;
  qword_4F99078 = 0;
  qword_4F99088 = (__int64)&unk_4FA01C0;
  qword_4F99098 = (__int64)&unk_4F990B8;
  qword_4F990A0 = (__int64)&unk_4F990B8;
  qword_4F99080 = 0;
  qword_4F99090 = 0;
  word_4F990F0 = 256;
  qword_4F990E8 = (__int64)&unk_49E74E8;
  qword_4F990A8 = 4;
  qword_4F99040 = (__int64)&unk_49EEC70;
  byte_4F990D8 = 0;
  qword_4F990F8 = (__int64)&unk_49EEDB0;
  dword_4F990B0 = 0;
  byte_4F990E0 = 0;
  sub_16B8280(&qword_4F99040, "check-sxtopt", 12);
  qword_4F99068 = (__int64)"Check if sign extension can be eliminated";
  word_4F990F0 = 257;
  byte_4F990E0 = 1;
  qword_4F99070 = 41;
  LOBYTE(word_4F9904C) = word_4F9904C & 0x9F | 0x20;
  sub_16B88A0(&qword_4F99040);
  __cxa_atexit(sub_12EDEC0, &qword_4F99040, &qword_4A427C0);
  qword_4F98F60 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F99010 = 256;
  word_4F98F6C &= 0xF000u;
  qword_4F98F70 = 0;
  qword_4F98F78 = 0;
  qword_4F98F80 = 0;
  dword_4F98F68 = v1;
  qword_4F99008 = (__int64)&unk_49E74E8;
  qword_4F98FA8 = (__int64)&unk_4FA01C0;
  qword_4F98FB8 = (__int64)&unk_4F98FD8;
  qword_4F98FC0 = (__int64)&unk_4F98FD8;
  qword_4F98F60 = (__int64)&unk_49EEC70;
  qword_4F99018 = (__int64)&unk_49EEDB0;
  qword_4F98F88 = 0;
  qword_4F98F90 = 0;
  qword_4F98F98 = 0;
  qword_4F98FA0 = 0;
  qword_4F98FB0 = 0;
  qword_4F98FC8 = 4;
  dword_4F98FD0 = 0;
  byte_4F98FF8 = 0;
  byte_4F99000 = 0;
  sub_16B8280(&qword_4F98F60, "iv-skip-sxt", 11);
  qword_4F98F88 = (__int64)"Ignore SignExtendedExpr for IV";
  word_4F99010 = 256;
  byte_4F99000 = 0;
  qword_4F98F90 = 30;
  LOBYTE(word_4F98F6C) = word_4F98F6C & 0x9F | 0x20;
  sub_16B88A0(&qword_4F98F60);
  return __cxa_atexit(sub_12EDEC0, &qword_4F98F60, &qword_4A427C0);
}
