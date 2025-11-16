// Function: ctor_317
// Address: 0x5032c0
//
int ctor_317()
{
  int v0; // edx

  qword_4FC9080 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC908C &= 0xF000u;
  qword_4FC9090 = 0;
  qword_4FC90C8 = (__int64)qword_4FA01C0;
  qword_4FC9098 = 0;
  qword_4FC90A0 = 0;
  qword_4FC90A8 = 0;
  dword_4FC9088 = v0;
  qword_4FC90D8 = (__int64)&unk_4FC90F8;
  qword_4FC90E0 = (__int64)&unk_4FC90F8;
  qword_4FC90B0 = 0;
  qword_4FC90B8 = 0;
  qword_4FC9128 = (__int64)&unk_49E74A8;
  qword_4FC90C0 = 0;
  qword_4FC90D0 = 0;
  qword_4FC9080 = (__int64)&unk_49EEAF0;
  qword_4FC90E8 = 4;
  dword_4FC90F0 = 0;
  qword_4FC9138 = (__int64)&unk_49EEE10;
  byte_4FC9118 = 0;
  dword_4FC9120 = 0;
  byte_4FC9134 = 1;
  dword_4FC9130 = 0;
  sub_16B8280(&qword_4FC9080, "warn-stack-size", 15);
  dword_4FC9120 = -1;
  byte_4FC9134 = 1;
  dword_4FC9130 = -1;
  qword_4FC90B0 = 48;
  LOBYTE(word_4FC908C) = word_4FC908C & 0x9F | 0x20;
  qword_4FC90A8 = (__int64)"Warn for stack size bigger than the given number";
  sub_16B88A0(&qword_4FC9080);
  return __cxa_atexit(sub_12EDE60, &qword_4FC9080, &qword_4A427C0);
}
