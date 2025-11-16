// Function: ctor_312
// Address: 0x501e60
//
int ctor_312()
{
  int v0; // eax
  int v1; // eax

  qword_4FC8320 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC832C &= 0xF000u;
  qword_4FC8330 = 0;
  qword_4FC8338 = 0;
  qword_4FC8340 = 0;
  qword_4FC8348 = 0;
  qword_4FC8350 = 0;
  dword_4FC8328 = v0;
  qword_4FC8358 = 0;
  qword_4FC8368 = (__int64)qword_4FA01C0;
  qword_4FC8378 = (__int64)&unk_4FC8398;
  qword_4FC8380 = (__int64)&unk_4FC8398;
  qword_4FC8360 = 0;
  qword_4FC8370 = 0;
  qword_4FC83C8 = (__int64)&unk_49E74A8;
  qword_4FC8320 = (__int64)&unk_49EEAF0;
  byte_4FC83B8 = 0;
  qword_4FC83D8 = (__int64)&unk_49EEE10;
  qword_4FC8388 = 4;
  dword_4FC8390 = 0;
  dword_4FC83C0 = 0;
  byte_4FC83D4 = 1;
  dword_4FC83D0 = 0;
  sub_16B8280(&qword_4FC8320, "canon-nth-function", 18);
  qword_4FC8358 = (__int64)"N";
  dword_4FC83C0 = -1;
  byte_4FC83D4 = 1;
  dword_4FC83D0 = -1;
  LOBYTE(word_4FC832C) = word_4FC832C & 0x9F | 0x20;
  qword_4FC8348 = (__int64)"Function number to canonicalize.";
  qword_4FC8360 = 1;
  qword_4FC8350 = 32;
  sub_16B88A0(&qword_4FC8320);
  __cxa_atexit(sub_12EDE60, &qword_4FC8320, &qword_4A427C0);
  qword_4FC8240 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC824C &= 0xF000u;
  qword_4FC8250 = 0;
  qword_4FC8258 = 0;
  qword_4FC8260 = 0;
  qword_4FC82E8 = (__int64)&unk_49E74A8;
  qword_4FC8268 = 0;
  dword_4FC8248 = v1;
  qword_4FC8240 = (__int64)&unk_49EEAF0;
  qword_4FC8288 = (__int64)qword_4FA01C0;
  qword_4FC8298 = (__int64)&unk_4FC82B8;
  qword_4FC82A0 = (__int64)&unk_4FC82B8;
  qword_4FC82F8 = (__int64)&unk_49EEE10;
  qword_4FC8270 = 0;
  qword_4FC8278 = 0;
  qword_4FC8280 = 0;
  qword_4FC8290 = 0;
  qword_4FC82A8 = 4;
  dword_4FC82B0 = 0;
  byte_4FC82D8 = 0;
  dword_4FC82E0 = 0;
  byte_4FC82F4 = 1;
  dword_4FC82F0 = 0;
  sub_16B8280(&qword_4FC8240, "canon-nth-basicblock", 20);
  qword_4FC8278 = (__int64)"N";
  dword_4FC82E0 = -1;
  byte_4FC82F4 = 1;
  dword_4FC82F0 = -1;
  LOBYTE(word_4FC824C) = word_4FC824C & 0x9F | 0x20;
  qword_4FC8280 = 1;
  qword_4FC8268 = (__int64)"BasicBlock number to canonicalize.";
  qword_4FC8270 = 34;
  sub_16B88A0(&qword_4FC8240);
  return __cxa_atexit(sub_12EDE60, &qword_4FC8240, &qword_4A427C0);
}
