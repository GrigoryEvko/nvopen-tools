// Function: ctor_325
// Address: 0x505020
//
int ctor_325()
{
  int v0; // edx

  qword_4FCA300 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCA30C &= 0xF000u;
  qword_4FCA310 = 0;
  qword_4FCA348 = (__int64)qword_4FA01C0;
  qword_4FCA318 = 0;
  qword_4FCA320 = 0;
  qword_4FCA328 = 0;
  dword_4FCA308 = v0;
  qword_4FCA358 = (__int64)&unk_4FCA378;
  qword_4FCA360 = (__int64)&unk_4FCA378;
  qword_4FCA330 = 0;
  qword_4FCA338 = 0;
  qword_4FCA3A8 = (__int64)&unk_49E74E8;
  word_4FCA3B0 = 256;
  qword_4FCA340 = 0;
  qword_4FCA350 = 0;
  qword_4FCA300 = (__int64)&unk_49EEC70;
  qword_4FCA368 = 4;
  byte_4FCA398 = 0;
  qword_4FCA3B8 = (__int64)&unk_49EEDB0;
  dword_4FCA370 = 0;
  byte_4FCA3A0 = 0;
  sub_16B8280(&qword_4FCA300, "safe-stack-layout", 17);
  qword_4FCA328 = (__int64)"enable safe stack layout";
  word_4FCA3B0 = 257;
  byte_4FCA3A0 = 1;
  qword_4FCA330 = 24;
  LOBYTE(word_4FCA30C) = word_4FCA30C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FCA300);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCA300, &qword_4A427C0);
}
