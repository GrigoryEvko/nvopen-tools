// Function: ctor_182
// Address: 0x4da480
//
int ctor_182()
{
  int v0; // edx

  qword_4FAA4C0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAA4CC &= 0xF000u;
  qword_4FAA4D0 = 0;
  qword_4FAA508 = (__int64)qword_4FA01C0;
  qword_4FAA4D8 = 0;
  qword_4FAA4E0 = 0;
  qword_4FAA4E8 = 0;
  dword_4FAA4C8 = v0;
  qword_4FAA518 = (__int64)&unk_4FAA538;
  qword_4FAA520 = (__int64)&unk_4FAA538;
  qword_4FAA4F0 = 0;
  qword_4FAA4F8 = 0;
  qword_4FAA568 = (__int64)&unk_49E74A8;
  qword_4FAA500 = 0;
  qword_4FAA510 = 0;
  qword_4FAA4C0 = (__int64)&unk_49EEAF0;
  qword_4FAA528 = 4;
  dword_4FAA530 = 0;
  qword_4FAA578 = (__int64)&unk_49EEE10;
  byte_4FAA558 = 0;
  dword_4FAA560 = 0;
  byte_4FAA574 = 1;
  dword_4FAA570 = 0;
  sub_16B8280(&qword_4FAA4C0, "cvp-max-functions-per-value", 27);
  dword_4FAA560 = 4;
  byte_4FAA574 = 1;
  dword_4FAA570 = 4;
  qword_4FAA4F0 = 58;
  LOBYTE(word_4FAA4CC) = word_4FAA4CC & 0x9F | 0x20;
  qword_4FAA4E8 = (__int64)"The maximum number of functions to track per lattice value";
  sub_16B88A0(&qword_4FAA4C0);
  return __cxa_atexit(sub_12EDE60, &qword_4FAA4C0, &qword_4A427C0);
}
