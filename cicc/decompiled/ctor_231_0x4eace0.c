// Function: ctor_231
// Address: 0x4eace0
//
int ctor_231()
{
  int v0; // edx

  qword_4FB5720 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB572C &= 0xF000u;
  qword_4FB5730 = 0;
  qword_4FB5768 = (__int64)qword_4FA01C0;
  qword_4FB5738 = 0;
  qword_4FB5740 = 0;
  qword_4FB5748 = 0;
  dword_4FB5728 = v0;
  qword_4FB5778 = (__int64)&unk_4FB5798;
  qword_4FB5780 = (__int64)&unk_4FB5798;
  qword_4FB5750 = 0;
  qword_4FB5758 = 0;
  qword_4FB57C8 = (__int64)&unk_49E74A8;
  qword_4FB5760 = 0;
  qword_4FB5770 = 0;
  qword_4FB5720 = (__int64)&unk_49EEAF0;
  qword_4FB5788 = 4;
  dword_4FB5790 = 0;
  qword_4FB57D8 = (__int64)&unk_49EEE10;
  byte_4FB57B8 = 0;
  dword_4FB57C0 = 0;
  byte_4FB57D4 = 1;
  dword_4FB57D0 = 0;
  sub_16B8280(&qword_4FB5720, "guards-predicate-pass-branch-weight", 35);
  dword_4FB57C0 = 0x100000;
  byte_4FB57D4 = 1;
  dword_4FB57D0 = 0x100000;
  qword_4FB5750 = 100;
  LOBYTE(word_4FB572C) = word_4FB572C & 0x9F | 0x20;
  qword_4FB5748 = (__int64)"The probability of a guard failing is assumed to be the reciprocal of this value (default = 1 << 20)";
  sub_16B88A0(&qword_4FB5720);
  return __cxa_atexit(sub_12EDE60, &qword_4FB5720, &qword_4A427C0);
}
