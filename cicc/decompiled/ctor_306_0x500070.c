// Function: ctor_306
// Address: 0x500070
//
int ctor_306()
{
  int v0; // edx

  qword_4FC6A20 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC6A2C &= 0xF000u;
  qword_4FC6A30 = 0;
  qword_4FC6A68 = (__int64)qword_4FA01C0;
  qword_4FC6A38 = 0;
  qword_4FC6A40 = 0;
  qword_4FC6A48 = 0;
  dword_4FC6A28 = v0;
  qword_4FC6A78 = (__int64)&unk_4FC6A98;
  qword_4FC6A80 = (__int64)&unk_4FC6A98;
  qword_4FC6A50 = 0;
  qword_4FC6A58 = 0;
  qword_4FC6AC8 = (__int64)&unk_49E74C8;
  qword_4FC6A60 = 0;
  qword_4FC6A70 = 0;
  qword_4FC6A20 = (__int64)&unk_49EEB70;
  qword_4FC6A88 = 4;
  dword_4FC6A90 = 0;
  qword_4FC6AD8 = (__int64)&unk_49EEDF0;
  byte_4FC6AB8 = 0;
  dword_4FC6AC0 = 0;
  byte_4FC6AD4 = 1;
  dword_4FC6AD0 = 0;
  sub_16B8280(&qword_4FC6A20, "print-regmask-num-regs", 22);
  qword_4FC6A50 = 90;
  qword_4FC6A48 = (__int64)"Number of registers to limit to when printing regmask operands in IR dumps. unlimited = -1";
  dword_4FC6AC0 = 32;
  byte_4FC6AD4 = 1;
  dword_4FC6AD0 = 32;
  LOBYTE(word_4FC6A2C) = word_4FC6A2C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC6A20);
  return __cxa_atexit(sub_12EDEA0, &qword_4FC6A20, &qword_4A427C0);
}
