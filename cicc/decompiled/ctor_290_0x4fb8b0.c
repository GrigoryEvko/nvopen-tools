// Function: ctor_290
// Address: 0x4fb8b0
//
int ctor_290()
{
  int v0; // edx

  qword_4FC3520 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC352C &= 0xF000u;
  qword_4FC3530 = 0;
  qword_4FC3568 = (__int64)qword_4FA01C0;
  qword_4FC3538 = 0;
  qword_4FC3540 = 0;
  qword_4FC3548 = 0;
  dword_4FC3528 = v0;
  qword_4FC3578 = (__int64)&unk_4FC3598;
  qword_4FC3580 = (__int64)&unk_4FC3598;
  qword_4FC3550 = 0;
  qword_4FC3558 = 0;
  qword_4FC35C8 = (__int64)&unk_49E74A8;
  qword_4FC3560 = 0;
  qword_4FC3570 = 0;
  qword_4FC3520 = (__int64)&unk_49EEAF0;
  qword_4FC3588 = 4;
  dword_4FC3590 = 0;
  qword_4FC35D8 = (__int64)&unk_49EEE10;
  byte_4FC35B8 = 0;
  dword_4FC35C0 = 0;
  byte_4FC35D4 = 1;
  dword_4FC35D0 = 0;
  sub_16B8280(&qword_4FC3520, "memcmp-num-loads-per-block", 26);
  dword_4FC35C0 = 1;
  byte_4FC35D4 = 1;
  dword_4FC35D0 = 1;
  qword_4FC3550 = 108;
  LOBYTE(word_4FC352C) = word_4FC352C & 0x9F | 0x20;
  qword_4FC3548 = (__int64)"The number of loads per basic block for inline expansion of memcmp that is only being compared against zero.";
  sub_16B88A0(&qword_4FC3520);
  return __cxa_atexit(sub_12EDE60, &qword_4FC3520, &qword_4A427C0);
}
