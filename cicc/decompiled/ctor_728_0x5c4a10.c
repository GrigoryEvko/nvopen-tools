// Function: ctor_728
// Address: 0x5c4a10
//
int ctor_728()
{
  int v0; // edx

  qword_50560A0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_50560AC &= 0xF000u;
  qword_50560B0 = 0;
  qword_50560E8 = (__int64)qword_4FA01C0;
  qword_50560B8 = 0;
  qword_50560C0 = 0;
  qword_50560C8 = 0;
  dword_50560A8 = v0;
  qword_50560F8 = (__int64)&unk_5056118;
  qword_5056100 = (__int64)&unk_5056118;
  qword_50560D0 = 0;
  qword_50560D8 = 0;
  qword_5056148 = (__int64)&unk_49E74E8;
  word_5056150 = 256;
  qword_50560E0 = 0;
  qword_50560F0 = 0;
  qword_50560A0 = (__int64)&unk_49EEC70;
  qword_5056108 = 4;
  byte_5056138 = 0;
  qword_5056158 = (__int64)&unk_49EEDB0;
  dword_5056110 = 0;
  byte_5056140 = 0;
  sub_16B8280(&qword_50560A0, "print-schedule", 14);
  word_5056150 = 256;
  byte_5056140 = 0;
  qword_50560D0 = 48;
  LOBYTE(word_50560AC) = word_50560AC & 0x9F | 0x20;
  qword_50560C8 = (__int64)"Print 'sched: [latency:throughput]' in .s output";
  sub_16B88A0(&qword_50560A0);
  return __cxa_atexit(sub_12EDEC0, &qword_50560A0, &qword_4A427C0);
}
