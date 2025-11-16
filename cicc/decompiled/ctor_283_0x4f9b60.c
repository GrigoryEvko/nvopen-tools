// Function: ctor_283
// Address: 0x4f9b60
//
int ctor_283()
{
  int v0; // edx

  qword_4FC14E0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC14EC &= 0xF000u;
  qword_4FC14F0 = 0;
  qword_4FC1528 = (__int64)qword_4FA01C0;
  qword_4FC14F8 = 0;
  qword_4FC1500 = 0;
  qword_4FC1508 = 0;
  dword_4FC14E8 = v0;
  qword_4FC1538 = (__int64)&unk_4FC1558;
  qword_4FC1540 = (__int64)&unk_4FC1558;
  qword_4FC1510 = 0;
  qword_4FC1518 = 0;
  qword_4FC1588 = (__int64)&unk_49E74C8;
  qword_4FC1520 = 0;
  qword_4FC1530 = 0;
  qword_4FC14E0 = (__int64)&unk_49EEB70;
  qword_4FC1548 = 4;
  dword_4FC1550 = 0;
  qword_4FC1598 = (__int64)&unk_49EEDF0;
  byte_4FC1578 = 0;
  dword_4FC1580 = 0;
  byte_4FC1594 = 1;
  dword_4FC1590 = 0;
  sub_16B8280(&qword_4FC14E0, "sched-high-latency-cycles", 25);
  dword_4FC1580 = 10;
  byte_4FC1594 = 1;
  dword_4FC1590 = 10;
  qword_4FC1510 = 104;
  LOBYTE(word_4FC14EC) = word_4FC14EC & 0x9F | 0x20;
  qword_4FC1508 = (__int64)"Roughly estimate the number of cycles that 'long latency'instructions take for targets with no itinerary";
  sub_16B88A0(&qword_4FC14E0);
  return __cxa_atexit(sub_12EDEA0, &qword_4FC14E0, &qword_4A427C0);
}
