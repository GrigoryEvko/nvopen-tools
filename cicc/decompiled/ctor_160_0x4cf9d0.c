// Function: ctor_160
// Address: 0x4cf9d0
//
int ctor_160()
{
  int v0; // edx

  qword_4FA0560 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA056C &= 0xF000u;
  qword_4FA0570 = 0;
  qword_4FA05A8 = (__int64)qword_4FA01C0;
  qword_4FA0578 = 0;
  qword_4FA0580 = 0;
  qword_4FA0588 = 0;
  dword_4FA0568 = v0;
  qword_4FA05B8 = (__int64)&unk_4FA05D8;
  qword_4FA05C0 = (__int64)&unk_4FA05D8;
  qword_4FA0590 = 0;
  qword_4FA0598 = 0;
  qword_4FA0608 = (__int64)&unk_49EF428;
  qword_4FA05A0 = 0;
  qword_4FA05B0 = 0;
  qword_4FA0560 = (__int64)&unk_49EF448;
  qword_4FA05C8 = 4;
  dword_4FA05D0 = 0;
  qword_4FA0620 = (__int64)&unk_49EEE30;
  byte_4FA05F8 = 0;
  qword_4FA0600 = 0;
  byte_4FA0618 = 1;
  qword_4FA0610 = 0;
  sub_16B8280(&qword_4FA0560, "rng-seed", 8);
  qword_4FA05A0 = 4;
  qword_4FA0598 = (__int64)"seed";
  qword_4FA0590 = 36;
  qword_4FA0600 = 0;
  byte_4FA0618 = 1;
  LOBYTE(word_4FA056C) = word_4FA056C & 0x9F | 0x20;
  qword_4FA0588 = (__int64)"Seed for the random number generator";
  qword_4FA0610 = 0;
  sub_16B88A0(&qword_4FA0560);
  return __cxa_atexit(sub_16C91A0, &qword_4FA0560, &qword_4A427C0);
}
