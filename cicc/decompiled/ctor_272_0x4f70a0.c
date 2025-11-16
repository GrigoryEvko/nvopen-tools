// Function: ctor_272
// Address: 0x4f70a0
//
int ctor_272()
{
  int v0; // edx

  qword_4FBED60 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBED6C &= 0xF000u;
  qword_4FBED70 = 0;
  qword_4FBEDA8 = (__int64)qword_4FA01C0;
  qword_4FBED78 = 0;
  qword_4FBED80 = 0;
  qword_4FBED88 = 0;
  dword_4FBED68 = v0;
  qword_4FBEDB8 = (__int64)&unk_4FBEDD8;
  qword_4FBEDC0 = (__int64)&unk_4FBEDD8;
  qword_4FBED90 = 0;
  qword_4FBED98 = 0;
  qword_4FBEE08 = (__int64)&unk_49E74C8;
  qword_4FBEDA0 = 0;
  qword_4FBEDB0 = 0;
  qword_4FBED60 = (__int64)&unk_49EEB70;
  qword_4FBEDC8 = 4;
  dword_4FBEDD0 = 0;
  qword_4FBEE18 = (__int64)&unk_49EEDF0;
  byte_4FBEDF8 = 0;
  dword_4FBEE00 = 0;
  byte_4FBEE14 = 1;
  dword_4FBEE10 = 0;
  sub_16B8280(&qword_4FBED60, "reuse-lmem-very-long-live-range", 31);
  dword_4FBEE00 = 5000;
  byte_4FBEE14 = 1;
  dword_4FBEE10 = 5000;
  qword_4FBED90 = 45;
  LOBYTE(word_4FBED6C) = word_4FBED6C & 0x9F | 0x20;
  qword_4FBED88 = (__int64)"Define the threshold for very long live range";
  sub_16B88A0(&qword_4FBED60);
  return __cxa_atexit(sub_12EDEA0, &qword_4FBED60, &qword_4A427C0);
}
