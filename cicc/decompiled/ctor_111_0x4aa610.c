// Function: ctor_111
// Address: 0x4aa610
//
int ctor_111()
{
  int v0; // edx

  qword_4F96DC0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F96DCC &= 0xF000u;
  qword_4F96DD0 = 0;
  qword_4F96E08 = (__int64)&unk_4FA01C0;
  qword_4F96DD8 = 0;
  qword_4F96DE0 = 0;
  qword_4F96DE8 = 0;
  dword_4F96DC8 = v0;
  qword_4F96E18 = (__int64)&unk_4F96E38;
  qword_4F96E20 = (__int64)&unk_4F96E38;
  qword_4F96DF0 = 0;
  qword_4F96DF8 = 0;
  qword_4F96E68 = (__int64)&unk_49E74E8;
  word_4F96E70 = 256;
  qword_4F96E00 = 0;
  qword_4F96E10 = 0;
  qword_4F96DC0 = (__int64)&unk_49EEC70;
  qword_4F96E28 = 4;
  byte_4F96E58 = 0;
  qword_4F96E78 = (__int64)&unk_49EEDB0;
  dword_4F96E30 = 0;
  byte_4F96E60 = 0;
  sub_16B8280(&qword_4F96DC0, "disable-basicaa", 15);
  word_4F96E70 = 256;
  byte_4F96E60 = 0;
  LOBYTE(word_4F96DCC) = word_4F96DCC & 0x9F | 0x20;
  sub_16B88A0(&qword_4F96DC0);
  return __cxa_atexit(sub_12EDEC0, &qword_4F96DC0, &qword_4A427C0);
}
