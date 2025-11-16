// Function: ctor_714
// Address: 0x5bf2c0
//
int ctor_714()
{
  int v0; // edx

  qword_50516E0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_50516EC &= 0xF000u;
  qword_50516F0 = 0;
  qword_5051728 = (__int64)qword_4FA01C0;
  qword_50516F8 = 0;
  qword_5051700 = 0;
  qword_5051708 = 0;
  dword_50516E8 = v0;
  qword_5051738 = (__int64)&unk_5051758;
  qword_5051740 = (__int64)&unk_5051758;
  qword_5051710 = 0;
  qword_5051718 = 0;
  qword_5051788 = (__int64)&unk_49E74A8;
  qword_5051720 = 0;
  qword_5051730 = 0;
  qword_50516E0 = (__int64)&unk_49EEAF0;
  qword_5051748 = 4;
  dword_5051750 = 0;
  qword_5051798 = (__int64)&unk_49EEE10;
  byte_5051778 = 0;
  dword_5051780 = 0;
  byte_5051794 = 1;
  dword_5051790 = 0;
  sub_16B8280(&qword_50516E0, "max-cg-scc-iterations", 21);
  dword_5051780 = 4;
  byte_5051794 = 1;
  dword_5051790 = 4;
  LOBYTE(word_50516EC) = word_50516EC & 0x9F | 0x40;
  sub_16B88A0(&qword_50516E0);
  return __cxa_atexit(sub_12EDE60, &qword_50516E0, &qword_4A427C0);
}
