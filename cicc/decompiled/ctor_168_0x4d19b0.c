// Function: ctor_168
// Address: 0x4d19b0
//
int ctor_168()
{
  int v0; // edx

  qword_4FA2800 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA280C &= 0xF000u;
  qword_4FA2810 = 0;
  qword_4FA2848 = (__int64)qword_4FA01C0;
  qword_4FA2818 = 0;
  qword_4FA2820 = 0;
  qword_4FA2828 = 0;
  dword_4FA2808 = v0;
  qword_4FA2858 = (__int64)&unk_4FA2878;
  qword_4FA2860 = (__int64)&unk_4FA2878;
  qword_4FA2830 = 0;
  qword_4FA2838 = 0;
  qword_4FA28A8 = (__int64)&unk_49E74A8;
  qword_4FA2840 = 0;
  qword_4FA2850 = 0;
  qword_4FA2800 = (__int64)&unk_49EEAF0;
  qword_4FA2868 = 4;
  dword_4FA2870 = 0;
  qword_4FA28B8 = (__int64)&unk_49EEE10;
  byte_4FA2898 = 0;
  dword_4FA28A0 = 0;
  byte_4FA28B4 = 1;
  dword_4FA28B0 = 0;
  sub_16B8280(&qword_4FA2800, "instcombine-max-num-phis", 24);
  dword_4FA28A0 = 512;
  byte_4FA28B4 = 1;
  dword_4FA28B0 = 512;
  qword_4FA2828 = (__int64)"Maximum number phis to handle in intptr/ptrint folding";
  qword_4FA2830 = 54;
  sub_16B88A0(&qword_4FA2800);
  return __cxa_atexit(sub_12EDE60, &qword_4FA2800, &qword_4A427C0);
}
