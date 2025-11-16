// Function: ctor_169
// Address: 0x4d1b50
//
int ctor_169()
{
  int v0; // edx

  qword_4FA28E0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA28EC &= 0xF000u;
  qword_4FA28F0 = 0;
  qword_4FA2928 = (__int64)qword_4FA01C0;
  qword_4FA28F8 = 0;
  qword_4FA2900 = 0;
  qword_4FA2908 = 0;
  dword_4FA28E8 = v0;
  qword_4FA2938 = (__int64)&unk_4FA2958;
  qword_4FA2940 = (__int64)&unk_4FA2958;
  qword_4FA2910 = 0;
  qword_4FA2918 = 0;
  qword_4FA2988 = (__int64)&unk_49E74E8;
  word_4FA2990 = 256;
  qword_4FA2920 = 0;
  qword_4FA2930 = 0;
  qword_4FA28E0 = (__int64)&unk_49EEC70;
  qword_4FA2948 = 4;
  byte_4FA2978 = 0;
  qword_4FA2998 = (__int64)&unk_49EEDB0;
  dword_4FA2950 = 0;
  byte_4FA2980 = 0;
  sub_16B8280(&qword_4FA28E0, "bounds-checking-single-trap", 27);
  qword_4FA2910 = 31;
  qword_4FA2908 = (__int64)"Use one trap block per function";
  sub_16B88A0(&qword_4FA28E0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FA28E0, &qword_4A427C0);
}
