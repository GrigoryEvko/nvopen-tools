// Function: ctor_213
// Address: 0x4e47d0
//
int ctor_213()
{
  int v0; // eax
  int v1; // eax

  qword_4FB1000 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB100C &= 0xF000u;
  qword_4FB1010 = 0;
  qword_4FB1018 = 0;
  qword_4FB1020 = 0;
  qword_4FB1028 = 0;
  qword_4FB1030 = 0;
  dword_4FB1008 = v0;
  qword_4FB1038 = 0;
  qword_4FB1048 = (__int64)qword_4FA01C0;
  qword_4FB1058 = (__int64)&unk_4FB1078;
  qword_4FB1060 = (__int64)&unk_4FB1078;
  qword_4FB1040 = 0;
  qword_4FB1050 = 0;
  qword_4FB10A8 = (__int64)&unk_49E74A8;
  qword_4FB1068 = 4;
  qword_4FB1000 = (__int64)&unk_49EEAF0;
  dword_4FB1070 = 0;
  qword_4FB10B8 = (__int64)&unk_49EEE10;
  byte_4FB1098 = 0;
  dword_4FB10A0 = 0;
  byte_4FB10B4 = 1;
  dword_4FB10B0 = 0;
  sub_16B8280(&qword_4FB1000, "sink-freq-percent-threshold", 27);
  dword_4FB10A0 = 90;
  byte_4FB10B4 = 1;
  dword_4FB10B0 = 90;
  qword_4FB1030 = 101;
  LOBYTE(word_4FB100C) = word_4FB100C & 0x9F | 0x20;
  qword_4FB1028 = (__int64)"Do not sink instructions that require cloning unless they execute less than this percent of the time.";
  sub_16B88A0(&qword_4FB1000);
  __cxa_atexit(sub_12EDE60, &qword_4FB1000, &qword_4A427C0);
  qword_4FB0F20 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB0F2C &= 0xF000u;
  qword_4FB0F30 = 0;
  qword_4FB0F38 = 0;
  qword_4FB0F40 = 0;
  qword_4FB0F48 = 0;
  qword_4FB0F50 = 0;
  dword_4FB0F28 = v1;
  qword_4FB0FC8 = (__int64)&unk_49E74A8;
  qword_4FB0F68 = (__int64)qword_4FA01C0;
  qword_4FB0F78 = (__int64)&unk_4FB0F98;
  qword_4FB0F80 = (__int64)&unk_4FB0F98;
  qword_4FB0F20 = (__int64)&unk_49EEAF0;
  qword_4FB0FD8 = (__int64)&unk_49EEE10;
  qword_4FB0F58 = 0;
  qword_4FB0F60 = 0;
  qword_4FB0F70 = 0;
  qword_4FB0F88 = 4;
  dword_4FB0F90 = 0;
  byte_4FB0FB8 = 0;
  dword_4FB0FC0 = 0;
  byte_4FB0FD4 = 1;
  dword_4FB0FD0 = 0;
  sub_16B8280(&qword_4FB0F20, "max-uses-for-sinking", 20);
  dword_4FB0FC0 = 30;
  byte_4FB0FD4 = 1;
  dword_4FB0FD0 = 30;
  qword_4FB0F50 = 49;
  LOBYTE(word_4FB0F2C) = word_4FB0F2C & 0x9F | 0x20;
  qword_4FB0F48 = (__int64)"Do not sink instructions that have too many uses.";
  sub_16B88A0(&qword_4FB0F20);
  return __cxa_atexit(sub_12EDE60, &qword_4FB0F20, &qword_4A427C0);
}
