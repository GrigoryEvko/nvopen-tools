// Function: ctor_311
// Address: 0x5019d0
//
int ctor_311()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4FC8140 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC814C &= 0xF000u;
  qword_4FC8188 = (__int64)qword_4FA01C0;
  qword_4FC8150 = 0;
  qword_4FC8158 = 0;
  qword_4FC8160 = 0;
  dword_4FC8148 = v0;
  qword_4FC8198 = (__int64)&unk_4FC81B8;
  qword_4FC81A0 = (__int64)&unk_4FC81B8;
  qword_4FC8168 = 0;
  qword_4FC8170 = 0;
  word_4FC81F0 = 256;
  qword_4FC81E8 = (__int64)&unk_49E74E8;
  qword_4FC8140 = (__int64)&unk_49EEC70;
  qword_4FC81F8 = (__int64)&unk_49EEDB0;
  qword_4FC8178 = 0;
  qword_4FC8180 = 0;
  qword_4FC8190 = 0;
  qword_4FC81A8 = 4;
  dword_4FC81B0 = 0;
  byte_4FC81D8 = 0;
  byte_4FC81E0 = 0;
  sub_16B8280(&qword_4FC8140, "machine-sink-split", 18);
  qword_4FC8168 = (__int64)"Split critical edges during machine sinking";
  word_4FC81F0 = 257;
  byte_4FC81E0 = 1;
  qword_4FC8170 = 43;
  LOBYTE(word_4FC814C) = word_4FC814C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC8140);
  __cxa_atexit(sub_12EDEC0, &qword_4FC8140, &qword_4A427C0);
  qword_4FC8060 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC8110 = 256;
  qword_4FC8070 = 0;
  word_4FC806C &= 0xF000u;
  qword_4FC8108 = (__int64)&unk_49E74E8;
  qword_4FC8060 = (__int64)&unk_49EEC70;
  dword_4FC8068 = v1;
  qword_4FC8118 = (__int64)&unk_49EEDB0;
  qword_4FC80A8 = (__int64)qword_4FA01C0;
  qword_4FC80B8 = (__int64)&unk_4FC80D8;
  qword_4FC80C0 = (__int64)&unk_4FC80D8;
  qword_4FC8078 = 0;
  qword_4FC8080 = 0;
  qword_4FC8088 = 0;
  qword_4FC8090 = 0;
  qword_4FC8098 = 0;
  qword_4FC80A0 = 0;
  qword_4FC80B0 = 0;
  qword_4FC80C8 = 4;
  dword_4FC80D0 = 0;
  byte_4FC80F8 = 0;
  byte_4FC8100 = 0;
  sub_16B8280(&qword_4FC8060, "machine-sink-bfi", 16);
  qword_4FC8088 = (__int64)"Use block frequency info to find successors to sink";
  word_4FC8110 = 257;
  byte_4FC8100 = 1;
  qword_4FC8090 = 51;
  LOBYTE(word_4FC806C) = word_4FC806C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC8060);
  __cxa_atexit(sub_12EDEC0, &qword_4FC8060, &qword_4A427C0);
  qword_4FC7F80 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC7F8C &= 0xF000u;
  qword_4FC7F90 = 0;
  qword_4FC7F98 = 0;
  qword_4FC7FA0 = 0;
  qword_4FC7FA8 = 0;
  qword_4FC7FB0 = 0;
  dword_4FC7F88 = v2;
  qword_4FC7FD8 = (__int64)&unk_4FC7FF8;
  qword_4FC7FE0 = (__int64)&unk_4FC7FF8;
  qword_4FC7FC8 = (__int64)qword_4FA01C0;
  qword_4FC7FB8 = 0;
  qword_4FC8028 = (__int64)&unk_49E74A8;
  qword_4FC7FC0 = 0;
  qword_4FC7FD0 = 0;
  qword_4FC7F80 = (__int64)&unk_49EEAF0;
  qword_4FC7FE8 = 4;
  dword_4FC7FF0 = 0;
  qword_4FC8038 = (__int64)&unk_49EEE10;
  byte_4FC8018 = 0;
  dword_4FC8020 = 0;
  byte_4FC8034 = 1;
  dword_4FC8030 = 0;
  sub_16B8280(&qword_4FC7F80, "machine-sink-split-probability-threshold", 40);
  qword_4FC7FB0 = 222;
  qword_4FC7FA8 = (__int64)"Percentage threshold for splitting single-instruction critical edge. If the branch threshold "
                           "is higher than this threshold, we allow speculative execution of up to 1 instruction to avoid"
                           " branching to splitted critical edge";
  dword_4FC8020 = 40;
  byte_4FC8034 = 1;
  dword_4FC8030 = 40;
  LOBYTE(word_4FC7F8C) = word_4FC7F8C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC7F80);
  return __cxa_atexit(sub_12EDE60, &qword_4FC7F80, &qword_4A427C0);
}
