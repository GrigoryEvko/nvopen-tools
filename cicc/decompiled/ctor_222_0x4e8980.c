// Function: ctor_222
// Address: 0x4e8980
//
int ctor_222()
{
  int v0; // eax
  int v1; // eax

  qword_4FB4140 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB414C &= 0xF000u;
  qword_4FB4150 = 0;
  qword_4FB4158 = 0;
  qword_4FB4160 = 0;
  qword_4FB4168 = 0;
  qword_4FB4170 = 0;
  dword_4FB4148 = v0;
  qword_4FB4178 = 0;
  qword_4FB4188 = (__int64)qword_4FA01C0;
  qword_4FB4198 = (__int64)&unk_4FB41B8;
  qword_4FB41A0 = (__int64)&unk_4FB41B8;
  qword_4FB4180 = 0;
  qword_4FB4190 = 0;
  word_4FB41F0 = 256;
  qword_4FB41E8 = (__int64)&unk_49E74E8;
  qword_4FB41A8 = 4;
  qword_4FB4140 = (__int64)&unk_49EEC70;
  byte_4FB41D8 = 0;
  qword_4FB41F8 = (__int64)&unk_49EEDB0;
  dword_4FB41B0 = 0;
  byte_4FB41E0 = 0;
  sub_16B8280(&qword_4FB4140, "disable-separate-const-offset-from-gep", 38);
  qword_4FB4168 = (__int64)"Do not separate the constant offset from a GEP instruction";
  word_4FB41F0 = 256;
  byte_4FB41E0 = 0;
  qword_4FB4170 = 58;
  LOBYTE(word_4FB414C) = word_4FB414C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB4140);
  __cxa_atexit(sub_12EDEC0, &qword_4FB4140, &qword_4A427C0);
  qword_4FB4060 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB4110 = 256;
  word_4FB406C &= 0xF000u;
  qword_4FB4070 = 0;
  qword_4FB4078 = 0;
  qword_4FB4080 = 0;
  dword_4FB4068 = v1;
  qword_4FB4108 = (__int64)&unk_49E74E8;
  qword_4FB40A8 = (__int64)qword_4FA01C0;
  qword_4FB40B8 = (__int64)&unk_4FB40D8;
  qword_4FB40C0 = (__int64)&unk_4FB40D8;
  qword_4FB4060 = (__int64)&unk_49EEC70;
  qword_4FB4118 = (__int64)&unk_49EEDB0;
  qword_4FB4088 = 0;
  qword_4FB4090 = 0;
  qword_4FB4098 = 0;
  qword_4FB40A0 = 0;
  qword_4FB40B0 = 0;
  qword_4FB40C8 = 4;
  dword_4FB40D0 = 0;
  byte_4FB40F8 = 0;
  byte_4FB4100 = 0;
  sub_16B8280(&qword_4FB4060, "reassociate-geps-verify-no-dead-code", 36);
  qword_4FB4088 = (__int64)"Verify this pass produces no dead code";
  word_4FB4110 = 256;
  byte_4FB4100 = 0;
  qword_4FB4090 = 38;
  LOBYTE(word_4FB406C) = word_4FB406C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB4060);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB4060, &qword_4A427C0);
}
