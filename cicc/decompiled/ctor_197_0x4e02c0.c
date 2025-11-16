// Function: ctor_197
// Address: 0x4e02c0
//
int ctor_197()
{
  int v0; // edx

  qword_4FAE280 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAE28C &= 0xF000u;
  qword_4FAE290 = 0;
  qword_4FAE2C8 = (__int64)qword_4FA01C0;
  qword_4FAE298 = 0;
  qword_4FAE2A0 = 0;
  qword_4FAE2A8 = 0;
  dword_4FAE288 = v0;
  qword_4FAE2D8 = (__int64)&unk_4FAE2F8;
  qword_4FAE2E0 = (__int64)&unk_4FAE2F8;
  qword_4FAE2B0 = 0;
  qword_4FAE2B8 = 0;
  qword_4FAE328 = (__int64)&unk_49E74E8;
  word_4FAE330 = 256;
  qword_4FAE2C0 = 0;
  qword_4FAE2D0 = 0;
  qword_4FAE280 = (__int64)&unk_49EEC70;
  qword_4FAE2E8 = 4;
  byte_4FAE318 = 0;
  qword_4FAE338 = (__int64)&unk_49EEDB0;
  dword_4FAE2F0 = 0;
  byte_4FAE320 = 0;
  sub_16B8280(&qword_4FAE280, "cvp-dont-process-adds", 21);
  byte_4FAE320 = 1;
  word_4FAE330 = 257;
  sub_16B88A0(&qword_4FAE280);
  return __cxa_atexit(sub_12EDEC0, &qword_4FAE280, &qword_4A427C0);
}
