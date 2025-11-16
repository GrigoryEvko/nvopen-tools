// Function: ctor_330
// Address: 0x505f50
//
int ctor_330()
{
  int v0; // edx

  qword_4FCABC0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCABCC &= 0xF000u;
  qword_4FCABD0 = 0;
  qword_4FCAC08 = (__int64)qword_4FA01C0;
  qword_4FCABD8 = 0;
  qword_4FCABE0 = 0;
  qword_4FCABE8 = 0;
  dword_4FCABC8 = v0;
  qword_4FCAC18 = (__int64)&unk_4FCAC38;
  qword_4FCAC20 = (__int64)&unk_4FCAC38;
  qword_4FCABF0 = 0;
  qword_4FCABF8 = 0;
  qword_4FCAC68 = (__int64)&unk_49E74E8;
  word_4FCAC70 = 256;
  qword_4FCAC00 = 0;
  qword_4FCAC10 = 0;
  qword_4FCABC0 = (__int64)&unk_49EEC70;
  qword_4FCAC28 = 4;
  byte_4FCAC58 = 0;
  qword_4FCAC78 = (__int64)&unk_49EEDB0;
  dword_4FCAC30 = 0;
  byte_4FCAC60 = 0;
  sub_16B8280(&qword_4FCABC0, "enable-selectiondag-sp", 22);
  word_4FCAC70 = 257;
  byte_4FCAC60 = 1;
  LOBYTE(word_4FCABCC) = word_4FCABCC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FCABC0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCABC0, &qword_4A427C0);
}
