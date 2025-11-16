// Function: ctor_240
// Address: 0x4ecb40
//
int ctor_240()
{
  int v0; // edx

  qword_4FB6B40 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB6B4C &= 0xF000u;
  qword_4FB6B50 = 0;
  qword_4FB6B88 = (__int64)qword_4FA01C0;
  qword_4FB6B58 = 0;
  qword_4FB6B60 = 0;
  qword_4FB6B68 = 0;
  dword_4FB6B48 = v0;
  qword_4FB6B98 = (__int64)&unk_4FB6BB8;
  qword_4FB6BA0 = (__int64)&unk_4FB6BB8;
  qword_4FB6B70 = 0;
  qword_4FB6B78 = 0;
  qword_4FB6BE8 = (__int64)&unk_49E74E8;
  word_4FB6BF0 = 256;
  qword_4FB6B80 = 0;
  qword_4FB6B90 = 0;
  qword_4FB6B40 = (__int64)&unk_49EEC70;
  qword_4FB6BA8 = 4;
  byte_4FB6BD8 = 0;
  qword_4FB6BF8 = (__int64)&unk_49EEDB0;
  dword_4FB6BB0 = 0;
  byte_4FB6BE0 = 0;
  sub_16B8280(&qword_4FB6B40, "loop-version-annotate-no-alias", 30);
  word_4FB6BF0 = 257;
  byte_4FB6BE0 = 1;
  qword_4FB6B70 = 76;
  LOBYTE(word_4FB6B4C) = word_4FB6B4C & 0x9F | 0x20;
  qword_4FB6B68 = (__int64)"Add no-alias annotation for instructions that are disambiguated by memchecks";
  sub_16B88A0(&qword_4FB6B40);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB6B40, &qword_4A427C0);
}
