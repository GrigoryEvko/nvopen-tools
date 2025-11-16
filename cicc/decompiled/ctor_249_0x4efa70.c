// Function: ctor_249
// Address: 0x4efa70
//
int ctor_249()
{
  int v0; // edx

  qword_4FB9660 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB966C &= 0xF000u;
  qword_4FB9670 = 0;
  qword_4FB96A8 = (__int64)qword_4FA01C0;
  qword_4FB9678 = 0;
  qword_4FB9680 = 0;
  qword_4FB9688 = 0;
  dword_4FB9668 = v0;
  qword_4FB96B8 = (__int64)&unk_4FB96D8;
  qword_4FB96C0 = (__int64)&unk_4FB96D8;
  qword_4FB9690 = 0;
  qword_4FB9698 = 0;
  qword_4FB9708 = (__int64)&unk_49E74E8;
  word_4FB9710 = 256;
  qword_4FB96A0 = 0;
  qword_4FB96B0 = 0;
  qword_4FB9660 = (__int64)&unk_49EEC70;
  qword_4FB96C8 = 4;
  byte_4FB96F8 = 0;
  qword_4FB9718 = (__int64)&unk_49EEDB0;
  dword_4FB96D0 = 0;
  byte_4FB9700 = 0;
  sub_16B8280(&qword_4FB9660, "vplan-verify-hcfg", 17);
  word_4FB9710 = 256;
  byte_4FB9700 = 0;
  qword_4FB9690 = 19;
  LOBYTE(word_4FB966C) = word_4FB966C & 0x9F | 0x20;
  qword_4FB9688 = (__int64)"Verify VPlan H-CFG.";
  sub_16B88A0(&qword_4FB9660);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB9660, &qword_4A427C0);
}
