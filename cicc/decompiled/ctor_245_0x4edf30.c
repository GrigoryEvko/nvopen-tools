// Function: ctor_245
// Address: 0x4edf30
//
int ctor_245()
{
  int v0; // edx

  qword_4FB77C0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB77CC &= 0xF000u;
  qword_4FB77D0 = 0;
  qword_4FB7808 = (__int64)qword_4FA01C0;
  qword_4FB77D8 = 0;
  qword_4FB77E0 = 0;
  qword_4FB77E8 = 0;
  dword_4FB77C8 = v0;
  qword_4FB7818 = (__int64)&unk_4FB7838;
  qword_4FB7820 = (__int64)&unk_4FB7838;
  qword_4FB77F0 = 0;
  qword_4FB77F8 = 0;
  qword_4FB7868 = (__int64)&unk_49E74E8;
  word_4FB7870 = 256;
  qword_4FB7800 = 0;
  qword_4FB7810 = 0;
  qword_4FB77C0 = (__int64)&unk_49EEC70;
  qword_4FB7828 = 4;
  byte_4FB7858 = 0;
  qword_4FB7878 = (__int64)&unk_49EEDB0;
  dword_4FB7830 = 0;
  byte_4FB7860 = 0;
  sub_16B8280(&qword_4FB77C0, "no-discriminators", 17);
  word_4FB7870 = 256;
  byte_4FB7860 = 0;
  qword_4FB77E8 = (__int64)"Disable generation of discriminator information.";
  qword_4FB77F0 = 48;
  sub_16B88A0(&qword_4FB77C0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB77C0, &qword_4A427C0);
}
