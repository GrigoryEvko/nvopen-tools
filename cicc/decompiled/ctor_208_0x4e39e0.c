// Function: ctor_208
// Address: 0x4e39e0
//
int ctor_208()
{
  int v0; // edx

  qword_4FB0740 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB074C &= 0xF000u;
  qword_4FB0750 = 0;
  qword_4FB0788 = (__int64)qword_4FA01C0;
  qword_4FB0758 = 0;
  qword_4FB0760 = 0;
  qword_4FB0768 = 0;
  dword_4FB0748 = v0;
  qword_4FB0798 = (__int64)&unk_4FB07B8;
  qword_4FB07A0 = (__int64)&unk_4FB07B8;
  qword_4FB0770 = 0;
  qword_4FB0778 = 0;
  qword_4FB07E8 = (__int64)&unk_49E74C8;
  qword_4FB0780 = 0;
  qword_4FB0790 = 0;
  qword_4FB0740 = (__int64)&unk_49EEB70;
  qword_4FB07A8 = 4;
  dword_4FB07B0 = 0;
  qword_4FB07F8 = (__int64)&unk_49EEDF0;
  byte_4FB07D8 = 0;
  dword_4FB07E0 = 0;
  byte_4FB07F4 = 1;
  dword_4FB07F0 = 0;
  sub_16B8280(&qword_4FB0740, "loop-interchange-threshold", 26);
  dword_4FB07E0 = 0;
  byte_4FB07F4 = 1;
  dword_4FB07F0 = 0;
  qword_4FB0770 = 45;
  LOBYTE(word_4FB074C) = word_4FB074C & 0x9F | 0x20;
  qword_4FB0768 = (__int64)"Interchange if you gain more than this number";
  sub_16B88A0(&qword_4FB0740);
  return __cxa_atexit(sub_12EDEA0, &qword_4FB0740, &qword_4A427C0);
}
