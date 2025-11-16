// Function: ctor_218
// Address: 0x4e7a30
//
int ctor_218()
{
  int v0; // eax
  int v1; // eax

  qword_4FB3840 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB384C &= 0xF000u;
  qword_4FB3888 = (__int64)qword_4FA01C0;
  qword_4FB3850 = 0;
  qword_4FB3858 = 0;
  qword_4FB3860 = 0;
  dword_4FB3848 = v0;
  qword_4FB3898 = (__int64)&unk_4FB38B8;
  qword_4FB38A0 = (__int64)&unk_4FB38B8;
  qword_4FB3868 = 0;
  qword_4FB3870 = 0;
  qword_4FB38E8 = (__int64)&unk_49F1258;
  qword_4FB3878 = 0;
  qword_4FB3880 = 0;
  qword_4FB3840 = (__int64)&unk_49F1278;
  qword_4FB3890 = 0;
  byte_4FB38D8 = 0;
  qword_4FB38F8 = (__int64)&unk_49EEE70;
  qword_4FB38A8 = 4;
  dword_4FB38B0 = 0;
  dword_4FB38E0 = 0;
  byte_4FB38F4 = 1;
  dword_4FB38F0 = 0;
  sub_16B8280(&qword_4FB3840, "licm-versioning-invariant-threshold", 35);
  qword_4FB3868 = (__int64)"LoopVersioningLICM's minimum allowed percentageof possible invariant instructions per loop";
  dword_4FB38E0 = 1103626240;
  dword_4FB38F0 = 1103626240;
  byte_4FB38F4 = 1;
  LOBYTE(word_4FB384C) = word_4FB384C & 0x9F | 0x20;
  qword_4FB3870 = 90;
  sub_16B88A0(&qword_4FB3840);
  __cxa_atexit(sub_1851380, &qword_4FB3840, &qword_4A427C0);
  qword_4FB3760 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB376C &= 0xF000u;
  qword_4FB3770 = 0;
  qword_4FB3778 = 0;
  qword_4FB3780 = 0;
  qword_4FB3788 = 0;
  qword_4FB3790 = 0;
  dword_4FB3768 = v1;
  qword_4FB37B8 = (__int64)&unk_4FB37D8;
  qword_4FB37C0 = (__int64)&unk_4FB37D8;
  qword_4FB37A8 = (__int64)qword_4FA01C0;
  qword_4FB3798 = 0;
  qword_4FB3808 = (__int64)&unk_49E74A8;
  qword_4FB37A0 = 0;
  qword_4FB37B0 = 0;
  qword_4FB3760 = (__int64)&unk_49EEAF0;
  qword_4FB37C8 = 4;
  dword_4FB37D0 = 0;
  qword_4FB3818 = (__int64)&unk_49EEE10;
  byte_4FB37F8 = 0;
  dword_4FB3800 = 0;
  byte_4FB3814 = 1;
  dword_4FB3810 = 0;
  sub_16B8280(&qword_4FB3760, "licm-versioning-max-depth-threshold", 35);
  qword_4FB3790 = 66;
  qword_4FB3788 = (__int64)"LoopVersioningLICM's threshold for maximum allowed loop nest/depth";
  dword_4FB3800 = 2;
  byte_4FB3814 = 1;
  dword_4FB3810 = 2;
  LOBYTE(word_4FB376C) = word_4FB376C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB3760);
  return __cxa_atexit(sub_12EDE60, &qword_4FB3760, &qword_4A427C0);
}
