// Function: ctor_125
// Address: 0x4adb50
//
int ctor_125()
{
  int v0; // eax
  int v1; // eax

  qword_4F99860 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9986C &= 0xF000u;
  qword_4F998A8 = (__int64)&unk_4FA01C0;
  qword_4F99870 = 0;
  qword_4F99878 = 0;
  qword_4F99880 = 0;
  dword_4F99868 = v0;
  qword_4F998B8 = (__int64)&unk_4F998D8;
  qword_4F998C0 = (__int64)&unk_4F998D8;
  qword_4F99888 = 0;
  qword_4F99890 = 0;
  qword_4F99908 = (__int64)&unk_49E74A8;
  qword_4F99898 = 0;
  qword_4F998A0 = 0;
  qword_4F99860 = (__int64)&unk_49EEAF0;
  qword_4F998B0 = 0;
  byte_4F998F8 = 0;
  qword_4F99918 = (__int64)&unk_49EEE10;
  qword_4F998C8 = 4;
  dword_4F998D0 = 0;
  dword_4F99900 = 0;
  byte_4F99914 = 1;
  dword_4F99910 = 0;
  sub_16B8280(&qword_4F99860, "memssa-check-limit", 18);
  dword_4F99900 = 100;
  byte_4F99914 = 1;
  dword_4F99910 = 100;
  qword_4F99890 = 92;
  LOBYTE(word_4F9986C) = word_4F9986C & 0x9F | 0x20;
  qword_4F99888 = (__int64)"The maximum number of stores/phis MemorySSAwill consider trying to walk past (default = 100)";
  sub_16B88A0(&qword_4F99860);
  __cxa_atexit(sub_12EDE60, &qword_4F99860, &qword_4A427C0);
  qword_4F99780 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9978C &= 0xF000u;
  qword_4F99790 = 0;
  qword_4F99798 = 0;
  qword_4F997A0 = 0;
  qword_4F997A8 = 0;
  qword_4F997B0 = 0;
  dword_4F99788 = v1;
  qword_4F997D8 = (__int64)&unk_4F997F8;
  qword_4F997E0 = (__int64)&unk_4F997F8;
  qword_4F997C8 = (__int64)&unk_4FA01C0;
  qword_4F997B8 = 0;
  qword_4F99828 = (__int64)&unk_49E74E8;
  word_4F99830 = 256;
  qword_4F997C0 = 0;
  qword_4F997D0 = 0;
  qword_4F99780 = (__int64)&unk_49EEC70;
  qword_4F997E8 = 4;
  byte_4F99818 = 0;
  qword_4F99838 = (__int64)&unk_49EEDB0;
  dword_4F997F0 = 0;
  byte_4F99820 = 0;
  sub_16B8280(&qword_4F99780, "verify-memoryssa", 16);
  word_4F99830 = 256;
  byte_4F99820 = 0;
  qword_4F997B0 = 40;
  LOBYTE(word_4F9978C) = word_4F9978C & 0x9F | 0x20;
  qword_4F997A8 = (__int64)"Verify MemorySSA in legacy printer pass.";
  sub_16B88A0(&qword_4F99780);
  return __cxa_atexit(sub_12EDEC0, &qword_4F99780, &qword_4A427C0);
}
