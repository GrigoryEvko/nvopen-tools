// Function: ctor_209
// Address: 0x4e3b90
//
int ctor_209()
{
  int v0; // eax
  int v1; // eax

  qword_4FB0900 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB090C &= 0xF000u;
  qword_4FB0910 = 0;
  qword_4FB0918 = 0;
  qword_4FB0920 = 0;
  qword_4FB0928 = 0;
  qword_4FB0930 = 0;
  dword_4FB0908 = v0;
  qword_4FB0938 = 0;
  qword_4FB0948 = (__int64)qword_4FA01C0;
  qword_4FB0958 = (__int64)&unk_4FB0978;
  qword_4FB0960 = (__int64)&unk_4FB0978;
  qword_4FB0940 = 0;
  qword_4FB0950 = 0;
  qword_4FB09A8 = (__int64)&unk_49E74A8;
  qword_4FB0968 = 4;
  qword_4FB0900 = (__int64)&unk_49EEAF0;
  dword_4FB0970 = 0;
  qword_4FB09B8 = (__int64)&unk_49EEE10;
  byte_4FB0998 = 0;
  dword_4FB09A0 = 0;
  byte_4FB09B4 = 1;
  dword_4FB09B0 = 0;
  sub_16B8280(&qword_4FB0900, "runtime-check-per-loop-load-elim", 32);
  qword_4FB0930 = 62;
  dword_4FB09A0 = 1;
  byte_4FB09B4 = 1;
  dword_4FB09B0 = 1;
  LOBYTE(word_4FB090C) = word_4FB090C & 0x9F | 0x20;
  qword_4FB0928 = (__int64)"Max number of memchecks allowed per eliminated load on average";
  sub_16B88A0(&qword_4FB0900);
  __cxa_atexit(sub_12EDE60, &qword_4FB0900, &qword_4A427C0);
  qword_4FB0820 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB082C &= 0xF000u;
  qword_4FB0830 = 0;
  qword_4FB0838 = 0;
  qword_4FB0840 = 0;
  qword_4FB0848 = 0;
  qword_4FB0850 = 0;
  dword_4FB0828 = v1;
  qword_4FB08C8 = (__int64)&unk_49E74A8;
  qword_4FB0868 = (__int64)qword_4FA01C0;
  qword_4FB0878 = (__int64)&unk_4FB0898;
  qword_4FB0880 = (__int64)&unk_4FB0898;
  qword_4FB0820 = (__int64)&unk_49EEAF0;
  qword_4FB08D8 = (__int64)&unk_49EEE10;
  qword_4FB0858 = 0;
  qword_4FB0860 = 0;
  qword_4FB0870 = 0;
  qword_4FB0888 = 4;
  dword_4FB0890 = 0;
  byte_4FB08B8 = 0;
  dword_4FB08C0 = 0;
  byte_4FB08D4 = 1;
  dword_4FB08D0 = 0;
  sub_16B8280(&qword_4FB0820, "loop-load-elimination-scev-check-threshold", 42);
  dword_4FB08C0 = 8;
  byte_4FB08D4 = 1;
  dword_4FB08D0 = 8;
  qword_4FB0850 = 67;
  LOBYTE(word_4FB082C) = word_4FB082C & 0x9F | 0x20;
  qword_4FB0848 = (__int64)"The maximum number of SCEV checks allowed for Loop Load Elimination";
  sub_16B88A0(&qword_4FB0820);
  return __cxa_atexit(sub_12EDE60, &qword_4FB0820, &qword_4A427C0);
}
