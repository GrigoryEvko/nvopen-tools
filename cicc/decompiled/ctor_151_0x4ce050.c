// Function: ctor_151
// Address: 0x4ce050
//
int ctor_151()
{
  int v0; // edx

  qword_4F9EF80 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9EF8C &= 0xF000u;
  qword_4F9EF90 = 0;
  qword_4F9EFC8 = (__int64)&unk_4FA01C0;
  qword_4F9EF98 = 0;
  qword_4F9EFA0 = 0;
  qword_4F9EFA8 = 0;
  dword_4F9EF88 = v0;
  qword_4F9EFD8 = (__int64)&unk_4F9EFF8;
  qword_4F9EFE0 = (__int64)&unk_4F9EFF8;
  qword_4F9EFB0 = 0;
  qword_4F9EFB8 = 0;
  qword_4F9F028 = (__int64)&unk_49E74A8;
  qword_4F9EFC0 = 0;
  qword_4F9EFD0 = 0;
  qword_4F9EF80 = (__int64)&unk_49EEAF0;
  qword_4F9EFE8 = 4;
  dword_4F9EFF0 = 0;
  qword_4F9F038 = (__int64)&unk_49EEE10;
  byte_4F9F018 = 0;
  dword_4F9F020 = 0;
  byte_4F9F034 = 1;
  dword_4F9F030 = 0;
  sub_16B8280(&qword_4F9EF80, "non-global-value-max-name-size", 30);
  dword_4F9F020 = 1024;
  byte_4F9F034 = 1;
  dword_4F9F030 = 1024;
  qword_4F9EFB0 = 47;
  LOBYTE(word_4F9EF8C) = word_4F9EF8C & 0x9F | 0x20;
  qword_4F9EFA8 = (__int64)"Maximum size for the name of non-global values.";
  sub_16B88A0(&qword_4F9EF80);
  return __cxa_atexit(sub_12EDE60, &qword_4F9EF80, &qword_4A427C0);
}
