// Function: ctor_113
// Address: 0x4ab310
//
int ctor_113()
{
  int v0; // edx

  qword_4F97AE0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F97AEC &= 0xF000u;
  qword_4F97AF0 = 0;
  qword_4F97B28 = (__int64)&unk_4FA01C0;
  qword_4F97AF8 = 0;
  qword_4F97B00 = 0;
  qword_4F97B08 = 0;
  dword_4F97AE8 = v0;
  qword_4F97B38 = (__int64)&unk_4F97B58;
  qword_4F97B40 = (__int64)&unk_4F97B58;
  qword_4F97B10 = 0;
  qword_4F97B18 = 0;
  qword_4F97B88 = (__int64)&unk_49E74A8;
  qword_4F97B20 = 0;
  qword_4F97B30 = 0;
  qword_4F97AE0 = (__int64)&unk_49EEAF0;
  qword_4F97B48 = 4;
  dword_4F97B50 = 0;
  qword_4F97B98 = (__int64)&unk_49EEE10;
  byte_4F97B78 = 0;
  dword_4F97B80 = 0;
  byte_4F97B94 = 1;
  dword_4F97B90 = 0;
  sub_16B8280(&qword_4F97AE0, "alias-set-saturation-threshold", 30);
  dword_4F97B80 = 250;
  byte_4F97B94 = 1;
  dword_4F97B90 = 250;
  qword_4F97B10 = 76;
  LOBYTE(word_4F97AEC) = word_4F97AEC & 0x9F | 0x20;
  qword_4F97B08 = (__int64)"The maximum number of pointers may-alias sets may contain before degradation";
  sub_16B88A0(&qword_4F97AE0);
  return __cxa_atexit(sub_12EDE60, &qword_4F97AE0, &qword_4A427C0);
}
