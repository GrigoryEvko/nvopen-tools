// Function: ctor_155
// Address: 0x4ce670
//
int ctor_155()
{
  int v0; // eax
  int v1; // eax

  qword_4F9F9A0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9F9AC &= 0xF000u;
  qword_4F9F9E8 = (__int64)&unk_4FA01C0;
  qword_4F9F9B0 = 0;
  qword_4F9F9B8 = 0;
  qword_4F9F9C0 = 0;
  dword_4F9F9A8 = v0;
  qword_4F9F9F8 = (__int64)&unk_4F9FA18;
  qword_4F9FA00 = (__int64)&unk_4F9FA18;
  qword_4F9F9C8 = 0;
  qword_4F9F9D0 = 0;
  qword_4F9FA48 = (__int64)&unk_49E74E8;
  word_4F9FA50 = 256;
  qword_4F9F9D8 = 0;
  qword_4F9F9E0 = 0;
  qword_4F9F9A0 = (__int64)&unk_49EEC70;
  qword_4F9F9F0 = 0;
  byte_4F9FA38 = 0;
  qword_4F9FA58 = (__int64)&unk_49EEDB0;
  qword_4F9FA08 = 4;
  dword_4F9FA10 = 0;
  byte_4F9FA40 = 0;
  sub_16B8280(&qword_4F9F9A0, "static-func-full-module-prefix", 30);
  word_4F9FA50 = 257;
  byte_4F9FA40 = 1;
  qword_4F9F9D0 = 78;
  LOBYTE(word_4F9F9AC) = word_4F9F9AC & 0x9F | 0x20;
  qword_4F9F9C8 = (__int64)"Use full module build paths in the profile counter names for static functions.";
  sub_16B88A0(&qword_4F9F9A0);
  __cxa_atexit(sub_12EDEC0, &qword_4F9F9A0, &qword_4A427C0);
  qword_4F9F8C0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9F8CC &= 0xF000u;
  qword_4F9F8D0 = 0;
  qword_4F9F8D8 = 0;
  qword_4F9F8E0 = 0;
  qword_4F9F8E8 = 0;
  qword_4F9F8F0 = 0;
  dword_4F9F8C8 = v1;
  qword_4F9F918 = (__int64)&unk_4F9F938;
  qword_4F9F920 = (__int64)&unk_4F9F938;
  qword_4F9F908 = (__int64)&unk_4FA01C0;
  qword_4F9F8F8 = 0;
  qword_4F9F968 = (__int64)&unk_49E74A8;
  qword_4F9F900 = 0;
  qword_4F9F910 = 0;
  qword_4F9F8C0 = (__int64)&unk_49EEAF0;
  qword_4F9F928 = 4;
  dword_4F9F930 = 0;
  qword_4F9F978 = (__int64)&unk_49EEE10;
  byte_4F9F958 = 0;
  dword_4F9F960 = 0;
  byte_4F9F974 = 1;
  dword_4F9F970 = 0;
  sub_16B8280(&qword_4F9F8C0, "static-func-strip-dirname-prefix", 32);
  dword_4F9F960 = 0;
  byte_4F9F974 = 1;
  dword_4F9F970 = 0;
  qword_4F9F8F0 = 106;
  LOBYTE(word_4F9F8CC) = word_4F9F8CC & 0x9F | 0x20;
  qword_4F9F8E8 = (__int64)"Strip specified level of directory name from source path in the profile counter name for static functions.";
  sub_16B88A0(&qword_4F9F8C0);
  return __cxa_atexit(sub_12EDE60, &qword_4F9F8C0, &qword_4A427C0);
}
