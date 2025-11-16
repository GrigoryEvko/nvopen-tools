// Function: ctor_300
// Address: 0x4fe640
//
int ctor_300()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4FC5BC0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC5BCC &= 0xF000u;
  qword_4FC5C08 = (__int64)qword_4FA01C0;
  qword_4FC5BD0 = 0;
  qword_4FC5BD8 = 0;
  qword_4FC5BE0 = 0;
  dword_4FC5BC8 = v0;
  qword_4FC5C18 = (__int64)&unk_4FC5C38;
  qword_4FC5C20 = (__int64)&unk_4FC5C38;
  qword_4FC5BE8 = 0;
  qword_4FC5BF0 = 0;
  qword_4FC5C68 = (__int64)&unk_49E74A8;
  qword_4FC5BF8 = 0;
  qword_4FC5C00 = 0;
  qword_4FC5BC0 = (__int64)&unk_49EEAF0;
  qword_4FC5C10 = 0;
  byte_4FC5C58 = 0;
  qword_4FC5C78 = (__int64)&unk_49EEE10;
  qword_4FC5C28 = 4;
  dword_4FC5C30 = 0;
  dword_4FC5C60 = 0;
  byte_4FC5C74 = 1;
  dword_4FC5C70 = 0;
  sub_16B8280(&qword_4FC5BC0, "machine-combiner-inc-threshold", 30);
  qword_4FC5BF0 = 83;
  dword_4FC5C60 = 500;
  byte_4FC5C74 = 1;
  dword_4FC5C70 = 500;
  LOBYTE(word_4FC5BCC) = word_4FC5BCC & 0x9F | 0x20;
  qword_4FC5BE8 = (__int64)"Incremental depth computation will be used for basic blocks with more instructions.";
  sub_16B88A0(&qword_4FC5BC0);
  __cxa_atexit(sub_12EDE60, &qword_4FC5BC0, &qword_4A427C0);
  qword_4FC5AE0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC5AEC &= 0xF000u;
  qword_4FC5AF0 = 0;
  qword_4FC5AF8 = 0;
  qword_4FC5B00 = 0;
  qword_4FC5B08 = 0;
  qword_4FC5B10 = 0;
  dword_4FC5AE8 = v1;
  qword_4FC5B38 = (__int64)&unk_4FC5B58;
  qword_4FC5B40 = (__int64)&unk_4FC5B58;
  qword_4FC5B28 = (__int64)qword_4FA01C0;
  qword_4FC5B18 = 0;
  word_4FC5B90 = 256;
  qword_4FC5B88 = (__int64)&unk_49E74E8;
  qword_4FC5AE0 = (__int64)&unk_49EEC70;
  qword_4FC5B98 = (__int64)&unk_49EEDB0;
  qword_4FC5B20 = 0;
  qword_4FC5B30 = 0;
  qword_4FC5B48 = 4;
  dword_4FC5B50 = 0;
  byte_4FC5B78 = 0;
  byte_4FC5B80 = 0;
  sub_16B8280(&qword_4FC5AE0, "machine-combiner-dump-subst-intrs", 33);
  word_4FC5B90 = 256;
  byte_4FC5B80 = 0;
  qword_4FC5B10 = 26;
  LOBYTE(word_4FC5AEC) = word_4FC5AEC & 0x9F | 0x20;
  qword_4FC5B08 = (__int64)"Dump all substituted intrs";
  sub_16B88A0(&qword_4FC5AE0);
  __cxa_atexit(sub_12EDEC0, &qword_4FC5AE0, &qword_4A427C0);
  qword_4FC5A00 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC5AB0 = 256;
  qword_4FC5A10 = 0;
  word_4FC5A0C &= 0xF000u;
  qword_4FC5AA8 = (__int64)&unk_49E74E8;
  qword_4FC5A00 = (__int64)&unk_49EEC70;
  dword_4FC5A08 = v2;
  qword_4FC5A48 = (__int64)qword_4FA01C0;
  qword_4FC5A58 = (__int64)&unk_4FC5A78;
  qword_4FC5A60 = (__int64)&unk_4FC5A78;
  qword_4FC5AB8 = (__int64)&unk_49EEDB0;
  qword_4FC5A18 = 0;
  qword_4FC5A20 = 0;
  qword_4FC5A28 = 0;
  qword_4FC5A30 = 0;
  qword_4FC5A38 = 0;
  qword_4FC5A40 = 0;
  qword_4FC5A50 = 0;
  qword_4FC5A68 = 4;
  dword_4FC5A70 = 0;
  byte_4FC5A98 = 0;
  byte_4FC5AA0 = 0;
  sub_16B8280(&qword_4FC5A00, "machine-combiner-verify-pattern-order", 37);
  word_4FC5AB0 = 256;
  byte_4FC5AA0 = 0;
  qword_4FC5A30 = 68;
  LOBYTE(word_4FC5A0C) = word_4FC5A0C & 0x9F | 0x20;
  qword_4FC5A28 = (__int64)"Verify that the generated patterns are ordered by increasing latency";
  sub_16B88A0(&qword_4FC5A00);
  return __cxa_atexit(sub_12EDEC0, &qword_4FC5A00, &qword_4A427C0);
}
