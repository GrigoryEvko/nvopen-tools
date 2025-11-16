// Function: ctor_226
// Address: 0x4e99a0
//
int ctor_226()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4FB4CA0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB4CAC &= 0xF000u;
  qword_4FB4CE8 = (__int64)qword_4FA01C0;
  qword_4FB4CB0 = 0;
  qword_4FB4CB8 = 0;
  qword_4FB4CC0 = 0;
  dword_4FB4CA8 = v0;
  qword_4FB4CF8 = (__int64)&unk_4FB4D18;
  qword_4FB4D00 = (__int64)&unk_4FB4D18;
  qword_4FB4CC8 = 0;
  qword_4FB4CD0 = 0;
  qword_4FB4D48 = (__int64)&unk_49E74A8;
  qword_4FB4CA0 = (__int64)&unk_49EEAF0;
  qword_4FB4D58 = (__int64)&unk_49EEE10;
  qword_4FB4CD8 = 0;
  qword_4FB4CE0 = 0;
  qword_4FB4CF0 = 0;
  qword_4FB4D08 = 4;
  dword_4FB4D10 = 0;
  byte_4FB4D38 = 0;
  dword_4FB4D40 = 0;
  byte_4FB4D54 = 1;
  dword_4FB4D50 = 0;
  sub_16B8280(&qword_4FB4CA0, "spec-exec-max-speculation-cost", 30);
  dword_4FB4D40 = 7;
  byte_4FB4D54 = 1;
  dword_4FB4D50 = 7;
  qword_4FB4CD0 = 132;
  LOBYTE(word_4FB4CAC) = word_4FB4CAC & 0x9F | 0x20;
  qword_4FB4CC8 = (__int64)"Speculative execution is not applied to basic blocks where the cost of the instructions to sp"
                           "eculatively execute exceeds this limit.";
  sub_16B88A0(&qword_4FB4CA0);
  __cxa_atexit(sub_12EDE60, &qword_4FB4CA0, &qword_4A427C0);
  qword_4FB4BC0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB4BCC &= 0xF000u;
  qword_4FB4BD0 = 0;
  qword_4FB4BD8 = 0;
  qword_4FB4BE0 = 0;
  qword_4FB4C68 = (__int64)&unk_49E74A8;
  qword_4FB4BC0 = (__int64)&unk_49EEAF0;
  dword_4FB4BC8 = v1;
  qword_4FB4C78 = (__int64)&unk_49EEE10;
  qword_4FB4C08 = (__int64)qword_4FA01C0;
  qword_4FB4C18 = (__int64)&unk_4FB4C38;
  qword_4FB4C20 = (__int64)&unk_4FB4C38;
  qword_4FB4BE8 = 0;
  qword_4FB4BF0 = 0;
  qword_4FB4BF8 = 0;
  qword_4FB4C00 = 0;
  qword_4FB4C10 = 0;
  qword_4FB4C28 = 4;
  dword_4FB4C30 = 0;
  byte_4FB4C58 = 0;
  dword_4FB4C60 = 0;
  byte_4FB4C74 = 1;
  dword_4FB4C70 = 0;
  sub_16B8280(&qword_4FB4BC0, "spec-exec-max-not-hoisted", 25);
  dword_4FB4C60 = 5;
  byte_4FB4C74 = 1;
  dword_4FB4C70 = 5;
  qword_4FB4BF0 = 146;
  LOBYTE(word_4FB4BCC) = word_4FB4BCC & 0x9F | 0x20;
  qword_4FB4BE8 = (__int64)"Speculative execution is not applied to basic blocks where the number of instructions that wo"
                           "uld not be speculatively executed exceeds this limit.";
  sub_16B88A0(&qword_4FB4BC0);
  __cxa_atexit(sub_12EDE60, &qword_4FB4BC0, &qword_4A427C0);
  qword_4FB4AE0 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB4AEC &= 0xF000u;
  qword_4FB4AF0 = 0;
  qword_4FB4AF8 = 0;
  qword_4FB4B00 = 0;
  qword_4FB4B08 = 0;
  qword_4FB4B10 = 0;
  dword_4FB4AE8 = v2;
  qword_4FB4B38 = (__int64)&unk_4FB4B58;
  qword_4FB4B40 = (__int64)&unk_4FB4B58;
  qword_4FB4B28 = (__int64)qword_4FA01C0;
  qword_4FB4B18 = 0;
  qword_4FB4B88 = (__int64)&unk_49E74E8;
  word_4FB4B90 = 256;
  qword_4FB4B20 = 0;
  qword_4FB4B30 = 0;
  qword_4FB4AE0 = (__int64)&unk_49EEC70;
  qword_4FB4B48 = 4;
  byte_4FB4B78 = 0;
  qword_4FB4B98 = (__int64)&unk_49EEDB0;
  dword_4FB4B50 = 0;
  byte_4FB4B80 = 0;
  sub_16B8280(&qword_4FB4AE0, "spec-exec-only-if-divergent-target", 34);
  word_4FB4B90 = 256;
  byte_4FB4B80 = 0;
  qword_4FB4B10 = 135;
  LOBYTE(word_4FB4AEC) = word_4FB4AEC & 0x9F | 0x20;
  qword_4FB4B08 = (__int64)"Speculative execution is applied only to targets with divergent branches, even if the pass wa"
                           "s configured to apply only to all targets.";
  sub_16B88A0(&qword_4FB4AE0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB4AE0, &qword_4A427C0);
}
