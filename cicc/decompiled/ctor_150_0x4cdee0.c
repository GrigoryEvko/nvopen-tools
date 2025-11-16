// Function: ctor_150
// Address: 0x4cdee0
//
int ctor_150()
{
  int v0; // edx

  qword_4F9EEA0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9EEAC &= 0xF000u;
  qword_4F9EEB0 = 0;
  qword_4F9EEE8 = (__int64)&unk_4FA01C0;
  qword_4F9EEB8 = 0;
  qword_4F9EEC0 = 0;
  qword_4F9EEC8 = 0;
  dword_4F9EEA8 = v0;
  qword_4F9EEF8 = (__int64)&unk_4F9EF18;
  qword_4F9EF00 = (__int64)&unk_4F9EF18;
  qword_4F9EED0 = 0;
  qword_4F9EED8 = 0;
  qword_4F9EF48 = (__int64)&unk_49E74E8;
  word_4F9EF50 = 256;
  qword_4F9EEE0 = 0;
  qword_4F9EEF0 = 0;
  qword_4F9EEA0 = (__int64)&unk_49EEC70;
  qword_4F9EF08 = 4;
  byte_4F9EF38 = 0;
  qword_4F9EF58 = (__int64)&unk_49EEDB0;
  dword_4F9EF10 = 0;
  byte_4F9EF40 = 0;
  sub_16B8280(&qword_4F9EEA0, "safepoint-ir-verifier-print-only", 32);
  byte_4F9EF40 = 0;
  word_4F9EF50 = 256;
  sub_16B88A0(&qword_4F9EEA0);
  return __cxa_atexit(sub_12EDEC0, &qword_4F9EEA0, &qword_4A427C0);
}
