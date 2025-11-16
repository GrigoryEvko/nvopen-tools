// Function: ctor_274
// Address: 0x4f75b0
//
int ctor_274()
{
  int v0; // edx

  qword_4FBF020 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBF02C &= 0xF000u;
  qword_4FBF030 = 0;
  qword_4FBF068 = (__int64)qword_4FA01C0;
  qword_4FBF038 = 0;
  qword_4FBF040 = 0;
  qword_4FBF048 = 0;
  dword_4FBF028 = v0;
  qword_4FBF078 = (__int64)&unk_4FBF098;
  qword_4FBF080 = (__int64)&unk_4FBF098;
  qword_4FBF050 = 0;
  qword_4FBF058 = 0;
  qword_4FBF0C8 = (__int64)&unk_49E74E8;
  word_4FBF0D0 = 256;
  qword_4FBF060 = 0;
  qword_4FBF070 = 0;
  qword_4FBF020 = (__int64)&unk_49EEC70;
  qword_4FBF088 = 4;
  byte_4FBF0B8 = 0;
  qword_4FBF0D8 = (__int64)&unk_49EEDB0;
  dword_4FBF090 = 0;
  byte_4FBF0C0 = 0;
  sub_16B8280(&qword_4FBF020, "use-max-local-array-alignment", 29);
  word_4FBF0D0 = 256;
  byte_4FBF0C0 = 0;
  qword_4FBF050 = 39;
  LOBYTE(word_4FBF02C) = word_4FBF02C & 0x9F | 0x20;
  qword_4FBF048 = (__int64)"Use mmaximum alignment for local memory";
  sub_16B88A0(&qword_4FBF020);
  return __cxa_atexit(sub_12EDEC0, &qword_4FBF020, &qword_4A427C0);
}
