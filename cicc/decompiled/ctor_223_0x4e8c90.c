// Function: ctor_223
// Address: 0x4e8c90
//
int ctor_223()
{
  int v0; // eax
  int v1; // eax

  qword_4FB4300 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB430C &= 0xF000u;
  qword_4FB4348 = (__int64)qword_4FA01C0;
  qword_4FB4310 = 0;
  qword_4FB4318 = 0;
  qword_4FB4320 = 0;
  dword_4FB4308 = v0;
  qword_4FB4358 = (__int64)&unk_4FB4378;
  qword_4FB4360 = (__int64)&unk_4FB4378;
  qword_4FB4328 = 0;
  qword_4FB4330 = 0;
  qword_4FB43A8 = (__int64)&unk_49E74E8;
  word_4FB43B0 = 256;
  qword_4FB4338 = 0;
  qword_4FB4340 = 0;
  qword_4FB4300 = (__int64)&unk_49EEC70;
  qword_4FB4350 = 0;
  byte_4FB4398 = 0;
  qword_4FB43B8 = (__int64)&unk_49EEDB0;
  qword_4FB4368 = 4;
  dword_4FB4370 = 0;
  byte_4FB43A0 = 0;
  sub_16B8280(&qword_4FB4300, "enable-nontrivial-unswitch", 26);
  word_4FB43B0 = 256;
  byte_4FB43A0 = 0;
  qword_4FB4330 = 107;
  LOBYTE(word_4FB430C) = word_4FB430C & 0x9F | 0x20;
  qword_4FB4328 = (__int64)"Forcibly enables non-trivial loop unswitching rather than following the configuration passed into the pass.";
  sub_16B88A0(&qword_4FB4300);
  __cxa_atexit(sub_12EDEC0, &qword_4FB4300, &qword_4A427C0);
  qword_4FB4220 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB422C &= 0xF000u;
  qword_4FB4230 = 0;
  qword_4FB4238 = 0;
  qword_4FB4240 = 0;
  qword_4FB4248 = 0;
  qword_4FB4250 = 0;
  dword_4FB4228 = v1;
  qword_4FB4278 = (__int64)&unk_4FB4298;
  qword_4FB4280 = (__int64)&unk_4FB4298;
  qword_4FB4268 = (__int64)qword_4FA01C0;
  qword_4FB4258 = 0;
  qword_4FB42C8 = (__int64)&unk_49E74C8;
  qword_4FB4260 = 0;
  qword_4FB4270 = 0;
  qword_4FB4220 = (__int64)&unk_49EEB70;
  qword_4FB4288 = 4;
  dword_4FB4290 = 0;
  qword_4FB42D8 = (__int64)&unk_49EEDF0;
  byte_4FB42B8 = 0;
  dword_4FB42C0 = 0;
  byte_4FB42D4 = 1;
  dword_4FB42D0 = 0;
  sub_16B8280(&qword_4FB4220, "unswitch-threshold", 18);
  dword_4FB42C0 = 50;
  byte_4FB42D4 = 1;
  dword_4FB42D0 = 50;
  qword_4FB4250 = 42;
  LOBYTE(word_4FB422C) = word_4FB422C & 0x9F | 0x20;
  qword_4FB4248 = (__int64)"The cost threshold for unswitching a loop.";
  sub_16B88A0(&qword_4FB4220);
  return __cxa_atexit(sub_12EDEA0, &qword_4FB4220, &qword_4A427C0);
}
