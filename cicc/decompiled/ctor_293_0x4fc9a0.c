// Function: ctor_293
// Address: 0x4fc9a0
//
int ctor_293()
{
  int v0; // edx

  qword_4FC4280 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC428C &= 0xF000u;
  qword_4FC4290 = 0;
  qword_4FC42C8 = (__int64)qword_4FA01C0;
  qword_4FC4298 = 0;
  qword_4FC42A0 = 0;
  qword_4FC42A8 = 0;
  dword_4FC4288 = v0;
  qword_4FC42D8 = (__int64)&unk_4FC42F8;
  qword_4FC42E0 = (__int64)&unk_4FC42F8;
  qword_4FC42B0 = 0;
  qword_4FC42B8 = 0;
  qword_4FC4328 = (__int64)&unk_49E74E8;
  word_4FC4330 = 256;
  qword_4FC42C0 = 0;
  qword_4FC42D0 = 0;
  qword_4FC4280 = (__int64)&unk_49EEC70;
  qword_4FC42E8 = 4;
  byte_4FC4318 = 0;
  qword_4FC4338 = (__int64)&unk_49EEDB0;
  dword_4FC42F0 = 0;
  byte_4FC4320 = 0;
  sub_16B8280(&qword_4FC4280, "lower-interleaved-accesses", 26);
  qword_4FC42A8 = (__int64)"Enable lowering interleaved accesses to intrinsics";
  word_4FC4330 = 257;
  byte_4FC4320 = 1;
  qword_4FC42B0 = 50;
  LOBYTE(word_4FC428C) = word_4FC428C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC4280);
  return __cxa_atexit(sub_12EDEC0, &qword_4FC4280, &qword_4A427C0);
}
