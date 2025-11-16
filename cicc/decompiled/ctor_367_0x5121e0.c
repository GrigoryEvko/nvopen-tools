// Function: ctor_367
// Address: 0x5121e0
//
int ctor_367()
{
  int v0; // edx

  qword_4FD4320 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FD432C &= 0xF000u;
  qword_4FD4330 = 0;
  qword_4FD4368 = (__int64)qword_4FA01C0;
  qword_4FD4338 = 0;
  qword_4FD4340 = 0;
  qword_4FD4348 = 0;
  dword_4FD4328 = v0;
  qword_4FD4378 = (__int64)&unk_4FD4398;
  qword_4FD4380 = (__int64)&unk_4FD4398;
  qword_4FD4350 = 0;
  qword_4FD4358 = 0;
  qword_4FD43C8 = (__int64)&unk_49E74E8;
  word_4FD43D0 = 256;
  qword_4FD4360 = 0;
  qword_4FD4370 = 0;
  qword_4FD4320 = (__int64)&unk_49EEC70;
  qword_4FD4388 = 4;
  byte_4FD43B8 = 0;
  qword_4FD43D8 = (__int64)&unk_49EEDB0;
  dword_4FD4390 = 0;
  byte_4FD43C0 = 0;
  sub_16B8280(&qword_4FD4320, "sink-ld-param", 13);
  word_4FD43D0 = 256;
  byte_4FD43C0 = 0;
  qword_4FD4350 = 38;
  LOBYTE(word_4FD432C) = word_4FD432C & 0x9F | 0x20;
  qword_4FD4348 = (__int64)"Sink one-use ld.param to the use point";
  sub_16B88A0(&qword_4FD4320);
  return __cxa_atexit(sub_12EDEC0, &qword_4FD4320, &qword_4A427C0);
}
