// Function: ctor_185
// Address: 0x4dbba0
//
int ctor_185()
{
  int v0; // eax
  int v1; // eax

  qword_4FAB320 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAB32C &= 0xF000u;
  qword_4FAB368 = (__int64)qword_4FA01C0;
  qword_4FAB330 = 0;
  qword_4FAB338 = 0;
  qword_4FAB340 = 0;
  dword_4FAB328 = v0;
  qword_4FAB378 = (__int64)&unk_4FAB398;
  qword_4FAB380 = (__int64)&unk_4FAB398;
  qword_4FAB348 = 0;
  qword_4FAB350 = 0;
  qword_4FAB3C8 = (__int64)&unk_49E74E8;
  word_4FAB3D0 = 256;
  qword_4FAB358 = 0;
  qword_4FAB360 = 0;
  qword_4FAB320 = (__int64)&unk_49EEC70;
  qword_4FAB370 = 0;
  byte_4FAB3B8 = 0;
  qword_4FAB3D8 = (__int64)&unk_49EEDB0;
  qword_4FAB388 = 4;
  dword_4FAB390 = 0;
  byte_4FAB3C0 = 0;
  sub_16B8280(&qword_4FAB320, "enable-coldcc-stress-test", 25);
  qword_4FAB348 = (__int64)"Enable stress test of coldcc by adding calling conv to all internal functions.";
  word_4FAB3D0 = 256;
  byte_4FAB3C0 = 0;
  qword_4FAB350 = 78;
  LOBYTE(word_4FAB32C) = word_4FAB32C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAB320);
  __cxa_atexit(sub_12EDEC0, &qword_4FAB320, &qword_4A427C0);
  qword_4FAB240 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAB24C &= 0xF000u;
  qword_4FAB250 = 0;
  qword_4FAB258 = 0;
  qword_4FAB260 = 0;
  qword_4FAB268 = 0;
  qword_4FAB270 = 0;
  dword_4FAB248 = v1;
  qword_4FAB298 = (__int64)&unk_4FAB2B8;
  qword_4FAB2A0 = (__int64)&unk_4FAB2B8;
  qword_4FAB288 = (__int64)qword_4FA01C0;
  qword_4FAB278 = 0;
  qword_4FAB2E8 = (__int64)&unk_49E74C8;
  qword_4FAB280 = 0;
  qword_4FAB290 = 0;
  qword_4FAB240 = (__int64)&unk_49EEB70;
  qword_4FAB2A8 = 4;
  dword_4FAB2B0 = 0;
  qword_4FAB2F8 = (__int64)&unk_49EEDF0;
  byte_4FAB2D8 = 0;
  dword_4FAB2E0 = 0;
  byte_4FAB2F4 = 1;
  dword_4FAB2F0 = 0;
  sub_16B8280(&qword_4FAB240, "coldcc-rel-freq", 15);
  dword_4FAB2E0 = 2;
  byte_4FAB2F4 = 1;
  dword_4FAB2F0 = 2;
  qword_4FAB270 = 136;
  LOBYTE(word_4FAB24C) = word_4FAB24C & 0x98 | 0x21;
  qword_4FAB268 = (__int64)"Maximum block frequency, expressed as a percentage of caller's entry frequency, for a call si"
                           "te to be considered cold for enablingcoldcc";
  sub_16B88A0(&qword_4FAB240);
  return __cxa_atexit(sub_12EDEA0, &qword_4FAB240, &qword_4A427C0);
}
