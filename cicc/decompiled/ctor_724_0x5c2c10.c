// Function: ctor_724
// Address: 0x5c2c10
//
int ctor_724()
{
  int v0; // edx

  qword_5054340 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_505434C &= 0xF000u;
  qword_5054350 = 0;
  qword_5054388 = (__int64)qword_4FA01C0;
  qword_5054358 = 0;
  qword_5054360 = 0;
  qword_5054368 = 0;
  dword_5054348 = v0;
  qword_5054398 = (__int64)&unk_50543B8;
  qword_50543A0 = (__int64)&unk_50543B8;
  qword_5054370 = 0;
  qword_5054378 = 0;
  qword_50543E8 = (__int64)&unk_49E74E8;
  word_50543F0 = 256;
  qword_5054380 = 0;
  qword_5054390 = 0;
  qword_5054340 = (__int64)&unk_49EEC70;
  qword_50543A8 = 4;
  byte_50543D8 = 0;
  qword_50543F8 = (__int64)&unk_49EEDB0;
  dword_50543B0 = 0;
  byte_50543E0 = 0;
  sub_16B8280(&qword_5054340, "enable-double-float-shrink", 26);
  word_50543F0 = 256;
  byte_50543E0 = 0;
  qword_5054370 = 58;
  LOBYTE(word_505434C) = word_505434C & 0x9F | 0x20;
  qword_5054368 = (__int64)"Enable unsafe double to float shrinking for math lib calls";
  sub_16B88A0(&qword_5054340);
  return __cxa_atexit(sub_12EDEC0, &qword_5054340, &qword_4A427C0);
}
