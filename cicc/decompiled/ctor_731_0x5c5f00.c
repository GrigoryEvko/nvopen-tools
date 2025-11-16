// Function: ctor_731
// Address: 0x5c5f00
//
int ctor_731()
{
  int v0; // edx

  qword_5057540 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_505754C &= 0xF000u;
  qword_5057550 = 0;
  qword_5057588 = (__int64)qword_4FA01C0;
  qword_5057558 = 0;
  qword_5057560 = 0;
  qword_5057568 = 0;
  dword_5057548 = v0;
  qword_5057598 = (__int64)&unk_50575B8;
  qword_50575A0 = (__int64)&unk_50575B8;
  qword_5057570 = 0;
  qword_5057578 = 0;
  qword_50575E8 = (__int64)&unk_49E74E8;
  word_50575F0 = 256;
  qword_5057580 = 0;
  qword_5057590 = 0;
  qword_5057540 = (__int64)&unk_49EEC70;
  qword_50575A8 = 4;
  byte_50575D8 = 0;
  qword_50575F8 = (__int64)&unk_49EEDB0;
  dword_50575B0 = 0;
  byte_50575E0 = 0;
  sub_16B8280(&qword_5057540, "trap-unreachable", 16);
  word_50575F0 = 256;
  byte_50575E0 = 0;
  qword_5057570 = 38;
  LOBYTE(word_505754C) = word_505754C & 0x98 | 0x21;
  qword_5057568 = (__int64)"Enable generating trap for unreachable";
  sub_16B88A0(&qword_5057540);
  return __cxa_atexit(sub_12EDEC0, &qword_5057540, &qword_4A427C0);
}
