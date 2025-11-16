// Function: ctor_239
// Address: 0x4ec9a0
//
int ctor_239()
{
  int v0; // edx

  qword_4FB6A60 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB6A6C &= 0xF000u;
  qword_4FB6A70 = 0;
  qword_4FB6AA8 = (__int64)qword_4FA01C0;
  qword_4FB6A78 = 0;
  qword_4FB6A80 = 0;
  qword_4FB6A88 = 0;
  dword_4FB6A68 = v0;
  qword_4FB6AB8 = (__int64)&unk_4FB6AD8;
  qword_4FB6AC0 = (__int64)&unk_4FB6AD8;
  qword_4FB6A90 = 0;
  qword_4FB6A98 = 0;
  qword_4FB6B08 = (__int64)&unk_49E74E8;
  word_4FB6B10 = 256;
  qword_4FB6AA0 = 0;
  qword_4FB6AB0 = 0;
  qword_4FB6A60 = (__int64)&unk_49EEC70;
  qword_4FB6AC8 = 4;
  byte_4FB6AF8 = 0;
  qword_4FB6B18 = (__int64)&unk_49EEDB0;
  dword_4FB6AD0 = 0;
  byte_4FB6B00 = 0;
  sub_16B8280(&qword_4FB6A60, "unroll-runtime-multi-exit", 25);
  word_4FB6B10 = 256;
  byte_4FB6B00 = 0;
  qword_4FB6A90 = 79;
  LOBYTE(word_4FB6A6C) = word_4FB6A6C & 0x9F | 0x20;
  qword_4FB6A88 = (__int64)"Allow runtime unrolling for loops with multiple exits, when epilog is generated";
  sub_16B88A0(&qword_4FB6A60);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB6A60, &qword_4A427C0);
}
