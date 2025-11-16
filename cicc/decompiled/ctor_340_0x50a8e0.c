// Function: ctor_340
// Address: 0x50a8e0
//
int ctor_340()
{
  int v0; // edx

  qword_4FCEA60 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCEA6C &= 0xF000u;
  qword_4FCEA70 = 0;
  qword_4FCEAA8 = (__int64)qword_4FA01C0;
  qword_4FCEA78 = 0;
  qword_4FCEA80 = 0;
  qword_4FCEA88 = 0;
  dword_4FCEA68 = v0;
  qword_4FCEAB8 = (__int64)&unk_4FCEAD8;
  qword_4FCEAC0 = (__int64)&unk_4FCEAD8;
  qword_4FCEA90 = 0;
  qword_4FCEA98 = 0;
  qword_4FCEB08 = (__int64)&unk_49E74E8;
  word_4FCEB10 = 256;
  qword_4FCEAA0 = 0;
  qword_4FCEAB0 = 0;
  qword_4FCEA60 = (__int64)&unk_49EEC70;
  qword_4FCEAC8 = 4;
  byte_4FCEAF8 = 0;
  qword_4FCEB18 = (__int64)&unk_49EEDB0;
  dword_4FCEAD0 = 0;
  byte_4FCEB00 = 0;
  sub_16B8280(&qword_4FCEA60, "fast-isel-sink-local-values", 27);
  word_4FCEB10 = 257;
  byte_4FCEB00 = 1;
  qword_4FCEA90 = 29;
  LOBYTE(word_4FCEA6C) = word_4FCEA6C & 0x9F | 0x20;
  qword_4FCEA88 = (__int64)"Sink local values in FastISel";
  sub_16B88A0(&qword_4FCEA60);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCEA60, &qword_4A427C0);
}
