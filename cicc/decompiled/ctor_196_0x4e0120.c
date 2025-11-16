// Function: ctor_196
// Address: 0x4e0120
//
int ctor_196()
{
  int v0; // edx

  qword_4FAE1A0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAE1AC &= 0xF000u;
  qword_4FAE1B0 = 0;
  qword_4FAE1E8 = (__int64)qword_4FA01C0;
  qword_4FAE1B8 = 0;
  qword_4FAE1C0 = 0;
  qword_4FAE1C8 = 0;
  dword_4FAE1A8 = v0;
  qword_4FAE1F8 = (__int64)&unk_4FAE218;
  qword_4FAE200 = (__int64)&unk_4FAE218;
  qword_4FAE1D0 = 0;
  qword_4FAE1D8 = 0;
  qword_4FAE248 = (__int64)&unk_49E74E8;
  word_4FAE250 = 256;
  qword_4FAE1E0 = 0;
  qword_4FAE1F0 = 0;
  qword_4FAE1A0 = (__int64)&unk_49EEC70;
  qword_4FAE208 = 4;
  byte_4FAE238 = 0;
  qword_4FAE258 = (__int64)&unk_49EEDB0;
  dword_4FAE210 = 0;
  byte_4FAE240 = 0;
  sub_16B8280(&qword_4FAE1A0, "consthoist-with-block-frequency", 31);
  word_4FAE250 = 257;
  byte_4FAE240 = 1;
  qword_4FAE1D0 = 139;
  LOBYTE(word_4FAE1AC) = word_4FAE1AC & 0x9F | 0x20;
  qword_4FAE1C8 = (__int64)"Enable the use of the block frequency analysis to reduce the chance to execute const material"
                           "ization more frequently than without hoisting.";
  sub_16B88A0(&qword_4FAE1A0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FAE1A0, &qword_4A427C0);
}
