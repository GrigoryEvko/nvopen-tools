// Function: ctor_142
// Address: 0x4b3cf0
//
int ctor_142()
{
  int v0; // eax
  int v1; // eax

  qword_4F9DEA0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9DEAC &= 0xF000u;
  qword_4F9DEE8 = (__int64)&unk_4FA01C0;
  qword_4F9DEB0 = 0;
  qword_4F9DEB8 = 0;
  qword_4F9DEC0 = 0;
  dword_4F9DEA8 = v0;
  qword_4F9DEF8 = (__int64)&unk_4F9DF18;
  qword_4F9DF00 = (__int64)&unk_4F9DF18;
  qword_4F9DEC8 = 0;
  qword_4F9DED0 = 0;
  qword_4F9DF48 = (__int64)&unk_49E74A8;
  qword_4F9DED8 = 0;
  qword_4F9DEE0 = 0;
  qword_4F9DEA0 = (__int64)&unk_49EEAF0;
  qword_4F9DEF0 = 0;
  byte_4F9DF38 = 0;
  qword_4F9DF58 = (__int64)&unk_49EEE10;
  qword_4F9DF08 = 4;
  dword_4F9DF10 = 0;
  dword_4F9DF40 = 0;
  byte_4F9DF54 = 1;
  dword_4F9DF50 = 0;
  sub_16B8280(&qword_4F9DEA0, "bitcode-mdindex-threshold", 25);
  dword_4F9DF40 = 25;
  byte_4F9DF54 = 1;
  dword_4F9DF50 = 25;
  qword_4F9DED0 = 71;
  LOBYTE(word_4F9DEAC) = word_4F9DEAC & 0x9F | 0x20;
  qword_4F9DEC8 = (__int64)"Number of metadatas above which we emit an index to enable lazy-loading";
  sub_16B88A0(&qword_4F9DEA0);
  __cxa_atexit(sub_12EDE60, &qword_4F9DEA0, &qword_4A427C0);
  qword_4F9DDC0[0] = &unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4F9DDC0[1]) &= 0xF000u;
  LODWORD(qword_4F9DDC0[1]) = v1;
  qword_4F9DDC0[11] = &qword_4F9DDC0[15];
  qword_4F9DDC0[12] = &qword_4F9DDC0[15];
  qword_4F9DDC0[9] = &unk_4FA01C0;
  qword_4F9DDC0[2] = 0;
  qword_4F9DDC0[21] = &unk_49E74E8;
  LOWORD(qword_4F9DDC0[22]) = 256;
  qword_4F9DDC0[3] = 0;
  qword_4F9DDC0[4] = 0;
  qword_4F9DDC0[0] = &unk_49EEC70;
  qword_4F9DDC0[5] = 0;
  qword_4F9DDC0[6] = 0;
  qword_4F9DDC0[23] = &unk_49EEDB0;
  qword_4F9DDC0[7] = 0;
  qword_4F9DDC0[8] = 0;
  qword_4F9DDC0[10] = 0;
  qword_4F9DDC0[13] = 4;
  LODWORD(qword_4F9DDC0[14]) = 0;
  LOBYTE(qword_4F9DDC0[19]) = 0;
  LOBYTE(qword_4F9DDC0[20]) = 0;
  sub_16B8280(qword_4F9DDC0, "write-relbf-to-summary", 22);
  LOWORD(qword_4F9DDC0[22]) = 256;
  LOBYTE(qword_4F9DDC0[20]) = 0;
  qword_4F9DDC0[6] = 51;
  BYTE4(qword_4F9DDC0[1]) = BYTE4(qword_4F9DDC0[1]) & 0x9F | 0x20;
  qword_4F9DDC0[5] = "Write relative block frequency to function summary ";
  sub_16B88A0(qword_4F9DDC0);
  return __cxa_atexit(sub_12EDEC0, qword_4F9DDC0, &qword_4A427C0);
}
