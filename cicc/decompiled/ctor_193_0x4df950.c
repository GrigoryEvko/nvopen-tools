// Function: ctor_193
// Address: 0x4df950
//
int ctor_193()
{
  int v0; // eax
  int v1; // eax

  qword_4FADD80 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FADD8C &= 0xF000u;
  qword_4FADDC8 = (__int64)qword_4FA01C0;
  qword_4FADD90 = 0;
  qword_4FADD98 = 0;
  qword_4FADDA0 = 0;
  dword_4FADD88 = v0;
  qword_4FADDD8 = (__int64)&unk_4FADDF8;
  qword_4FADDE0 = (__int64)&unk_4FADDF8;
  qword_4FADE20 = (__int64)&byte_4FADE30;
  qword_4FADE48 = (__int64)&byte_4FADE58;
  qword_4FADDA8 = 0;
  qword_4FADDB0 = 0;
  qword_4FADE40 = (__int64)&unk_49EED10;
  qword_4FADDB8 = 0;
  qword_4FADDC0 = 0;
  qword_4FADD80 = (__int64)&unk_49EEBF0;
  qword_4FADDD0 = 0;
  byte_4FADE18 = 0;
  qword_4FADE70 = (__int64)&unk_49EEE90;
  qword_4FADE78 = (__int64)&byte_4FADE88;
  qword_4FADDE8 = 4;
  dword_4FADDF0 = 0;
  qword_4FADE28 = 0;
  byte_4FADE30 = 0;
  qword_4FADE50 = 0;
  byte_4FADE58 = 0;
  byte_4FADE68 = 0;
  qword_4FADE80 = 0;
  byte_4FADE88 = 0;
  sub_16B8280(&qword_4FADD80, "extract-blocks-file", 19);
  qword_4FADDC0 = 8;
  qword_4FADDB8 = (__int64)"filename";
  qword_4FADDA8 = (__int64)"A file containing list of basic blocks to extract";
  qword_4FADDB0 = 49;
  LOBYTE(word_4FADD8C) = word_4FADD8C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FADD80);
  __cxa_atexit(sub_12F0C20, &qword_4FADD80, &qword_4A427C0);
  qword_4FADCA0[0] = &unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FADCA0[1]) &= 0xF000u;
  LODWORD(qword_4FADCA0[1]) = v1;
  qword_4FADCA0[11] = &qword_4FADCA0[15];
  qword_4FADCA0[12] = &qword_4FADCA0[15];
  qword_4FADCA0[9] = qword_4FA01C0;
  qword_4FADCA0[2] = 0;
  qword_4FADCA0[21] = &unk_49E74E8;
  LOWORD(qword_4FADCA0[22]) = 256;
  qword_4FADCA0[3] = 0;
  qword_4FADCA0[4] = 0;
  qword_4FADCA0[0] = &unk_49EEC70;
  qword_4FADCA0[5] = 0;
  qword_4FADCA0[6] = 0;
  qword_4FADCA0[23] = &unk_49EEDB0;
  qword_4FADCA0[7] = 0;
  qword_4FADCA0[8] = 0;
  qword_4FADCA0[10] = 0;
  qword_4FADCA0[13] = 4;
  LODWORD(qword_4FADCA0[14]) = 0;
  LOBYTE(qword_4FADCA0[19]) = 0;
  LOBYTE(qword_4FADCA0[20]) = 0;
  sub_16B8280(qword_4FADCA0, "extract-blocks-erase-funcs", 26);
  qword_4FADCA0[6] = 28;
  qword_4FADCA0[5] = "Erase the existing functions";
  BYTE4(qword_4FADCA0[1]) = BYTE4(qword_4FADCA0[1]) & 0x9F | 0x20;
  sub_16B88A0(qword_4FADCA0);
  return __cxa_atexit(sub_12EDEC0, qword_4FADCA0, &qword_4A427C0);
}
