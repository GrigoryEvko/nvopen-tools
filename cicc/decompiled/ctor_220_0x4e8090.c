// Function: ctor_220
// Address: 0x4e8090
//
int ctor_220()
{
  int v0; // eax
  int v1; // eax

  dword_4FB3CA8 = sub_19EC580("newgvn-vn", 9, "Controls which instructions are value numbered", 46);
  sub_19EC580("newgvn-phi", 10, "Controls which instructions we create phi of ops for", 52);
  qword_4FB3BE0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB3BEC &= 0xF000u;
  qword_4FB3BF0 = 0;
  qword_4FB3BF8 = 0;
  qword_4FB3C00 = 0;
  qword_4FB3C08 = 0;
  qword_4FB3C10 = 0;
  dword_4FB3BE8 = v0;
  qword_4FB3C18 = 0;
  qword_4FB3C28 = (__int64)qword_4FA01C0;
  qword_4FB3C38 = (__int64)&unk_4FB3C58;
  qword_4FB3C40 = (__int64)&unk_4FB3C58;
  qword_4FB3C20 = 0;
  qword_4FB3C30 = 0;
  word_4FB3C90 = 256;
  qword_4FB3C88 = (__int64)&unk_49E74E8;
  qword_4FB3C48 = 4;
  qword_4FB3BE0 = (__int64)&unk_49EEC70;
  byte_4FB3C78 = 0;
  qword_4FB3C98 = (__int64)&unk_49EEDB0;
  dword_4FB3C50 = 0;
  byte_4FB3C80 = 0;
  sub_16B8280(&qword_4FB3BE0, "enable-store-refinement", 23);
  word_4FB3C90 = 256;
  byte_4FB3C80 = 0;
  LOBYTE(word_4FB3BEC) = word_4FB3BEC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB3BE0);
  __cxa_atexit(sub_12EDEC0, &qword_4FB3BE0, &qword_4A427C0);
  qword_4FB3B00 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB3BB0 = 256;
  word_4FB3B0C &= 0xF000u;
  qword_4FB3B10 = 0;
  qword_4FB3B18 = 0;
  qword_4FB3B20 = 0;
  dword_4FB3B08 = v1;
  qword_4FB3BA8 = (__int64)&unk_49E74E8;
  qword_4FB3B48 = (__int64)qword_4FA01C0;
  qword_4FB3B58 = (__int64)&unk_4FB3B78;
  qword_4FB3B60 = (__int64)&unk_4FB3B78;
  qword_4FB3B00 = (__int64)&unk_49EEC70;
  qword_4FB3BB8 = (__int64)&unk_49EEDB0;
  qword_4FB3B28 = 0;
  qword_4FB3B30 = 0;
  qword_4FB3B38 = 0;
  qword_4FB3B40 = 0;
  qword_4FB3B50 = 0;
  qword_4FB3B68 = 4;
  dword_4FB3B70 = 0;
  byte_4FB3B98 = 0;
  byte_4FB3BA0 = 0;
  sub_16B8280(&qword_4FB3B00, "enable-phi-of-ops", 17);
  word_4FB3BB0 = 257;
  byte_4FB3BA0 = 1;
  LOBYTE(word_4FB3B0C) = word_4FB3B0C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB3B00);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB3B00, &qword_4A427C0);
}
