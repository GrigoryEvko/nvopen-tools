// Function: ctor_170
// Address: 0x4d1cd0
//
int ctor_170()
{
  int v0; // eax
  int v1; // eax
  _QWORD v3[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v4[8]; // [rsp+10h] [rbp-40h] BYREF

  qword_4FA2AA0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA2AAC &= 0xF000u;
  qword_4FA2AB0 = 0;
  qword_4FA2AB8 = 0;
  qword_4FA2AC0 = 0;
  qword_4FA2AC8 = 0;
  dword_4FA2AA8 = v0;
  qword_4FA2AF8 = (__int64)&unk_4FA2B18;
  qword_4FA2B00 = (__int64)&unk_4FA2B18;
  qword_4FA2B40 = (__int64)&byte_4FA2B50;
  qword_4FA2B68 = (__int64)&byte_4FA2B78;
  qword_4FA2AD0 = 0;
  qword_4FA2AE8 = (__int64)qword_4FA01C0;
  qword_4FA2B60 = (__int64)&unk_49EED10;
  qword_4FA2AD8 = 0;
  qword_4FA2AE0 = 0;
  qword_4FA2AA0 = (__int64)&unk_49EEBF0;
  qword_4FA2AF0 = 0;
  byte_4FA2B38 = 0;
  qword_4FA2B90 = (__int64)&unk_49EEE90;
  qword_4FA2B98 = (__int64)&byte_4FA2BA8;
  qword_4FA2B08 = 4;
  dword_4FA2B10 = 0;
  qword_4FA2B48 = 0;
  byte_4FA2B50 = 0;
  qword_4FA2B70 = 0;
  byte_4FA2B78 = 0;
  byte_4FA2B88 = 0;
  qword_4FA2BA0 = 0;
  byte_4FA2BA8 = 0;
  sub_16B8280(&byte_4FA2BA8 - 264, "default-gcov-version", 20);
  v3[0] = v4;
  sub_17B71F0(v3, "402*");
  sub_2240AE0(&qword_4FA2B40, v3);
  byte_4FA2B88 = 1;
  sub_2240AE0(&qword_4FA2B68, v3);
  if ( (_QWORD *)v3[0] != v4 )
    j_j___libc_free_0(v3[0], v4[0] + 1LL);
  LOBYTE(word_4FA2AAC) = word_4FA2AAC & 0x87 | 0x30;
  sub_16B88A0(&qword_4FA2AA0);
  __cxa_atexit(sub_12F0C20, &qword_4FA2AA0, &qword_4A427C0);
  qword_4FA29C0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA29CC &= 0xF000u;
  qword_4FA29D0 = 0;
  qword_4FA29D8 = 0;
  qword_4FA29E0 = 0;
  qword_4FA29E8 = 0;
  dword_4FA29C8 = v1;
  qword_4FA2A18 = (__int64)&unk_4FA2A38;
  qword_4FA2A20 = (__int64)&unk_4FA2A38;
  qword_4FA2A08 = (__int64)qword_4FA01C0;
  qword_4FA29F0 = 0;
  qword_4FA2A68 = (__int64)&unk_49E74E8;
  word_4FA2A70 = 256;
  qword_4FA29F8 = 0;
  qword_4FA2A00 = 0;
  qword_4FA29C0 = (__int64)&unk_49EEC70;
  qword_4FA2A10 = 0;
  byte_4FA2A58 = 0;
  qword_4FA2A78 = (__int64)&unk_49EEDB0;
  qword_4FA2A28 = 4;
  dword_4FA2A30 = 0;
  byte_4FA2A60 = 0;
  sub_16B8280(&qword_4FA29C0, "gcov-exit-block-before-body", 27);
  word_4FA2A70 = 256;
  byte_4FA2A60 = 0;
  LOBYTE(word_4FA29CC) = word_4FA29CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA29C0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FA29C0, &qword_4A427C0);
}
