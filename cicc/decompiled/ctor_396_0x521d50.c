// Function: ctor_396
// Address: 0x521d50
//
int ctor_396()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  _QWORD v9[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v10[6]; // [rsp+10h] [rbp-30h] BYREF

  qword_4FE3BA0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FE3C1C = 1;
  qword_4FE3BF0 = 0x100000000LL;
  dword_4FE3BAC &= 0x8000u;
  qword_4FE3BB8 = 0;
  qword_4FE3BC0 = 0;
  qword_4FE3BC8 = 0;
  dword_4FE3BA8 = v0;
  word_4FE3BB0 = 0;
  qword_4FE3BD0 = 0;
  qword_4FE3BD8 = 0;
  qword_4FE3BE0 = 0;
  qword_4FE3BE8 = (__int64)&unk_4FE3BF8;
  qword_4FE3C00 = 0;
  qword_4FE3C08 = (__int64)&unk_4FE3C20;
  qword_4FE3C10 = 1;
  dword_4FE3C18 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FE3BF0;
  v3 = (unsigned int)qword_4FE3BF0 + 1LL;
  if ( v3 > HIDWORD(qword_4FE3BF0) )
  {
    sub_C8D5F0((char *)&unk_4FE3BF8 - 16, &unk_4FE3BF8, v3, 8);
    v2 = (unsigned int)qword_4FE3BF0;
  }
  *(_QWORD *)(qword_4FE3BE8 + 8 * v2) = v1;
  qword_4FE3C28 = (__int64)&byte_4FE3C38;
  qword_4FE3C50 = (__int64)&byte_4FE3C60;
  LODWORD(qword_4FE3BF0) = qword_4FE3BF0 + 1;
  qword_4FE3C30 = 0;
  qword_4FE3C48 = (__int64)&unk_49DC130;
  byte_4FE3C38 = 0;
  byte_4FE3C60 = 0;
  qword_4FE3BA0 = (__int64)&unk_49DC010;
  qword_4FE3C58 = 0;
  byte_4FE3C70 = 0;
  qword_4FE3C78 = (__int64)&unk_49DC350;
  qword_4FE3C98 = (__int64)nullsub_92;
  qword_4FE3C90 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FE3BA0, "default-gcov-version", 20);
  v9[0] = v10;
  sub_2425560(v9, "0000");
  sub_2240AE0(&qword_4FE3C28, v9);
  byte_4FE3C70 = 1;
  sub_2240AE0(&qword_4FE3C50, v9);
  if ( (_QWORD *)v9[0] != v10 )
    j_j___libc_free_0(v9[0], v10[0] + 1LL);
  LOBYTE(dword_4FE3BAC) = dword_4FE3BAC & 0x87 | 0x30;
  sub_C53130(&qword_4FE3BA0);
  __cxa_atexit(sub_BC5A40, &qword_4FE3BA0, &qword_4A427C0);
  qword_4FE3AC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FE3ACC &= 0x8000u;
  word_4FE3AD0 = 0;
  qword_4FE3B10 = 0x100000000LL;
  qword_4FE3AD8 = 0;
  qword_4FE3AE0 = 0;
  qword_4FE3AE8 = 0;
  dword_4FE3AC8 = v4;
  qword_4FE3AF0 = 0;
  qword_4FE3AF8 = 0;
  qword_4FE3B00 = 0;
  qword_4FE3B08 = (__int64)&unk_4FE3B18;
  qword_4FE3B20 = 0;
  qword_4FE3B28 = (__int64)&unk_4FE3B40;
  qword_4FE3B30 = 1;
  dword_4FE3B38 = 0;
  byte_4FE3B3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FE3B10;
  v7 = (unsigned int)qword_4FE3B10 + 1LL;
  if ( v7 > HIDWORD(qword_4FE3B10) )
  {
    sub_C8D5F0((char *)&unk_4FE3B18 - 16, &unk_4FE3B18, v7, 8);
    v6 = (unsigned int)qword_4FE3B10;
  }
  *(_QWORD *)(qword_4FE3B08 + 8 * v6) = v5;
  LODWORD(qword_4FE3B10) = qword_4FE3B10 + 1;
  qword_4FE3B48 = 0;
  qword_4FE3B50 = (__int64)&unk_49D9748;
  qword_4FE3B58 = 0;
  qword_4FE3AC0 = (__int64)&unk_49DC090;
  qword_4FE3B60 = (__int64)&unk_49DC1D0;
  qword_4FE3B80 = (__int64)nullsub_23;
  qword_4FE3B78 = (__int64)sub_984030;
  sub_C53080(&qword_4FE3AC0, "gcov-atomic-counter", 19);
  qword_4FE3AF0 = 27;
  LOBYTE(dword_4FE3ACC) = dword_4FE3ACC & 0x9F | 0x20;
  qword_4FE3AE8 = (__int64)"Make counter updates atomic";
  sub_C53130(&qword_4FE3AC0);
  return __cxa_atexit(sub_984900, &qword_4FE3AC0, &qword_4A427C0);
}
