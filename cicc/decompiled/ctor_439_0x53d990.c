// Function: ctor_439
// Address: 0x53d990
//
int ctor_439()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_4FF9C00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9C50 = 0x100000000LL;
  dword_4FF9C0C &= 0x8000u;
  word_4FF9C10 = 0;
  qword_4FF9C18 = 0;
  qword_4FF9C20 = 0;
  dword_4FF9C08 = v0;
  qword_4FF9C28 = 0;
  qword_4FF9C30 = 0;
  qword_4FF9C38 = 0;
  qword_4FF9C40 = 0;
  qword_4FF9C48 = (__int64)&unk_4FF9C58;
  qword_4FF9C60 = 0;
  qword_4FF9C68 = (__int64)&unk_4FF9C80;
  qword_4FF9C70 = 1;
  dword_4FF9C78 = 0;
  byte_4FF9C7C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF9C50;
  v3 = (unsigned int)qword_4FF9C50 + 1LL;
  if ( v3 > HIDWORD(qword_4FF9C50) )
  {
    sub_C8D5F0((char *)&unk_4FF9C58 - 16, &unk_4FF9C58, v3, 8);
    v2 = (unsigned int)qword_4FF9C50;
  }
  *(_QWORD *)(qword_4FF9C48 + 8 * v2) = v1;
  qword_4FF9C90 = (__int64)&unk_49D9748;
  LODWORD(qword_4FF9C50) = qword_4FF9C50 + 1;
  qword_4FF9C88 = 0;
  qword_4FF9C00 = (__int64)&unk_49DC090;
  qword_4FF9CA0 = (__int64)&unk_49DC1D0;
  qword_4FF9C98 = 0;
  qword_4FF9CC0 = (__int64)nullsub_23;
  qword_4FF9CB8 = (__int64)sub_984030;
  sub_C53080(&qword_4FF9C00, "adce-remove-control-flow", 24);
  LOWORD(qword_4FF9C98) = 257;
  LOBYTE(qword_4FF9C88) = 1;
  LOBYTE(dword_4FF9C0C) = dword_4FF9C0C & 0x9F | 0x20;
  sub_C53130(&qword_4FF9C00);
  __cxa_atexit(sub_984900, &qword_4FF9C00, &qword_4A427C0);
  qword_4FF9B20 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9B70 = 0x100000000LL;
  word_4FF9B30 = 0;
  dword_4FF9B2C &= 0x8000u;
  qword_4FF9B38 = 0;
  qword_4FF9B40 = 0;
  dword_4FF9B28 = v4;
  qword_4FF9B48 = 0;
  qword_4FF9B50 = 0;
  qword_4FF9B58 = 0;
  qword_4FF9B60 = 0;
  qword_4FF9B68 = (__int64)&unk_4FF9B78;
  qword_4FF9B80 = 0;
  qword_4FF9B88 = (__int64)&unk_4FF9BA0;
  qword_4FF9B90 = 1;
  dword_4FF9B98 = 0;
  byte_4FF9B9C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF9B70;
  v7 = (unsigned int)qword_4FF9B70 + 1LL;
  if ( v7 > HIDWORD(qword_4FF9B70) )
  {
    sub_C8D5F0((char *)&unk_4FF9B78 - 16, &unk_4FF9B78, v7, 8);
    v6 = (unsigned int)qword_4FF9B70;
  }
  *(_QWORD *)(qword_4FF9B68 + 8 * v6) = v5;
  qword_4FF9BB0 = (__int64)&unk_49D9748;
  LODWORD(qword_4FF9B70) = qword_4FF9B70 + 1;
  qword_4FF9BA8 = 0;
  qword_4FF9B20 = (__int64)&unk_49DC090;
  qword_4FF9BC0 = (__int64)&unk_49DC1D0;
  qword_4FF9BB8 = 0;
  qword_4FF9BE0 = (__int64)nullsub_23;
  qword_4FF9BD8 = (__int64)sub_984030;
  sub_C53080(&qword_4FF9B20, "adce-remove-loops", 17);
  LOBYTE(qword_4FF9BA8) = 0;
  LOWORD(qword_4FF9BB8) = 256;
  LOBYTE(dword_4FF9B2C) = dword_4FF9B2C & 0x9F | 0x20;
  sub_C53130(&qword_4FF9B20);
  return __cxa_atexit(sub_984900, &qword_4FF9B20, &qword_4A427C0);
}
