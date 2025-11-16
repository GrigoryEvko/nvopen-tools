// Function: ctor_520
// Address: 0x562990
//
int ctor_520()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_5010C00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5010C50 = 0x100000000LL;
  dword_5010C0C &= 0x8000u;
  word_5010C10 = 0;
  qword_5010C18 = 0;
  qword_5010C20 = 0;
  dword_5010C08 = v0;
  qword_5010C28 = 0;
  qword_5010C30 = 0;
  qword_5010C38 = 0;
  qword_5010C40 = 0;
  qword_5010C48 = (__int64)&unk_5010C58;
  qword_5010C60 = 0;
  qword_5010C68 = (__int64)&unk_5010C80;
  qword_5010C70 = 1;
  dword_5010C78 = 0;
  byte_5010C7C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5010C50;
  v3 = (unsigned int)qword_5010C50 + 1LL;
  if ( v3 > HIDWORD(qword_5010C50) )
  {
    sub_C8D5F0((char *)&unk_5010C58 - 16, &unk_5010C58, v3, 8);
    v2 = (unsigned int)qword_5010C50;
  }
  *(_QWORD *)(qword_5010C48 + 8 * v2) = v1;
  qword_5010C90 = (__int64)&unk_49D9748;
  qword_5010C00 = (__int64)&unk_49DC090;
  LODWORD(qword_5010C50) = qword_5010C50 + 1;
  qword_5010C88 = 0;
  qword_5010CA0 = (__int64)&unk_49DC1D0;
  qword_5010C98 = 0;
  qword_5010CC0 = (__int64)nullsub_23;
  qword_5010CB8 = (__int64)sub_984030;
  sub_C53080(&qword_5010C00, "disable-vector-combine", 22);
  LOWORD(qword_5010C98) = 256;
  LOBYTE(qword_5010C88) = 0;
  qword_5010C30 = 37;
  LOBYTE(dword_5010C0C) = dword_5010C0C & 0x9F | 0x20;
  qword_5010C28 = (__int64)"Disable all vector combine transforms";
  sub_C53130(&qword_5010C00);
  __cxa_atexit(sub_984900, &qword_5010C00, &qword_4A427C0);
  qword_5010B20 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5010B70 = 0x100000000LL;
  dword_5010B2C &= 0x8000u;
  word_5010B30 = 0;
  qword_5010B38 = 0;
  qword_5010B40 = 0;
  dword_5010B28 = v4;
  qword_5010B48 = 0;
  qword_5010B50 = 0;
  qword_5010B58 = 0;
  qword_5010B60 = 0;
  qword_5010B68 = (__int64)&unk_5010B78;
  qword_5010B80 = 0;
  qword_5010B88 = (__int64)&unk_5010BA0;
  qword_5010B90 = 1;
  dword_5010B98 = 0;
  byte_5010B9C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5010B70;
  if ( (unsigned __int64)(unsigned int)qword_5010B70 + 1 > HIDWORD(qword_5010B70) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_5010B78 - 16, &unk_5010B78, (unsigned int)qword_5010B70 + 1LL, 8);
    v6 = (unsigned int)qword_5010B70;
    v5 = v12;
  }
  *(_QWORD *)(qword_5010B68 + 8 * v6) = v5;
  qword_5010BB0 = (__int64)&unk_49D9748;
  qword_5010B20 = (__int64)&unk_49DC090;
  LODWORD(qword_5010B70) = qword_5010B70 + 1;
  qword_5010BA8 = 0;
  qword_5010BC0 = (__int64)&unk_49DC1D0;
  qword_5010BB8 = 0;
  qword_5010BE0 = (__int64)nullsub_23;
  qword_5010BD8 = (__int64)sub_984030;
  sub_C53080(&qword_5010B20, "disable-binop-extract-shuffle", 29);
  LOBYTE(qword_5010BA8) = 0;
  LOWORD(qword_5010BB8) = 256;
  qword_5010B50 = 43;
  LOBYTE(dword_5010B2C) = dword_5010B2C & 0x9F | 0x20;
  qword_5010B48 = (__int64)"Disable binop extract to shuffle transforms";
  sub_C53130(&qword_5010B20);
  __cxa_atexit(sub_984900, &qword_5010B20, &qword_4A427C0);
  qword_5010A40 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5010ABC = 1;
  qword_5010A90 = 0x100000000LL;
  dword_5010A4C &= 0x8000u;
  qword_5010A58 = 0;
  qword_5010A60 = 0;
  qword_5010A68 = 0;
  dword_5010A48 = v7;
  word_5010A50 = 0;
  qword_5010A70 = 0;
  qword_5010A78 = 0;
  qword_5010A80 = 0;
  qword_5010A88 = (__int64)&unk_5010A98;
  qword_5010AA0 = 0;
  qword_5010AA8 = (__int64)&unk_5010AC0;
  qword_5010AB0 = 1;
  dword_5010AB8 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_5010A90;
  v10 = (unsigned int)qword_5010A90 + 1LL;
  if ( v10 > HIDWORD(qword_5010A90) )
  {
    sub_C8D5F0((char *)&unk_5010A98 - 16, &unk_5010A98, v10, 8);
    v9 = (unsigned int)qword_5010A90;
  }
  *(_QWORD *)(qword_5010A88 + 8 * v9) = v8;
  LODWORD(qword_5010A90) = qword_5010A90 + 1;
  qword_5010AC8 = 0;
  qword_5010AD0 = (__int64)&unk_49D9728;
  qword_5010AD8 = 0;
  qword_5010A40 = (__int64)&unk_49DBF10;
  qword_5010AE0 = (__int64)&unk_49DC290;
  qword_5010B00 = (__int64)nullsub_24;
  qword_5010AF8 = (__int64)sub_984050;
  sub_C53080(&qword_5010A40, "vector-combine-max-scan-instrs", 30);
  LODWORD(qword_5010AC8) = 30;
  BYTE4(qword_5010AD8) = 1;
  LODWORD(qword_5010AD8) = 30;
  qword_5010A70 = 56;
  LOBYTE(dword_5010A4C) = dword_5010A4C & 0x9F | 0x20;
  qword_5010A68 = (__int64)"Max number of instructions to scan for vector combining.";
  sub_C53130(&qword_5010A40);
  return __cxa_atexit(sub_984970, &qword_5010A40, &qword_4A427C0);
}
