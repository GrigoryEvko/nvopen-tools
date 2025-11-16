// Function: ctor_588
// Address: 0x57b3b0
//
int __fastcall ctor_588(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_50250C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5025110 = 0x100000000LL;
  dword_50250CC &= 0x8000u;
  word_50250D0 = 0;
  qword_50250D8 = 0;
  qword_50250E0 = 0;
  dword_50250C8 = v4;
  qword_50250E8 = 0;
  qword_50250F0 = 0;
  qword_50250F8 = 0;
  qword_5025100 = 0;
  qword_5025108 = (__int64)&unk_5025118;
  qword_5025120 = 0;
  qword_5025128 = (__int64)&unk_5025140;
  qword_5025130 = 1;
  dword_5025138 = 0;
  byte_502513C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5025110;
  v7 = (unsigned int)qword_5025110 + 1LL;
  if ( v7 > HIDWORD(qword_5025110) )
  {
    sub_C8D5F0((char *)&unk_5025118 - 16, &unk_5025118, v7, 8);
    v6 = (unsigned int)qword_5025110;
  }
  *(_QWORD *)(qword_5025108 + 8 * v6) = v5;
  qword_5025150 = (__int64)&unk_49D9748;
  LODWORD(qword_5025110) = qword_5025110 + 1;
  qword_5025148 = 0;
  qword_50250C0 = (__int64)&unk_49DC090;
  qword_5025160 = (__int64)&unk_49DC1D0;
  qword_5025158 = 0;
  qword_5025180 = (__int64)nullsub_23;
  qword_5025178 = (__int64)sub_984030;
  sub_C53080(&qword_50250C0, "safestack-use-pointer-address", 29);
  LOWORD(qword_5025158) = 256;
  LOBYTE(qword_5025148) = 0;
  LOBYTE(dword_50250CC) = dword_50250CC & 0x9F | 0x20;
  sub_C53130(&qword_50250C0);
  __cxa_atexit(sub_984900, &qword_50250C0, &qword_4A427C0);
  qword_5024FE0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_50250C0, v8, v9), 1u);
  qword_5025030 = 0x100000000LL;
  word_5024FF0 = 0;
  dword_5024FEC &= 0x8000u;
  qword_5024FF8 = 0;
  qword_5025000 = 0;
  dword_5024FE8 = v10;
  qword_5025008 = 0;
  qword_5025010 = 0;
  qword_5025018 = 0;
  qword_5025020 = 0;
  qword_5025028 = (__int64)&unk_5025038;
  qword_5025040 = 0;
  qword_5025048 = (__int64)&unk_5025060;
  qword_5025050 = 1;
  dword_5025058 = 0;
  byte_502505C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5025030;
  v13 = (unsigned int)qword_5025030 + 1LL;
  if ( v13 > HIDWORD(qword_5025030) )
  {
    sub_C8D5F0((char *)&unk_5025038 - 16, &unk_5025038, v13, 8);
    v12 = (unsigned int)qword_5025030;
  }
  *(_QWORD *)(qword_5025028 + 8 * v12) = v11;
  qword_5025070 = (__int64)&unk_49D9748;
  LODWORD(qword_5025030) = qword_5025030 + 1;
  qword_5025068 = 0;
  qword_5024FE0 = (__int64)&unk_49DC090;
  qword_5025080 = (__int64)&unk_49DC1D0;
  qword_5025078 = 0;
  qword_50250A0 = (__int64)nullsub_23;
  qword_5025098 = (__int64)sub_984030;
  sub_C53080(&qword_5024FE0, "safe-stack-coloring", 19);
  qword_5025010 = 26;
  qword_5025008 = (__int64)"enable safe stack coloring";
  LOBYTE(qword_5025068) = 1;
  LOBYTE(dword_5024FEC) = dword_5024FEC & 0x9F | 0x20;
  LOWORD(qword_5025078) = 257;
  sub_C53130(&qword_5024FE0);
  return __cxa_atexit(sub_984900, &qword_5024FE0, &qword_4A427C0);
}
