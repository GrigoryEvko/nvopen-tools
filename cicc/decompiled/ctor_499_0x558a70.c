// Function: ctor_499
// Address: 0x558a70
//
int ctor_499()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+8h] [rbp-38h]
  __int64 v24; // [rsp+8h] [rbp-38h]

  qword_5009E20 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5009E70 = 0x100000000LL;
  dword_5009E2C &= 0x8000u;
  word_5009E30 = 0;
  qword_5009E38 = 0;
  qword_5009E40 = 0;
  dword_5009E28 = v0;
  qword_5009E48 = 0;
  qword_5009E50 = 0;
  qword_5009E58 = 0;
  qword_5009E60 = 0;
  qword_5009E68 = (__int64)&unk_5009E78;
  qword_5009E80 = 0;
  qword_5009E88 = (__int64)&unk_5009EA0;
  qword_5009E90 = 1;
  dword_5009E98 = 0;
  byte_5009E9C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5009E70;
  v3 = (unsigned int)qword_5009E70 + 1LL;
  if ( v3 > HIDWORD(qword_5009E70) )
  {
    sub_C8D5F0((char *)&unk_5009E78 - 16, &unk_5009E78, v3, 8);
    v2 = (unsigned int)qword_5009E70;
  }
  *(_QWORD *)(qword_5009E68 + 8 * v2) = v1;
  qword_5009EB0 = (__int64)&unk_49D9728;
  qword_5009E20 = (__int64)&unk_49DBF10;
  LODWORD(qword_5009E70) = qword_5009E70 + 1;
  qword_5009EA8 = 0;
  qword_5009EC0 = (__int64)&unk_49DC290;
  qword_5009EB8 = 0;
  qword_5009EE0 = (__int64)nullsub_24;
  qword_5009ED8 = (__int64)sub_984050;
  sub_C53080(&qword_5009E20, "unroll-peel-count", 17);
  qword_5009E50 = 50;
  LOBYTE(dword_5009E2C) = dword_5009E2C & 0x9F | 0x20;
  qword_5009E48 = (__int64)"Set the unroll peeling count, for testing purposes";
  sub_C53130(&qword_5009E20);
  __cxa_atexit(sub_984970, &qword_5009E20, &qword_4A427C0);
  qword_5009D40 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5009D90 = 0x100000000LL;
  dword_5009D4C &= 0x8000u;
  qword_5009D88 = (__int64)&unk_5009D98;
  word_5009D50 = 0;
  qword_5009D58 = 0;
  dword_5009D48 = v4;
  qword_5009D60 = 0;
  qword_5009D68 = 0;
  qword_5009D70 = 0;
  qword_5009D78 = 0;
  qword_5009D80 = 0;
  qword_5009DA0 = 0;
  qword_5009DA8 = (__int64)&unk_5009DC0;
  qword_5009DB0 = 1;
  dword_5009DB8 = 0;
  byte_5009DBC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5009D90;
  v7 = (unsigned int)qword_5009D90 + 1LL;
  if ( v7 > HIDWORD(qword_5009D90) )
  {
    sub_C8D5F0((char *)&unk_5009D98 - 16, &unk_5009D98, v7, 8);
    v6 = (unsigned int)qword_5009D90;
  }
  *(_QWORD *)(qword_5009D88 + 8 * v6) = v5;
  LODWORD(qword_5009D90) = qword_5009D90 + 1;
  qword_5009DC8 = 0;
  qword_5009DD0 = (__int64)&unk_49D9748;
  qword_5009DD8 = 0;
  qword_5009D40 = (__int64)&unk_49DC090;
  qword_5009DE0 = (__int64)&unk_49DC1D0;
  qword_5009E00 = (__int64)nullsub_23;
  qword_5009DF8 = (__int64)sub_984030;
  sub_C53080(&qword_5009D40, "unroll-allow-peeling", 20);
  LOWORD(qword_5009DD8) = 257;
  LOBYTE(qword_5009DC8) = 1;
  qword_5009D70 = 73;
  LOBYTE(dword_5009D4C) = dword_5009D4C & 0x9F | 0x20;
  qword_5009D68 = (__int64)"Allows loops to be peeled when the dynamic trip count is known to be low.";
  sub_C53130(&qword_5009D40);
  __cxa_atexit(sub_984900, &qword_5009D40, &qword_4A427C0);
  qword_5009C60 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5009CB0 = 0x100000000LL;
  dword_5009C6C &= 0x8000u;
  qword_5009CA8 = (__int64)&unk_5009CB8;
  word_5009C70 = 0;
  qword_5009C78 = 0;
  dword_5009C68 = v8;
  qword_5009C80 = 0;
  qword_5009C88 = 0;
  qword_5009C90 = 0;
  qword_5009C98 = 0;
  qword_5009CA0 = 0;
  qword_5009CC0 = 0;
  qword_5009CC8 = (__int64)&unk_5009CE0;
  qword_5009CD0 = 1;
  dword_5009CD8 = 0;
  byte_5009CDC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_5009CB0;
  if ( (unsigned __int64)(unsigned int)qword_5009CB0 + 1 > HIDWORD(qword_5009CB0) )
  {
    v22 = v9;
    sub_C8D5F0((char *)&unk_5009CB8 - 16, &unk_5009CB8, (unsigned int)qword_5009CB0 + 1LL, 8);
    v10 = (unsigned int)qword_5009CB0;
    v9 = v22;
  }
  *(_QWORD *)(qword_5009CA8 + 8 * v10) = v9;
  LODWORD(qword_5009CB0) = qword_5009CB0 + 1;
  qword_5009CE8 = 0;
  qword_5009CF0 = (__int64)&unk_49D9748;
  qword_5009CF8 = 0;
  qword_5009C60 = (__int64)&unk_49DC090;
  qword_5009D00 = (__int64)&unk_49DC1D0;
  qword_5009D20 = (__int64)nullsub_23;
  qword_5009D18 = (__int64)sub_984030;
  sub_C53080(&qword_5009C60, "unroll-allow-loop-nests-peeling", 31);
  LOWORD(qword_5009CF8) = 256;
  LOBYTE(qword_5009CE8) = 0;
  qword_5009C90 = 31;
  LOBYTE(dword_5009C6C) = dword_5009C6C & 0x9F | 0x20;
  qword_5009C88 = (__int64)"Allows loop nests to be peeled.";
  sub_C53130(&qword_5009C60);
  __cxa_atexit(sub_984900, &qword_5009C60, &qword_4A427C0);
  qword_5009B80 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5009BD0 = 0x100000000LL;
  dword_5009B8C &= 0x8000u;
  qword_5009BC8 = (__int64)&unk_5009BD8;
  word_5009B90 = 0;
  qword_5009B98 = 0;
  dword_5009B88 = v11;
  qword_5009BA0 = 0;
  qword_5009BA8 = 0;
  qword_5009BB0 = 0;
  qword_5009BB8 = 0;
  qword_5009BC0 = 0;
  qword_5009BE0 = 0;
  qword_5009BE8 = (__int64)&unk_5009C00;
  qword_5009BF0 = 1;
  dword_5009BF8 = 0;
  byte_5009BFC = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_5009BD0;
  if ( (unsigned __int64)(unsigned int)qword_5009BD0 + 1 > HIDWORD(qword_5009BD0) )
  {
    v23 = v12;
    sub_C8D5F0((char *)&unk_5009BD8 - 16, &unk_5009BD8, (unsigned int)qword_5009BD0 + 1LL, 8);
    v13 = (unsigned int)qword_5009BD0;
    v12 = v23;
  }
  *(_QWORD *)(qword_5009BC8 + 8 * v13) = v12;
  qword_5009C10 = (__int64)&unk_49D9728;
  qword_5009B80 = (__int64)&unk_49DBF10;
  LODWORD(qword_5009BD0) = qword_5009BD0 + 1;
  qword_5009C08 = 0;
  qword_5009C20 = (__int64)&unk_49DC290;
  qword_5009C18 = 0;
  qword_5009C40 = (__int64)nullsub_24;
  qword_5009C38 = (__int64)sub_984050;
  sub_C53080(&qword_5009B80, "unroll-peel-max-count", 21);
  LODWORD(qword_5009C08) = 7;
  BYTE4(qword_5009C18) = 1;
  LODWORD(qword_5009C18) = 7;
  qword_5009BB0 = 53;
  LOBYTE(dword_5009B8C) = dword_5009B8C & 0x9F | 0x20;
  qword_5009BA8 = (__int64)"Max average trip count which will cause loop peeling.";
  sub_C53130(&qword_5009B80);
  __cxa_atexit(sub_984970, &qword_5009B80, &qword_4A427C0);
  qword_5009AA0 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5009AF0 = 0x100000000LL;
  dword_5009AAC &= 0x8000u;
  word_5009AB0 = 0;
  qword_5009AE8 = (__int64)&unk_5009AF8;
  qword_5009AB8 = 0;
  dword_5009AA8 = v14;
  qword_5009AC0 = 0;
  qword_5009AC8 = 0;
  qword_5009AD0 = 0;
  qword_5009AD8 = 0;
  qword_5009AE0 = 0;
  qword_5009B00 = 0;
  qword_5009B08 = (__int64)&unk_5009B20;
  qword_5009B10 = 1;
  dword_5009B18 = 0;
  byte_5009B1C = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_5009AF0;
  if ( (unsigned __int64)(unsigned int)qword_5009AF0 + 1 > HIDWORD(qword_5009AF0) )
  {
    v24 = v15;
    sub_C8D5F0((char *)&unk_5009AF8 - 16, &unk_5009AF8, (unsigned int)qword_5009AF0 + 1LL, 8);
    v16 = (unsigned int)qword_5009AF0;
    v15 = v24;
  }
  *(_QWORD *)(qword_5009AE8 + 8 * v16) = v15;
  qword_5009B30 = (__int64)&unk_49D9728;
  qword_5009AA0 = (__int64)&unk_49DBF10;
  LODWORD(qword_5009AF0) = qword_5009AF0 + 1;
  qword_5009B28 = 0;
  qword_5009B40 = (__int64)&unk_49DC290;
  qword_5009B38 = 0;
  qword_5009B60 = (__int64)nullsub_24;
  qword_5009B58 = (__int64)sub_984050;
  sub_C53080(&qword_5009AA0, "unroll-force-peel-count", 23);
  LODWORD(qword_5009B28) = 0;
  BYTE4(qword_5009B38) = 1;
  LODWORD(qword_5009B38) = 0;
  qword_5009AD0 = 55;
  LOBYTE(dword_5009AAC) = dword_5009AAC & 0x9F | 0x20;
  qword_5009AC8 = (__int64)"Force a peel count regardless of profiling information.";
  sub_C53130(&qword_5009AA0);
  __cxa_atexit(sub_984970, &qword_5009AA0, &qword_4A427C0);
  qword_50099C0 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5009A3C = 1;
  qword_5009A10 = 0x100000000LL;
  dword_50099CC &= 0x8000u;
  qword_50099D8 = 0;
  qword_50099E0 = 0;
  qword_50099E8 = 0;
  dword_50099C8 = v17;
  word_50099D0 = 0;
  qword_50099F0 = 0;
  qword_50099F8 = 0;
  qword_5009A00 = 0;
  qword_5009A08 = (__int64)&unk_5009A18;
  qword_5009A20 = 0;
  qword_5009A28 = (__int64)&unk_5009A40;
  qword_5009A30 = 1;
  dword_5009A38 = 0;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_5009A10;
  v20 = (unsigned int)qword_5009A10 + 1LL;
  if ( v20 > HIDWORD(qword_5009A10) )
  {
    sub_C8D5F0((char *)&unk_5009A18 - 16, &unk_5009A18, v20, 8);
    v19 = (unsigned int)qword_5009A10;
  }
  *(_QWORD *)(qword_5009A08 + 8 * v19) = v18;
  LODWORD(qword_5009A10) = qword_5009A10 + 1;
  qword_5009A48 = 0;
  qword_5009A50 = (__int64)&unk_49D9748;
  qword_5009A58 = 0;
  qword_50099C0 = (__int64)&unk_49DC090;
  qword_5009A60 = (__int64)&unk_49DC1D0;
  qword_5009A80 = (__int64)nullsub_23;
  qword_5009A78 = (__int64)sub_984030;
  sub_C53080(&qword_50099C0, "disable-advanced-peeling", 24);
  LOBYTE(qword_5009A48) = 0;
  LOWORD(qword_5009A58) = 256;
  qword_50099F0 = 65;
  LOBYTE(dword_50099CC) = dword_50099CC & 0x9F | 0x20;
  qword_50099E8 = (__int64)"Disable advance peeling. Issues for convergent targets (D134803).";
  sub_C53130(&qword_50099C0);
  return __cxa_atexit(sub_984900, &qword_50099C0, &qword_4A427C0);
}
