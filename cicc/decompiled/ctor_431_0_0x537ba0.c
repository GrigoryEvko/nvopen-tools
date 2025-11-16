// Function: ctor_431_0
// Address: 0x537ba0
//
int ctor_431_0()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  int v20; // edx
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // edx
  __int64 v28; // r14
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  int v31; // edx
  __int64 v32; // r14
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  int v35; // edx
  __int64 v36; // rax
  __int64 v37; // rdx
  int v38; // edx
  __int64 v39; // r14
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]
  __int64 v45; // [rsp+8h] [rbp-38h]
  __int64 v46; // [rsp+8h] [rbp-38h]
  __int64 v47; // [rsp+8h] [rbp-38h]
  __int64 v48; // [rsp+8h] [rbp-38h]

  qword_4FF5EC0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF5F10 = 0x100000000LL;
  word_4FF5ED0 = 0;
  dword_4FF5ECC &= 0x8000u;
  qword_4FF5ED8 = 0;
  qword_4FF5EE0 = 0;
  dword_4FF5EC8 = v0;
  qword_4FF5EE8 = 0;
  qword_4FF5EF0 = 0;
  qword_4FF5EF8 = 0;
  qword_4FF5F00 = 0;
  qword_4FF5F08 = (__int64)&unk_4FF5F18;
  qword_4FF5F20 = 0;
  qword_4FF5F28 = (__int64)&unk_4FF5F40;
  qword_4FF5F30 = 1;
  dword_4FF5F38 = 0;
  byte_4FF5F3C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF5F10;
  v3 = (unsigned int)qword_4FF5F10 + 1LL;
  if ( v3 > HIDWORD(qword_4FF5F10) )
  {
    sub_C8D5F0((char *)&unk_4FF5F18 - 16, &unk_4FF5F18, v3, 8);
    v2 = (unsigned int)qword_4FF5F10;
  }
  *(_QWORD *)(qword_4FF5F08 + 8 * v2) = v1;
  qword_4FF5F50 = (__int64)&unk_49D9748;
  qword_4FF5EC0 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF5F10) = qword_4FF5F10 + 1;
  qword_4FF5F80 = (__int64)nullsub_23;
  qword_4FF5F60 = (__int64)&unk_49DC1D0;
  qword_4FF5F48 = 0;
  qword_4FF5F78 = (__int64)sub_984030;
  qword_4FF5F58 = 0;
  sub_C53080(&qword_4FF5EC0, "disable-partial-inlining", 24);
  LOBYTE(qword_4FF5F48) = 0;
  LOWORD(qword_4FF5F58) = 256;
  qword_4FF5EF0 = 24;
  LOBYTE(dword_4FF5ECC) = dword_4FF5ECC & 0x9F | 0x20;
  qword_4FF5EE8 = (__int64)"Disable partial inlining";
  sub_C53130(&qword_4FF5EC0);
  __cxa_atexit(sub_984900, &qword_4FF5EC0, &qword_4A427C0);
  qword_4FF5DE0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF5E5C = 1;
  qword_4FF5E30 = 0x100000000LL;
  dword_4FF5DEC &= 0x8000u;
  qword_4FF5E28 = (__int64)&unk_4FF5E38;
  qword_4FF5DF8 = 0;
  qword_4FF5E00 = 0;
  dword_4FF5DE8 = v4;
  word_4FF5DF0 = 0;
  qword_4FF5E08 = 0;
  qword_4FF5E10 = 0;
  qword_4FF5E18 = 0;
  qword_4FF5E20 = 0;
  qword_4FF5E40 = 0;
  qword_4FF5E48 = (__int64)&unk_4FF5E60;
  qword_4FF5E50 = 1;
  dword_4FF5E58 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF5E30;
  if ( (unsigned __int64)(unsigned int)qword_4FF5E30 + 1 > HIDWORD(qword_4FF5E30) )
  {
    v43 = v5;
    sub_C8D5F0((char *)&unk_4FF5E38 - 16, &unk_4FF5E38, (unsigned int)qword_4FF5E30 + 1LL, 8);
    v6 = (unsigned int)qword_4FF5E30;
    v5 = v43;
  }
  *(_QWORD *)(qword_4FF5E28 + 8 * v6) = v5;
  qword_4FF5E70 = (__int64)&unk_49D9748;
  qword_4FF5DE0 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF5E30) = qword_4FF5E30 + 1;
  qword_4FF5EA0 = (__int64)nullsub_23;
  qword_4FF5E80 = (__int64)&unk_49DC1D0;
  qword_4FF5E68 = 0;
  qword_4FF5E98 = (__int64)sub_984030;
  qword_4FF5E78 = 0;
  sub_C53080(&qword_4FF5DE0, "disable-mr-partial-inlining", 27);
  LOBYTE(qword_4FF5E68) = 0;
  LOWORD(qword_4FF5E78) = 256;
  qword_4FF5E10 = 37;
  LOBYTE(dword_4FF5DEC) = dword_4FF5DEC & 0x9F | 0x20;
  qword_4FF5E08 = (__int64)"Disable multi-region partial inlining";
  sub_C53130(&qword_4FF5DE0);
  __cxa_atexit(sub_984900, &qword_4FF5DE0, &qword_4A427C0);
  qword_4FF5D00 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FF5D0C &= 0x8000u;
  word_4FF5D10 = 0;
  qword_4FF5D50 = 0x100000000LL;
  qword_4FF5D48 = (__int64)&unk_4FF5D58;
  qword_4FF5D18 = 0;
  qword_4FF5D20 = 0;
  dword_4FF5D08 = v7;
  qword_4FF5D28 = 0;
  qword_4FF5D30 = 0;
  qword_4FF5D38 = 0;
  qword_4FF5D40 = 0;
  qword_4FF5D60 = 0;
  qword_4FF5D68 = (__int64)&unk_4FF5D80;
  qword_4FF5D70 = 1;
  dword_4FF5D78 = 0;
  byte_4FF5D7C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FF5D50;
  if ( (unsigned __int64)(unsigned int)qword_4FF5D50 + 1 > HIDWORD(qword_4FF5D50) )
  {
    v44 = v8;
    sub_C8D5F0((char *)&unk_4FF5D58 - 16, &unk_4FF5D58, (unsigned int)qword_4FF5D50 + 1LL, 8);
    v9 = (unsigned int)qword_4FF5D50;
    v8 = v44;
  }
  *(_QWORD *)(qword_4FF5D48 + 8 * v9) = v8;
  qword_4FF5D90 = (__int64)&unk_49D9748;
  qword_4FF5D00 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF5D50) = qword_4FF5D50 + 1;
  qword_4FF5DC0 = (__int64)nullsub_23;
  qword_4FF5DA0 = (__int64)&unk_49DC1D0;
  qword_4FF5D88 = 0;
  qword_4FF5DB8 = (__int64)sub_984030;
  qword_4FF5D98 = 0;
  sub_C53080(&qword_4FF5D00, "pi-force-live-exit-outline", 26);
  LOBYTE(qword_4FF5D88) = 0;
  LOWORD(qword_4FF5D98) = 256;
  qword_4FF5D30 = 37;
  LOBYTE(dword_4FF5D0C) = dword_4FF5D0C & 0x9F | 0x20;
  qword_4FF5D28 = (__int64)"Force outline regions with live exits";
  sub_C53130(&qword_4FF5D00);
  __cxa_atexit(sub_984900, &qword_4FF5D00, &qword_4A427C0);
  qword_4FF5C20 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FF5C2C &= 0x8000u;
  word_4FF5C30 = 0;
  qword_4FF5C70 = 0x100000000LL;
  qword_4FF5C68 = (__int64)&unk_4FF5C78;
  qword_4FF5C38 = 0;
  qword_4FF5C40 = 0;
  dword_4FF5C28 = v10;
  qword_4FF5C48 = 0;
  qword_4FF5C50 = 0;
  qword_4FF5C58 = 0;
  qword_4FF5C60 = 0;
  qword_4FF5C80 = 0;
  qword_4FF5C88 = (__int64)&unk_4FF5CA0;
  qword_4FF5C90 = 1;
  dword_4FF5C98 = 0;
  byte_4FF5C9C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_4FF5C70;
  if ( (unsigned __int64)(unsigned int)qword_4FF5C70 + 1 > HIDWORD(qword_4FF5C70) )
  {
    v45 = v11;
    sub_C8D5F0((char *)&unk_4FF5C78 - 16, &unk_4FF5C78, (unsigned int)qword_4FF5C70 + 1LL, 8);
    v12 = (unsigned int)qword_4FF5C70;
    v11 = v45;
  }
  *(_QWORD *)(qword_4FF5C68 + 8 * v12) = v11;
  qword_4FF5CB0 = (__int64)&unk_49D9748;
  qword_4FF5C20 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF5C70) = qword_4FF5C70 + 1;
  qword_4FF5CE0 = (__int64)nullsub_23;
  qword_4FF5CC0 = (__int64)&unk_49DC1D0;
  qword_4FF5CA8 = 0;
  qword_4FF5CD8 = (__int64)sub_984030;
  qword_4FF5CB8 = 0;
  sub_C53080(&qword_4FF5C20, "pi-mark-coldcc", 14);
  LOWORD(qword_4FF5CB8) = 256;
  LOBYTE(qword_4FF5CA8) = 0;
  qword_4FF5C50 = 39;
  LOBYTE(dword_4FF5C2C) = dword_4FF5C2C & 0x9F | 0x20;
  qword_4FF5C48 = (__int64)"Mark outline function calls with ColdCC";
  sub_C53130(&qword_4FF5C20);
  __cxa_atexit(sub_984900, &qword_4FF5C20, &qword_4A427C0);
  qword_4FF5B40 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF5B90 = 0x100000000LL;
  dword_4FF5B4C &= 0x8000u;
  qword_4FF5B88 = (__int64)&unk_4FF5B98;
  word_4FF5B50 = 0;
  qword_4FF5B58 = 0;
  dword_4FF5B48 = v13;
  qword_4FF5B60 = 0;
  qword_4FF5B68 = 0;
  qword_4FF5B70 = 0;
  qword_4FF5B78 = 0;
  qword_4FF5B80 = 0;
  qword_4FF5BA0 = 0;
  qword_4FF5BA8 = (__int64)&unk_4FF5BC0;
  qword_4FF5BB0 = 1;
  dword_4FF5BB8 = 0;
  byte_4FF5BBC = 1;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_4FF5B90;
  if ( (unsigned __int64)(unsigned int)qword_4FF5B90 + 1 > HIDWORD(qword_4FF5B90) )
  {
    v46 = v14;
    sub_C8D5F0((char *)&unk_4FF5B98 - 16, &unk_4FF5B98, (unsigned int)qword_4FF5B90 + 1LL, 8);
    v15 = (unsigned int)qword_4FF5B90;
    v14 = v46;
  }
  *(_QWORD *)(qword_4FF5B88 + 8 * v15) = v14;
  qword_4FF5BD0 = (__int64)&unk_49D9748;
  qword_4FF5B40 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF5B90) = qword_4FF5B90 + 1;
  qword_4FF5C00 = (__int64)nullsub_23;
  qword_4FF5BE0 = (__int64)&unk_49DC1D0;
  qword_4FF5BC8 = 0;
  qword_4FF5BF8 = (__int64)sub_984030;
  qword_4FF5BD8 = 0;
  sub_C53080(&qword_4FF5B40, "skip-partial-inlining-cost-analysis", 35);
  qword_4FF5B70 = 18;
  LOBYTE(dword_4FF5B4C) = dword_4FF5B4C & 0x9F | 0x40;
  qword_4FF5B68 = (__int64)"Skip Cost Analysis";
  sub_C53130(&qword_4FF5B40);
  __cxa_atexit(sub_984900, &qword_4FF5B40, &qword_4A427C0);
  qword_4FF5A60 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF5AB0 = 0x100000000LL;
  dword_4FF5A6C &= 0x8000u;
  word_4FF5A70 = 0;
  qword_4FF5A78 = 0;
  qword_4FF5A80 = 0;
  dword_4FF5A68 = v16;
  qword_4FF5A88 = 0;
  qword_4FF5A90 = 0;
  qword_4FF5A98 = 0;
  qword_4FF5AA0 = 0;
  qword_4FF5AA8 = (__int64)&unk_4FF5AB8;
  qword_4FF5AC0 = 0;
  qword_4FF5AC8 = (__int64)&unk_4FF5AE0;
  qword_4FF5AD0 = 1;
  dword_4FF5AD8 = 0;
  byte_4FF5ADC = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_4FF5AB0;
  v19 = (unsigned int)qword_4FF5AB0 + 1LL;
  if ( v19 > HIDWORD(qword_4FF5AB0) )
  {
    sub_C8D5F0((char *)&unk_4FF5AB8 - 16, &unk_4FF5AB8, v19, 8);
    v18 = (unsigned int)qword_4FF5AB0;
  }
  *(_QWORD *)(qword_4FF5AA8 + 8 * v18) = v17;
  LODWORD(qword_4FF5AB0) = qword_4FF5AB0 + 1;
  qword_4FF5AE8 = 0;
  qword_4FF5AF0 = (__int64)&unk_49E5940;
  qword_4FF5AF8 = 0;
  qword_4FF5A60 = (__int64)&unk_49E5960;
  qword_4FF5B00 = (__int64)&unk_49DC320;
  qword_4FF5B20 = (__int64)nullsub_385;
  qword_4FF5B18 = (__int64)sub_1038930;
  sub_C53080(&qword_4FF5A60, "min-region-size-ratio", 21);
  LODWORD(qword_4FF5AE8) = 1036831949;
  BYTE4(qword_4FF5AF8) = 1;
  LODWORD(qword_4FF5AF8) = 1036831949;
  qword_4FF5A90 = 86;
  LOBYTE(dword_4FF5A6C) = dword_4FF5A6C & 0x9F | 0x20;
  qword_4FF5A88 = (__int64)"Minimum ratio comparing relative sizes of each outline candidate and original function";
  sub_C53130(&qword_4FF5A60);
  __cxa_atexit(sub_1038DB0, &qword_4FF5A60, &qword_4A427C0);
  qword_4FF5980 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF59D0 = 0x100000000LL;
  dword_4FF598C &= 0x8000u;
  word_4FF5990 = 0;
  qword_4FF5998 = 0;
  qword_4FF59A0 = 0;
  dword_4FF5988 = v20;
  qword_4FF59A8 = 0;
  qword_4FF59B0 = 0;
  qword_4FF59B8 = 0;
  qword_4FF59C0 = 0;
  qword_4FF59C8 = (__int64)&unk_4FF59D8;
  qword_4FF59E0 = 0;
  qword_4FF59E8 = (__int64)&unk_4FF5A00;
  qword_4FF59F0 = 1;
  dword_4FF59F8 = 0;
  byte_4FF59FC = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_4FF59D0;
  v23 = (unsigned int)qword_4FF59D0 + 1LL;
  if ( v23 > HIDWORD(qword_4FF59D0) )
  {
    sub_C8D5F0((char *)&unk_4FF59D8 - 16, &unk_4FF59D8, v23, 8);
    v22 = (unsigned int)qword_4FF59D0;
  }
  *(_QWORD *)(qword_4FF59C8 + 8 * v22) = v21;
  LODWORD(qword_4FF59D0) = qword_4FF59D0 + 1;
  qword_4FF5A08 = 0;
  qword_4FF5A10 = (__int64)&unk_49D9728;
  qword_4FF5980 = (__int64)&unk_49DBF10;
  qword_4FF5A20 = (__int64)&unk_49DC290;
  qword_4FF5A18 = 0;
  qword_4FF5A40 = (__int64)nullsub_24;
  qword_4FF5A38 = (__int64)sub_984050;
  sub_C53080(&qword_4FF5980, "min-block-execution", 19);
  LODWORD(qword_4FF5A08) = 100;
  BYTE4(qword_4FF5A18) = 1;
  LODWORD(qword_4FF5A18) = 100;
  qword_4FF59B0 = 68;
  LOBYTE(dword_4FF598C) = dword_4FF598C & 0x9F | 0x20;
  qword_4FF59A8 = (__int64)"Minimum block executions to consider its BranchProbabilityInfo valid";
  sub_C53130(&qword_4FF5980);
  __cxa_atexit(sub_984970, &qword_4FF5980, &qword_4A427C0);
  qword_4FF58A0 = (__int64)&unk_49DC150;
  v24 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF58F0 = 0x100000000LL;
  dword_4FF58AC &= 0x8000u;
  qword_4FF58E8 = (__int64)&unk_4FF58F8;
  word_4FF58B0 = 0;
  qword_4FF58B8 = 0;
  dword_4FF58A8 = v24;
  qword_4FF58C0 = 0;
  qword_4FF58C8 = 0;
  qword_4FF58D0 = 0;
  qword_4FF58D8 = 0;
  qword_4FF58E0 = 0;
  qword_4FF5900 = 0;
  qword_4FF5908 = (__int64)&unk_4FF5920;
  qword_4FF5910 = 1;
  dword_4FF5918 = 0;
  byte_4FF591C = 1;
  v25 = sub_C57470();
  v26 = (unsigned int)qword_4FF58F0;
  if ( (unsigned __int64)(unsigned int)qword_4FF58F0 + 1 > HIDWORD(qword_4FF58F0) )
  {
    v47 = v25;
    sub_C8D5F0((char *)&unk_4FF58F8 - 16, &unk_4FF58F8, (unsigned int)qword_4FF58F0 + 1LL, 8);
    v26 = (unsigned int)qword_4FF58F0;
    v25 = v47;
  }
  *(_QWORD *)(qword_4FF58E8 + 8 * v26) = v25;
  LODWORD(qword_4FF58F0) = qword_4FF58F0 + 1;
  qword_4FF5928 = 0;
  qword_4FF5930 = (__int64)&unk_49E5940;
  qword_4FF5938 = 0;
  qword_4FF58A0 = (__int64)&unk_49E5960;
  qword_4FF5940 = (__int64)&unk_49DC320;
  qword_4FF5960 = (__int64)nullsub_385;
  qword_4FF5958 = (__int64)sub_1038930;
  sub_C53080(&qword_4FF58A0, "cold-branch-ratio", 17);
  LODWORD(qword_4FF5928) = 1036831949;
  BYTE4(qword_4FF5938) = 1;
  LODWORD(qword_4FF5938) = 1036831949;
  qword_4FF58D0 = 52;
  LOBYTE(dword_4FF58AC) = dword_4FF58AC & 0x9F | 0x20;
  qword_4FF58C8 = (__int64)"Minimum BranchProbability to consider a region cold.";
  sub_C53130(&qword_4FF58A0);
  __cxa_atexit(sub_1038DB0, &qword_4FF58A0, &qword_4A427C0);
  qword_4FF57C0 = (__int64)&unk_49DC150;
  v27 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF583C = 1;
  word_4FF57D0 = 0;
  qword_4FF5810 = 0x100000000LL;
  dword_4FF57CC &= 0x8000u;
  qword_4FF5808 = (__int64)&unk_4FF5818;
  qword_4FF57D8 = 0;
  dword_4FF57C8 = v27;
  qword_4FF57E0 = 0;
  qword_4FF57E8 = 0;
  qword_4FF57F0 = 0;
  qword_4FF57F8 = 0;
  qword_4FF5800 = 0;
  qword_4FF5820 = 0;
  qword_4FF5828 = (__int64)&unk_4FF5840;
  qword_4FF5830 = 1;
  dword_4FF5838 = 0;
  v28 = sub_C57470();
  v29 = (unsigned int)qword_4FF5810;
  v30 = (unsigned int)qword_4FF5810 + 1LL;
  if ( v30 > HIDWORD(qword_4FF5810) )
  {
    sub_C8D5F0((char *)&unk_4FF5818 - 16, &unk_4FF5818, v30, 8);
    v29 = (unsigned int)qword_4FF5810;
  }
  *(_QWORD *)(qword_4FF5808 + 8 * v29) = v28;
  LODWORD(qword_4FF5810) = qword_4FF5810 + 1;
  qword_4FF5848 = 0;
  qword_4FF5850 = (__int64)&unk_49D9728;
  qword_4FF57C0 = (__int64)&unk_49DBF10;
  qword_4FF5860 = (__int64)&unk_49DC290;
  qword_4FF5858 = 0;
  qword_4FF5880 = (__int64)nullsub_24;
  qword_4FF5878 = (__int64)sub_984050;
  sub_C53080(&qword_4FF57C0, "max-num-inline-blocks", 21);
  LODWORD(qword_4FF5848) = 5;
  BYTE4(qword_4FF5858) = 1;
  LODWORD(qword_4FF5858) = 5;
  qword_4FF57F0 = 44;
  LOBYTE(dword_4FF57CC) = dword_4FF57CC & 0x9F | 0x20;
  qword_4FF57E8 = (__int64)"Max number of blocks to be partially inlined";
  sub_C53130(&qword_4FF57C0);
  __cxa_atexit(sub_984970, &qword_4FF57C0, &qword_4A427C0);
  qword_4FF56E0 = (__int64)&unk_49DC150;
  v31 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF5730 = 0x100000000LL;
  dword_4FF56EC &= 0x8000u;
  word_4FF56F0 = 0;
  qword_4FF5728 = (__int64)&unk_4FF5738;
  qword_4FF56F8 = 0;
  dword_4FF56E8 = v31;
  qword_4FF5700 = 0;
  qword_4FF5708 = 0;
  qword_4FF5710 = 0;
  qword_4FF5718 = 0;
  qword_4FF5720 = 0;
  qword_4FF5740 = 0;
  qword_4FF5748 = (__int64)&unk_4FF5760;
  qword_4FF5750 = 1;
  dword_4FF5758 = 0;
  byte_4FF575C = 1;
  v32 = sub_C57470();
  v33 = (unsigned int)qword_4FF5730;
  v34 = (unsigned int)qword_4FF5730 + 1LL;
  if ( v34 > HIDWORD(qword_4FF5730) )
  {
    sub_C8D5F0((char *)&unk_4FF5738 - 16, &unk_4FF5738, v34, 8);
    v33 = (unsigned int)qword_4FF5730;
  }
  *(_QWORD *)(qword_4FF5728 + 8 * v33) = v32;
  LODWORD(qword_4FF5730) = qword_4FF5730 + 1;
  qword_4FF5768 = 0;
  qword_4FF5770 = (__int64)&unk_49DA090;
  qword_4FF5778 = 0;
  qword_4FF56E0 = (__int64)&unk_49DBF90;
  qword_4FF5780 = (__int64)&unk_49DC230;
  qword_4FF57A0 = (__int64)nullsub_58;
  qword_4FF5798 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FF56E0, "max-partial-inlining", 20);
  LODWORD(qword_4FF5768) = -1;
  BYTE4(qword_4FF5778) = 1;
  LODWORD(qword_4FF5778) = -1;
  qword_4FF5710 = 56;
  LOBYTE(dword_4FF56EC) = dword_4FF56EC & 0x9F | 0x20;
  qword_4FF5708 = (__int64)"Max number of partial inlining. The default is unlimited";
  sub_C53130(&qword_4FF56E0);
  __cxa_atexit(sub_B2B680, &qword_4FF56E0, &qword_4A427C0);
  qword_4FF5600 = (__int64)&unk_49DC150;
  v35 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF567C = 1;
  qword_4FF5650 = 0x100000000LL;
  dword_4FF560C &= 0x8000u;
  qword_4FF5648 = (__int64)&unk_4FF5658;
  qword_4FF5618 = 0;
  qword_4FF5620 = 0;
  dword_4FF5608 = v35;
  word_4FF5610 = 0;
  qword_4FF5628 = 0;
  qword_4FF5630 = 0;
  qword_4FF5638 = 0;
  qword_4FF5640 = 0;
  qword_4FF5660 = 0;
  qword_4FF5668 = (__int64)&unk_4FF5680;
  qword_4FF5670 = 1;
  dword_4FF5678 = 0;
  v36 = sub_C57470();
  v37 = (unsigned int)qword_4FF5650;
  if ( (unsigned __int64)(unsigned int)qword_4FF5650 + 1 > HIDWORD(qword_4FF5650) )
  {
    v48 = v36;
    sub_C8D5F0((char *)&unk_4FF5658 - 16, &unk_4FF5658, (unsigned int)qword_4FF5650 + 1LL, 8);
    v37 = (unsigned int)qword_4FF5650;
    v36 = v48;
  }
  *(_QWORD *)(qword_4FF5648 + 8 * v37) = v36;
  LODWORD(qword_4FF5650) = qword_4FF5650 + 1;
  qword_4FF5688 = 0;
  qword_4FF5690 = (__int64)&unk_49DA090;
  qword_4FF5698 = 0;
  qword_4FF5600 = (__int64)&unk_49DBF90;
  qword_4FF56A0 = (__int64)&unk_49DC230;
  qword_4FF56C0 = (__int64)nullsub_58;
  qword_4FF56B8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FF5600, "outline-region-freq-percent", 27);
  LODWORD(qword_4FF5688) = 75;
  BYTE4(qword_4FF5698) = 1;
  LODWORD(qword_4FF5698) = 75;
  qword_4FF5630 = 55;
  LOBYTE(dword_4FF560C) = dword_4FF560C & 0x9F | 0x20;
  qword_4FF5628 = (__int64)"Relative frequency of outline region to the entry block";
  sub_C53130(&qword_4FF5600);
  __cxa_atexit(sub_B2B680, &qword_4FF5600, &qword_4A427C0);
  qword_4FF5520 = (__int64)&unk_49DC150;
  v38 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FF552C &= 0x8000u;
  word_4FF5530 = 0;
  qword_4FF5570 = 0x100000000LL;
  qword_4FF5568 = (__int64)&unk_4FF5578;
  qword_4FF5538 = 0;
  qword_4FF5540 = 0;
  dword_4FF5528 = v38;
  qword_4FF5548 = 0;
  qword_4FF5550 = 0;
  qword_4FF5558 = 0;
  qword_4FF5560 = 0;
  qword_4FF5580 = 0;
  qword_4FF5588 = (__int64)&unk_4FF55A0;
  qword_4FF5590 = 1;
  dword_4FF5598 = 0;
  byte_4FF559C = 1;
  v39 = sub_C57470();
  v40 = (unsigned int)qword_4FF5570;
  v41 = (unsigned int)qword_4FF5570 + 1LL;
  if ( v41 > HIDWORD(qword_4FF5570) )
  {
    sub_C8D5F0((char *)&unk_4FF5578 - 16, &unk_4FF5578, v41, 8);
    v40 = (unsigned int)qword_4FF5570;
  }
  *(_QWORD *)(qword_4FF5568 + 8 * v40) = v39;
  LODWORD(qword_4FF5570) = qword_4FF5570 + 1;
  qword_4FF55A8 = 0;
  qword_4FF55B0 = (__int64)&unk_49D9728;
  qword_4FF5520 = (__int64)&unk_49DBF10;
  qword_4FF55C0 = (__int64)&unk_49DC290;
  qword_4FF55B8 = 0;
  qword_4FF55E0 = (__int64)nullsub_24;
  qword_4FF55D8 = (__int64)sub_984050;
  sub_C53080(&qword_4FF5520, "partial-inlining-extra-penalty", 30);
  LODWORD(qword_4FF55A8) = 0;
  BYTE4(qword_4FF55B8) = 1;
  LODWORD(qword_4FF55B8) = 0;
  qword_4FF5550 = 61;
  LOBYTE(dword_4FF552C) = dword_4FF552C & 0x9F | 0x20;
  qword_4FF5548 = (__int64)"A debug option to add additional penalty to the computed one.";
  sub_C53130(&qword_4FF5520);
  return __cxa_atexit(sub_984970, &qword_4FF5520, &qword_4A427C0);
}
