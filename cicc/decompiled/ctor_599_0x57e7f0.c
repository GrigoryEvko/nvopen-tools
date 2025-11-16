// Function: ctor_599
// Address: 0x57e7f0
//
int __fastcall ctor_599(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // edx
  __int64 v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]

  qword_5026DC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5026E10 = 0x100000000LL;
  dword_5026DCC &= 0x8000u;
  word_5026DD0 = 0;
  qword_5026DD8 = 0;
  qword_5026DE0 = 0;
  dword_5026DC8 = v4;
  qword_5026DE8 = 0;
  qword_5026DF0 = 0;
  qword_5026DF8 = 0;
  qword_5026E00 = 0;
  qword_5026E08 = (__int64)&unk_5026E18;
  qword_5026E20 = 0;
  qword_5026E28 = (__int64)&unk_5026E40;
  qword_5026E30 = 1;
  dword_5026E38 = 0;
  byte_5026E3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5026E10;
  v7 = (unsigned int)qword_5026E10 + 1LL;
  if ( v7 > HIDWORD(qword_5026E10) )
  {
    sub_C8D5F0((char *)&unk_5026E18 - 16, &unk_5026E18, v7, 8);
    v6 = (unsigned int)qword_5026E10;
  }
  *(_QWORD *)(qword_5026E08 + 8 * v6) = v5;
  LODWORD(qword_5026E10) = qword_5026E10 + 1;
  qword_5026E48 = 0;
  qword_5026E50 = (__int64)&unk_49D9748;
  qword_5026E58 = 0;
  qword_5026DC0 = (__int64)&unk_49DC090;
  qword_5026E60 = (__int64)&unk_49DC1D0;
  qword_5026E80 = (__int64)nullsub_23;
  qword_5026E78 = (__int64)sub_984030;
  sub_C53080(&qword_5026DC0, "jump-is-expensive", 17);
  LOBYTE(qword_5026E48) = 0;
  qword_5026DE8 = (__int64)"Do not create extra branches to split comparison logic.";
  LOWORD(qword_5026E58) = 256;
  qword_5026DF0 = 55;
  LOBYTE(dword_5026DCC) = dword_5026DCC & 0x9F | 0x20;
  sub_C53130(&qword_5026DC0);
  __cxa_atexit(sub_984900, &qword_5026DC0, &qword_4A427C0);
  qword_5026CE0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5026DC0, v8, v9), 1u);
  qword_5026D30 = 0x100000000LL;
  dword_5026CEC &= 0x8000u;
  word_5026CF0 = 0;
  qword_5026CF8 = 0;
  qword_5026D00 = 0;
  dword_5026CE8 = v10;
  qword_5026D08 = 0;
  qword_5026D10 = 0;
  qword_5026D18 = 0;
  qword_5026D20 = 0;
  qword_5026D28 = (__int64)&unk_5026D38;
  qword_5026D40 = 0;
  qword_5026D48 = (__int64)&unk_5026D60;
  qword_5026D50 = 1;
  dword_5026D58 = 0;
  byte_5026D5C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5026D30;
  v13 = (unsigned int)qword_5026D30 + 1LL;
  if ( v13 > HIDWORD(qword_5026D30) )
  {
    sub_C8D5F0((char *)&unk_5026D38 - 16, &unk_5026D38, v13, 8);
    v12 = (unsigned int)qword_5026D30;
  }
  *(_QWORD *)(qword_5026D28 + 8 * v12) = v11;
  qword_5026D70 = (__int64)&unk_49D9728;
  qword_5026CE0 = (__int64)&unk_49DBF10;
  LODWORD(qword_5026D30) = qword_5026D30 + 1;
  qword_5026D68 = 0;
  qword_5026D80 = (__int64)&unk_49DC290;
  qword_5026D78 = 0;
  qword_5026DA0 = (__int64)nullsub_24;
  qword_5026D98 = (__int64)sub_984050;
  sub_C53080(&qword_5026CE0, "min-jump-table-entries", 22);
  LODWORD(qword_5026D68) = 4;
  BYTE4(qword_5026D78) = 1;
  LODWORD(qword_5026D78) = 4;
  qword_5026D10 = 50;
  LOBYTE(dword_5026CEC) = dword_5026CEC & 0x9F | 0x20;
  qword_5026D08 = (__int64)"Set minimum number of entries to use a jump table.";
  sub_C53130(&qword_5026CE0);
  __cxa_atexit(sub_984970, &qword_5026CE0, &qword_4A427C0);
  qword_5026C00 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5026CE0, v14, v15), 1u);
  byte_5026C7C = 1;
  word_5026C10 = 0;
  qword_5026C50 = 0x100000000LL;
  dword_5026C0C &= 0x8000u;
  qword_5026C48 = (__int64)&unk_5026C58;
  qword_5026C18 = 0;
  dword_5026C08 = v16;
  qword_5026C20 = 0;
  qword_5026C28 = 0;
  qword_5026C30 = 0;
  qword_5026C38 = 0;
  qword_5026C40 = 0;
  qword_5026C60 = 0;
  qword_5026C68 = (__int64)&unk_5026C80;
  qword_5026C70 = 1;
  dword_5026C78 = 0;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_5026C50;
  if ( (unsigned __int64)(unsigned int)qword_5026C50 + 1 > HIDWORD(qword_5026C50) )
  {
    v31 = v17;
    sub_C8D5F0((char *)&unk_5026C58 - 16, &unk_5026C58, (unsigned int)qword_5026C50 + 1LL, 8);
    v18 = (unsigned int)qword_5026C50;
    v17 = v31;
  }
  *(_QWORD *)(qword_5026C48 + 8 * v18) = v17;
  qword_5026C90 = (__int64)&unk_49D9728;
  qword_5026C00 = (__int64)&unk_49DBF10;
  LODWORD(qword_5026C50) = qword_5026C50 + 1;
  qword_5026C88 = 0;
  qword_5026CA0 = (__int64)&unk_49DC290;
  qword_5026C98 = 0;
  qword_5026CC0 = (__int64)nullsub_24;
  qword_5026CB8 = (__int64)sub_984050;
  sub_C53080(&qword_5026C00, "max-jump-table-size", 19);
  LODWORD(qword_5026C88) = -1;
  BYTE4(qword_5026C98) = 1;
  LODWORD(qword_5026C98) = -1;
  qword_5026C30 = 32;
  LOBYTE(dword_5026C0C) = dword_5026C0C & 0x9F | 0x20;
  qword_5026C28 = (__int64)"Set maximum size of jump tables.";
  sub_C53130(&qword_5026C00);
  __cxa_atexit(sub_984970, &qword_5026C00, &qword_4A427C0);
  qword_5026B20 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5026C00, v19, v20), 1u);
  qword_5026B70 = 0x100000000LL;
  dword_5026B2C &= 0x8000u;
  word_5026B30 = 0;
  qword_5026B68 = (__int64)&unk_5026B78;
  qword_5026B38 = 0;
  dword_5026B28 = v21;
  qword_5026B40 = 0;
  qword_5026B48 = 0;
  qword_5026B50 = 0;
  qword_5026B58 = 0;
  qword_5026B60 = 0;
  qword_5026B80 = 0;
  qword_5026B88 = (__int64)&unk_5026BA0;
  qword_5026B90 = 1;
  dword_5026B98 = 0;
  byte_5026B9C = 1;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_5026B70;
  if ( (unsigned __int64)(unsigned int)qword_5026B70 + 1 > HIDWORD(qword_5026B70) )
  {
    v32 = v22;
    sub_C8D5F0((char *)&unk_5026B78 - 16, &unk_5026B78, (unsigned int)qword_5026B70 + 1LL, 8);
    v23 = (unsigned int)qword_5026B70;
    v22 = v32;
  }
  *(_QWORD *)(qword_5026B68 + 8 * v23) = v22;
  qword_5026BB0 = (__int64)&unk_49D9728;
  qword_5026B20 = (__int64)&unk_49DBF10;
  LODWORD(qword_5026B70) = qword_5026B70 + 1;
  qword_5026BA8 = 0;
  qword_5026BC0 = (__int64)&unk_49DC290;
  qword_5026BB8 = 0;
  qword_5026BE0 = (__int64)nullsub_24;
  qword_5026BD8 = (__int64)sub_984050;
  sub_C53080(&qword_5026B20, "optsize-jump-table-density", 26);
  LODWORD(qword_5026BA8) = 40;
  BYTE4(qword_5026BB8) = 1;
  LODWORD(qword_5026BB8) = 40;
  qword_5026B50 = 64;
  LOBYTE(dword_5026B2C) = dword_5026B2C & 0x9F | 0x20;
  qword_5026B48 = (__int64)"Minimum density for building a jump table in an optsize function";
  sub_C53130(&qword_5026B20);
  __cxa_atexit(sub_984970, &qword_5026B20, &qword_4A427C0);
  qword_5026A40 = (__int64)&unk_49DC150;
  v26 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5026B20, v24, v25), 1u);
  byte_5026ABC = 1;
  qword_5026A90 = 0x100000000LL;
  dword_5026A4C &= 0x8000u;
  qword_5026A58 = 0;
  qword_5026A60 = 0;
  qword_5026A68 = 0;
  dword_5026A48 = v26;
  word_5026A50 = 0;
  qword_5026A70 = 0;
  qword_5026A78 = 0;
  qword_5026A80 = 0;
  qword_5026A88 = (__int64)&unk_5026A98;
  qword_5026AA0 = 0;
  qword_5026AA8 = (__int64)&unk_5026AC0;
  qword_5026AB0 = 1;
  dword_5026AB8 = 0;
  v27 = sub_C57470();
  v28 = (unsigned int)qword_5026A90;
  v29 = (unsigned int)qword_5026A90 + 1LL;
  if ( v29 > HIDWORD(qword_5026A90) )
  {
    sub_C8D5F0((char *)&unk_5026A98 - 16, &unk_5026A98, v29, 8);
    v28 = (unsigned int)qword_5026A90;
  }
  *(_QWORD *)(qword_5026A88 + 8 * v28) = v27;
  LODWORD(qword_5026A90) = qword_5026A90 + 1;
  qword_5026AC8 = 0;
  qword_5026AD0 = (__int64)&unk_49D9748;
  qword_5026AD8 = 0;
  qword_5026A40 = (__int64)&unk_49DC090;
  qword_5026AE0 = (__int64)&unk_49DC1D0;
  qword_5026B00 = (__int64)nullsub_23;
  qword_5026AF8 = (__int64)sub_984030;
  sub_C53080(&qword_5026A40, "disable-strictnode-mutation", 27);
  qword_5026A70 = 49;
  qword_5026A68 = (__int64)"Don't mutate strict-float node to a legalize node";
  LOWORD(qword_5026AD8) = 256;
  LOBYTE(qword_5026AC8) = 0;
  LOBYTE(dword_5026A4C) = dword_5026A4C & 0x9F | 0x20;
  sub_C53130(&qword_5026A40);
  return __cxa_atexit(sub_984900, &qword_5026A40, &qword_4A427C0);
}
