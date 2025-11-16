// Function: ctor_403_0
// Address: 0x527d40
//
int ctor_403_0()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // edx
  __int64 v30; // r15
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  int v33; // edx
  __int64 v34; // rbx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  int v37; // edx
  __int64 v38; // rbx
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  int v41; // edx
  __int64 v42; // rbx
  __int64 v43; // rax
  int v44; // edx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v48; // [rsp+8h] [rbp-78h]
  __int64 v49; // [rsp+8h] [rbp-78h]
  __int64 v50; // [rsp+8h] [rbp-78h]
  __int64 v51; // [rsp+18h] [rbp-68h]
  int v52; // [rsp+20h] [rbp-60h] BYREF
  int v53; // [rsp+24h] [rbp-5Ch] BYREF
  int *v54; // [rsp+28h] [rbp-58h] BYREF
  const char *v55; // [rsp+30h] [rbp-50h] BYREF
  __int64 v56; // [rsp+38h] [rbp-48h]
  char v57[64]; // [rsp+40h] [rbp-40h] BYREF

  qword_4FE9FC0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEA010 = 0x100000000LL;
  dword_4FE9FCC &= 0x8000u;
  word_4FE9FD0 = 0;
  qword_4FE9FD8 = 0;
  qword_4FE9FE0 = 0;
  dword_4FE9FC8 = v0;
  qword_4FE9FE8 = 0;
  qword_4FE9FF0 = 0;
  qword_4FE9FF8 = 0;
  qword_4FEA000 = 0;
  qword_4FEA008 = (__int64)&unk_4FEA018;
  qword_4FEA020 = 0;
  qword_4FEA028 = (__int64)&unk_4FEA040;
  qword_4FEA030 = 1;
  dword_4FEA038 = 0;
  byte_4FEA03C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FEA010;
  v3 = (unsigned int)qword_4FEA010 + 1LL;
  if ( v3 > HIDWORD(qword_4FEA010) )
  {
    sub_C8D5F0((char *)&unk_4FEA018 - 16, &unk_4FEA018, v3, 8);
    v2 = (unsigned int)qword_4FEA010;
  }
  *(_QWORD *)(qword_4FEA008 + 8 * v2) = v1;
  LODWORD(qword_4FEA010) = qword_4FEA010 + 1;
  qword_4FEA048 = 0;
  qword_4FEA050 = (__int64)&unk_49D9748;
  qword_4FEA058 = 0;
  qword_4FE9FC0 = (__int64)&unk_49DC090;
  qword_4FEA060 = (__int64)&unk_49DC1D0;
  qword_4FEA080 = (__int64)nullsub_23;
  qword_4FEA078 = (__int64)sub_984030;
  sub_C53080(&qword_4FE9FC0, "memprof-guard-against-version-mismatch", 38);
  qword_4FE9FE8 = (__int64)"Guard against compiler/runtime version mismatch.";
  LOWORD(qword_4FEA058) = 257;
  LOBYTE(qword_4FEA048) = 1;
  qword_4FE9FF0 = 48;
  LOBYTE(dword_4FE9FCC) = dword_4FE9FCC & 0x9F | 0x20;
  sub_C53130(&qword_4FE9FC0);
  __cxa_atexit(sub_984900, &qword_4FE9FC0, &qword_4A427C0);
  v54 = &v52;
  v55 = "instrument read instructions";
  LOBYTE(v52) = 1;
  v53 = 1;
  v56 = 28;
  sub_243B8A0(&unk_4FE9EE0, "memprof-instrument-reads", &v55, &v53, &v54);
  __cxa_atexit(sub_984900, &unk_4FE9EE0, &qword_4A427C0);
  qword_4FE9E00 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE9E50 = 0x100000000LL;
  dword_4FE9E0C &= 0x8000u;
  qword_4FE9E48 = (__int64)&unk_4FE9E58;
  word_4FE9E10 = 0;
  qword_4FE9E18 = 0;
  dword_4FE9E08 = v4;
  qword_4FE9E20 = 0;
  qword_4FE9E28 = 0;
  qword_4FE9E30 = 0;
  qword_4FE9E38 = 0;
  qword_4FE9E40 = 0;
  qword_4FE9E60 = 0;
  qword_4FE9E68 = (__int64)&unk_4FE9E80;
  qword_4FE9E70 = 1;
  dword_4FE9E78 = 0;
  byte_4FE9E7C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FE9E50;
  v7 = (unsigned int)qword_4FE9E50 + 1LL;
  if ( v7 > HIDWORD(qword_4FE9E50) )
  {
    sub_C8D5F0((char *)&unk_4FE9E58 - 16, &unk_4FE9E58, v7, 8);
    v6 = (unsigned int)qword_4FE9E50;
  }
  *(_QWORD *)(qword_4FE9E48 + 8 * v6) = v5;
  LODWORD(qword_4FE9E50) = qword_4FE9E50 + 1;
  qword_4FE9E88 = 0;
  qword_4FE9E90 = (__int64)&unk_49D9748;
  qword_4FE9E98 = 0;
  qword_4FE9E00 = (__int64)&unk_49DC090;
  qword_4FE9EA0 = (__int64)&unk_49DC1D0;
  qword_4FE9EC0 = (__int64)nullsub_23;
  qword_4FE9EB8 = (__int64)sub_984030;
  sub_C53080(&qword_4FE9E00, "memprof-instrument-writes", 25);
  qword_4FE9E28 = (__int64)"instrument write instructions";
  LOWORD(qword_4FE9E98) = 257;
  LOBYTE(qword_4FE9E88) = 1;
  qword_4FE9E30 = 29;
  LOBYTE(dword_4FE9E0C) = dword_4FE9E0C & 0x9F | 0x20;
  sub_C53130(&qword_4FE9E00);
  __cxa_atexit(sub_984900, &qword_4FE9E00, &qword_4A427C0);
  LOBYTE(v52) = 1;
  v54 = &v52;
  v55 = "instrument atomic instructions (rmw, cmpxchg)";
  v53 = 1;
  v56 = 45;
  sub_248A750(&unk_4FE9D20, "memprof-instrument-atomics", &v55, &v53, &v54);
  __cxa_atexit(sub_984900, &unk_4FE9D20, &qword_4A427C0);
  qword_4FE9C40 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE9C90 = 0x100000000LL;
  dword_4FE9C4C &= 0x8000u;
  word_4FE9C50 = 0;
  qword_4FE9C88 = (__int64)&unk_4FE9C98;
  qword_4FE9C58 = 0;
  dword_4FE9C48 = v8;
  qword_4FE9C60 = 0;
  qword_4FE9C68 = 0;
  qword_4FE9C70 = 0;
  qword_4FE9C78 = 0;
  qword_4FE9C80 = 0;
  qword_4FE9CA0 = 0;
  qword_4FE9CA8 = (__int64)&unk_4FE9CC0;
  qword_4FE9CB0 = 1;
  dword_4FE9CB8 = 0;
  byte_4FE9CBC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FE9C90;
  v11 = (unsigned int)qword_4FE9C90 + 1LL;
  if ( v11 > HIDWORD(qword_4FE9C90) )
  {
    sub_C8D5F0((char *)&unk_4FE9C98 - 16, &unk_4FE9C98, v11, 8);
    v10 = (unsigned int)qword_4FE9C90;
  }
  *(_QWORD *)(qword_4FE9C88 + 8 * v10) = v9;
  LODWORD(qword_4FE9C90) = qword_4FE9C90 + 1;
  qword_4FE9CC8 = 0;
  qword_4FE9CD0 = (__int64)&unk_49D9748;
  qword_4FE9CD8 = 0;
  qword_4FE9C40 = (__int64)&unk_49DC090;
  qword_4FE9CE0 = (__int64)&unk_49DC1D0;
  qword_4FE9D00 = (__int64)nullsub_23;
  qword_4FE9CF8 = (__int64)sub_984030;
  sub_C53080(&qword_4FE9C40, "memprof-use-callbacks", 21);
  qword_4FE9C70 = 58;
  qword_4FE9C68 = (__int64)"Use callbacks instead of inline instrumentation sequences.";
  LOBYTE(qword_4FE9CC8) = 0;
  LOBYTE(dword_4FE9C4C) = dword_4FE9C4C & 0x9F | 0x20;
  LOWORD(qword_4FE9CD8) = 256;
  sub_C53130(&qword_4FE9C40);
  __cxa_atexit(sub_984900, &qword_4FE9C40, &qword_4A427C0);
  qword_4FE9B40 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FE9B4C &= 0x8000u;
  word_4FE9B50 = 0;
  qword_4FE9B90 = 0x100000000LL;
  qword_4FE9B88 = (__int64)&unk_4FE9B98;
  qword_4FE9B58 = 0;
  qword_4FE9B60 = 0;
  dword_4FE9B48 = v12;
  qword_4FE9B68 = 0;
  qword_4FE9B70 = 0;
  qword_4FE9B78 = 0;
  qword_4FE9B80 = 0;
  qword_4FE9BA0 = 0;
  qword_4FE9BA8 = (__int64)&unk_4FE9BC0;
  qword_4FE9BB0 = 1;
  dword_4FE9BB8 = 0;
  byte_4FE9BBC = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_4FE9B90;
  v15 = (unsigned int)qword_4FE9B90 + 1LL;
  if ( v15 > HIDWORD(qword_4FE9B90) )
  {
    sub_C8D5F0((char *)&unk_4FE9B98 - 16, &unk_4FE9B98, v15, 8);
    v14 = (unsigned int)qword_4FE9B90;
  }
  *(_QWORD *)(qword_4FE9B88 + 8 * v14) = v13;
  qword_4FE9BC8 = (__int64)&byte_4FE9BD8;
  qword_4FE9BF0 = (__int64)&byte_4FE9C00;
  LODWORD(qword_4FE9B90) = qword_4FE9B90 + 1;
  qword_4FE9BD0 = 0;
  qword_4FE9BE8 = (__int64)&unk_49DC130;
  byte_4FE9BD8 = 0;
  byte_4FE9C00 = 0;
  qword_4FE9B40 = (__int64)&unk_49DC010;
  qword_4FE9BF8 = 0;
  byte_4FE9C10 = 0;
  qword_4FE9C18 = (__int64)&unk_49DC350;
  qword_4FE9C38 = (__int64)nullsub_92;
  qword_4FE9C30 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FE9B40, "memprof-memory-access-callback-prefix", 37);
  qword_4FE9B68 = (__int64)"Prefix for memory access callbacks";
  qword_4FE9B70 = 34;
  v55 = v57;
  v56 = 10;
  LOBYTE(dword_4FE9B4C) = dword_4FE9B4C & 0x9F | 0x20;
  strcpy(v57, "__memprof_");
  sub_2240AE0(&qword_4FE9BC8, &v55);
  byte_4FE9C10 = 1;
  sub_2240AE0(&qword_4FE9BF0, &v55);
  if ( v55 != v57 )
    j_j___libc_free_0(v55, *(_QWORD *)v57 + 1LL);
  sub_C53130(&qword_4FE9B40);
  __cxa_atexit(sub_BC5A40, &qword_4FE9B40, &qword_4A427C0);
  qword_4FE9A60 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FE9A6C &= 0x8000u;
  word_4FE9A70 = 0;
  qword_4FE9AB0 = 0x100000000LL;
  qword_4FE9AA8 = (__int64)&unk_4FE9AB8;
  qword_4FE9A78 = 0;
  qword_4FE9A80 = 0;
  dword_4FE9A68 = v16;
  qword_4FE9A88 = 0;
  qword_4FE9A90 = 0;
  qword_4FE9A98 = 0;
  qword_4FE9AA0 = 0;
  qword_4FE9AC0 = 0;
  qword_4FE9AC8 = (__int64)&unk_4FE9AE0;
  qword_4FE9AD0 = 1;
  dword_4FE9AD8 = 0;
  byte_4FE9ADC = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_4FE9AB0;
  v19 = (unsigned int)qword_4FE9AB0 + 1LL;
  if ( v19 > HIDWORD(qword_4FE9AB0) )
  {
    sub_C8D5F0((char *)&unk_4FE9AB8 - 16, &unk_4FE9AB8, v19, 8);
    v18 = (unsigned int)qword_4FE9AB0;
  }
  *(_QWORD *)(qword_4FE9AA8 + 8 * v18) = v17;
  LODWORD(qword_4FE9AB0) = qword_4FE9AB0 + 1;
  qword_4FE9AE8 = 0;
  qword_4FE9AF0 = (__int64)&unk_49DA090;
  qword_4FE9AF8 = 0;
  qword_4FE9A60 = (__int64)&unk_49DBF90;
  qword_4FE9B00 = (__int64)&unk_49DC230;
  qword_4FE9B20 = (__int64)nullsub_58;
  qword_4FE9B18 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FE9A60, "memprof-mapping-scale", 21);
  qword_4FE9A90 = 31;
  qword_4FE9A88 = (__int64)"scale of memprof shadow mapping";
  LODWORD(qword_4FE9AE8) = 3;
  BYTE4(qword_4FE9AF8) = 1;
  LODWORD(qword_4FE9AF8) = 3;
  LOBYTE(dword_4FE9A6C) = dword_4FE9A6C & 0x9F | 0x20;
  sub_C53130(&qword_4FE9A60);
  __cxa_atexit(sub_B2B680, &qword_4FE9A60, &qword_4A427C0);
  qword_4FE9980 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FE998C &= 0x8000u;
  word_4FE9990 = 0;
  qword_4FE99D0 = 0x100000000LL;
  qword_4FE99C8 = (__int64)&unk_4FE99D8;
  qword_4FE9998 = 0;
  qword_4FE99A0 = 0;
  dword_4FE9988 = v20;
  qword_4FE99A8 = 0;
  qword_4FE99B0 = 0;
  qword_4FE99B8 = 0;
  qword_4FE99C0 = 0;
  qword_4FE99E0 = 0;
  qword_4FE99E8 = (__int64)&unk_4FE9A00;
  qword_4FE99F0 = 1;
  dword_4FE99F8 = 0;
  byte_4FE99FC = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_4FE99D0;
  if ( (unsigned __int64)(unsigned int)qword_4FE99D0 + 1 > HIDWORD(qword_4FE99D0) )
  {
    v48 = v21;
    sub_C8D5F0((char *)&unk_4FE99D8 - 16, &unk_4FE99D8, (unsigned int)qword_4FE99D0 + 1LL, 8);
    v22 = (unsigned int)qword_4FE99D0;
    v21 = v48;
  }
  *(_QWORD *)(qword_4FE99C8 + 8 * v22) = v21;
  LODWORD(qword_4FE99D0) = qword_4FE99D0 + 1;
  qword_4FE9A08 = 0;
  qword_4FE9A10 = (__int64)&unk_49DA090;
  qword_4FE9A18 = 0;
  qword_4FE9980 = (__int64)&unk_49DBF90;
  qword_4FE9A20 = (__int64)&unk_49DC230;
  qword_4FE9A40 = (__int64)nullsub_58;
  qword_4FE9A38 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FE9980, "memprof-mapping-granularity", 27);
  qword_4FE99B0 = 37;
  qword_4FE99A8 = (__int64)"granularity of memprof shadow mapping";
  LODWORD(qword_4FE9A08) = 64;
  BYTE4(qword_4FE9A18) = 1;
  LODWORD(qword_4FE9A18) = 64;
  LOBYTE(dword_4FE998C) = dword_4FE998C & 0x9F | 0x20;
  sub_C53130(&qword_4FE9980);
  __cxa_atexit(sub_B2B680, &qword_4FE9980, &qword_4A427C0);
  LOBYTE(v52) = 0;
  v54 = &v52;
  v55 = "Instrument scalar stack variables";
  v53 = 1;
  v56 = 33;
  sub_243B8A0(&unk_4FE98A0, "memprof-instrument-stack", &v55, &v53, &v54);
  __cxa_atexit(sub_984900, &unk_4FE98A0, &qword_4A427C0);
  qword_4FE97C0 = (__int64)&unk_49DC150;
  v23 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE9810 = 0x100000000LL;
  dword_4FE97CC &= 0x8000u;
  qword_4FE9808 = (__int64)&unk_4FE9818;
  word_4FE97D0 = 0;
  qword_4FE97D8 = 0;
  dword_4FE97C8 = v23;
  qword_4FE97E0 = 0;
  qword_4FE97E8 = 0;
  qword_4FE97F0 = 0;
  qword_4FE97F8 = 0;
  qword_4FE9800 = 0;
  qword_4FE9820 = 0;
  qword_4FE9828 = (__int64)&unk_4FE9840;
  qword_4FE9830 = 1;
  dword_4FE9838 = 0;
  byte_4FE983C = 1;
  v24 = sub_C57470();
  v25 = (unsigned int)qword_4FE9810;
  if ( (unsigned __int64)(unsigned int)qword_4FE9810 + 1 > HIDWORD(qword_4FE9810) )
  {
    v49 = v24;
    sub_C8D5F0((char *)&unk_4FE9818 - 16, &unk_4FE9818, (unsigned int)qword_4FE9810 + 1LL, 8);
    v25 = (unsigned int)qword_4FE9810;
    v24 = v49;
  }
  *(_QWORD *)(qword_4FE9808 + 8 * v25) = v24;
  LODWORD(qword_4FE9810) = qword_4FE9810 + 1;
  qword_4FE9848 = 0;
  qword_4FE9850 = (__int64)&unk_49DA090;
  qword_4FE9858 = 0;
  qword_4FE97C0 = (__int64)&unk_49DBF90;
  qword_4FE9860 = (__int64)&unk_49DC230;
  qword_4FE9880 = (__int64)nullsub_58;
  qword_4FE9878 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FE97C0, "memprof-debug", 13);
  qword_4FE97F0 = 5;
  qword_4FE97E8 = (__int64)"debug";
  LODWORD(qword_4FE9848) = 0;
  BYTE4(qword_4FE9858) = 1;
  LODWORD(qword_4FE9858) = 0;
  LOBYTE(dword_4FE97CC) = dword_4FE97CC & 0x9F | 0x20;
  sub_C53130(&qword_4FE97C0);
  __cxa_atexit(sub_B2B680, &qword_4FE97C0, &qword_4A427C0);
  qword_4FE96C0 = (__int64)&unk_49DC150;
  v26 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE9710 = 0x100000000LL;
  dword_4FE96CC &= 0x8000u;
  qword_4FE9708 = (__int64)&unk_4FE9718;
  word_4FE96D0 = 0;
  qword_4FE96D8 = 0;
  dword_4FE96C8 = v26;
  qword_4FE96E0 = 0;
  qword_4FE96E8 = 0;
  qword_4FE96F0 = 0;
  qword_4FE96F8 = 0;
  qword_4FE9700 = 0;
  qword_4FE9720 = 0;
  qword_4FE9728 = (__int64)&unk_4FE9740;
  qword_4FE9730 = 1;
  dword_4FE9738 = 0;
  byte_4FE973C = 1;
  v27 = sub_C57470();
  v28 = (unsigned int)qword_4FE9710;
  if ( (unsigned __int64)(unsigned int)qword_4FE9710 + 1 > HIDWORD(qword_4FE9710) )
  {
    v50 = v27;
    sub_C8D5F0((char *)&unk_4FE9718 - 16, &unk_4FE9718, (unsigned int)qword_4FE9710 + 1LL, 8);
    v28 = (unsigned int)qword_4FE9710;
    v27 = v50;
  }
  *(_QWORD *)(qword_4FE9708 + 8 * v28) = v27;
  qword_4FE9748 = &byte_4FE9758;
  qword_4FE9770 = (__int64)&byte_4FE9780;
  LODWORD(qword_4FE9710) = qword_4FE9710 + 1;
  qword_4FE9750 = 0;
  qword_4FE9768 = (__int64)&unk_49DC130;
  byte_4FE9758 = 0;
  byte_4FE9780 = 0;
  qword_4FE96C0 = (__int64)&unk_49DC010;
  qword_4FE9778 = 0;
  byte_4FE9790 = 0;
  qword_4FE9798 = (__int64)&unk_49DC350;
  qword_4FE97B8 = (__int64)nullsub_92;
  qword_4FE97B0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FE96C0, "memprof-debug-func", 18);
  qword_4FE96F0 = 10;
  LOBYTE(dword_4FE96CC) = dword_4FE96CC & 0x9F | 0x20;
  qword_4FE96E8 = (__int64)"Debug func";
  sub_C53130(&qword_4FE96C0);
  __cxa_atexit(sub_BC5A40, &qword_4FE96C0, &qword_4A427C0);
  v54 = &v53;
  v55 = "Debug min inst";
  v53 = -1;
  v52 = 1;
  v56 = 14;
  sub_248A960(&unk_4FE95E0, "memprof-debug-min", &v55, &v52, &v54);
  __cxa_atexit(sub_B2B680, &unk_4FE95E0, &qword_4A427C0);
  v54 = &v53;
  v55 = "Debug max inst";
  v53 = -1;
  v52 = 1;
  v56 = 14;
  sub_248A960(&unk_4FE9500, "memprof-debug-max", &v55, &v52, &v54);
  __cxa_atexit(sub_B2B680, &unk_4FE9500, &qword_4A427C0);
  LOBYTE(v52) = 0;
  v54 = &v52;
  v55 = "Match allocation profiles onto existing hot/cold operator new calls";
  v53 = 1;
  v56 = 67;
  sub_248A750(&unk_4FE9420, "memprof-match-hot-cold-new", &v55, &v53, &v54);
  __cxa_atexit(sub_984900, &unk_4FE9420, &qword_4A427C0);
  qword_4FE9340 = (__int64)&unk_49DC150;
  v29 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE9390 = 0x100000000LL;
  dword_4FE934C &= 0x8000u;
  qword_4FE9388 = (__int64)&unk_4FE9398;
  word_4FE9350 = 0;
  qword_4FE9358 = 0;
  dword_4FE9348 = v29;
  qword_4FE9360 = 0;
  qword_4FE9368 = 0;
  qword_4FE9370 = 0;
  qword_4FE9378 = 0;
  qword_4FE9380 = 0;
  qword_4FE93A0 = 0;
  qword_4FE93A8 = (__int64)&unk_4FE93C0;
  qword_4FE93B0 = 1;
  dword_4FE93B8 = 0;
  byte_4FE93BC = 1;
  v30 = sub_C57470();
  v31 = (unsigned int)qword_4FE9390;
  v32 = (unsigned int)qword_4FE9390 + 1LL;
  if ( v32 > HIDWORD(qword_4FE9390) )
  {
    sub_C8D5F0((char *)&unk_4FE9398 - 16, &unk_4FE9398, v32, 8);
    v31 = (unsigned int)qword_4FE9390;
  }
  *(_QWORD *)(qword_4FE9388 + 8 * v31) = v30;
  LODWORD(qword_4FE9390) = qword_4FE9390 + 1;
  qword_4FE93C8 = 0;
  qword_4FE93D0 = (__int64)&unk_49D9748;
  qword_4FE93D8 = 0;
  qword_4FE9340 = (__int64)&unk_49DC090;
  qword_4FE93E0 = (__int64)&unk_49DC1D0;
  qword_4FE9400 = (__int64)nullsub_23;
  qword_4FE93F8 = (__int64)sub_984030;
  sub_C53080(&qword_4FE9340, "memprof-histogram", 17);
  qword_4FE9370 = 31;
  qword_4FE9368 = (__int64)"Collect access count histograms";
  LOWORD(qword_4FE93D8) = 256;
  LOBYTE(qword_4FE93C8) = 0;
  LOBYTE(dword_4FE934C) = dword_4FE934C & 0x9F | 0x20;
  sub_C53130(&qword_4FE9340);
  __cxa_atexit(sub_984900, &qword_4FE9340, &qword_4A427C0);
  LOBYTE(v52) = 0;
  v54 = &v52;
  v55 = "Print matching stats for each allocation context in this module's profiles";
  v53 = 1;
  v56 = 74;
  sub_243B8A0(&unk_4FE9260, "memprof-print-match-info", &v55, &v53, &v54);
  __cxa_atexit(sub_984900, &unk_4FE9260, &qword_4A427C0);
  qword_4FE9160 = (__int64)&unk_49DC150;
  v33 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE91B0 = 0x100000000LL;
  dword_4FE916C &= 0x8000u;
  word_4FE9170 = 0;
  qword_4FE9178 = 0;
  qword_4FE9180 = 0;
  dword_4FE9168 = v33;
  qword_4FE9188 = 0;
  qword_4FE9190 = 0;
  qword_4FE9198 = 0;
  qword_4FE91A0 = 0;
  qword_4FE91A8 = (__int64)&unk_4FE91B8;
  qword_4FE91C0 = 0;
  qword_4FE91C8 = (__int64)&unk_4FE91E0;
  qword_4FE91D0 = 1;
  dword_4FE91D8 = 0;
  byte_4FE91DC = 1;
  v34 = sub_C57470();
  v35 = (unsigned int)qword_4FE91B0;
  v36 = (unsigned int)qword_4FE91B0 + 1LL;
  if ( v36 > HIDWORD(qword_4FE91B0) )
  {
    sub_C8D5F0((char *)&unk_4FE91B8 - 16, &unk_4FE91B8, v36, 8);
    v35 = (unsigned int)qword_4FE91B0;
  }
  *(_QWORD *)(qword_4FE91A8 + 8 * v35) = v34;
  qword_4FE91E8 = (__int64)&byte_4FE91F8;
  qword_4FE9210 = (__int64)&byte_4FE9220;
  LODWORD(qword_4FE91B0) = qword_4FE91B0 + 1;
  qword_4FE91F0 = 0;
  qword_4FE9208 = (__int64)&unk_49DC130;
  byte_4FE91F8 = 0;
  byte_4FE9220 = 0;
  qword_4FE9160 = (__int64)&unk_49DC010;
  qword_4FE9218 = 0;
  byte_4FE9230 = 0;
  qword_4FE9238 = (__int64)&unk_49DC350;
  qword_4FE9258 = (__int64)nullsub_92;
  qword_4FE9250 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FE9160, "memprof-runtime-default-options", 31);
  v57[0] = 0;
  qword_4FE9190 = 27;
  qword_4FE9188 = (__int64)"The default memprof options";
  v55 = v57;
  v56 = 0;
  LOBYTE(dword_4FE916C) = dword_4FE916C & 0x9F | 0x20;
  sub_2240AE0(&qword_4FE91E8, &v55);
  byte_4FE9230 = 1;
  sub_2240AE0(&qword_4FE9210, &v55);
  if ( v55 != v57 )
    j_j___libc_free_0(v55, *(_QWORD *)v57 + 1LL);
  sub_C53130(&qword_4FE9160);
  __cxa_atexit(sub_BC5A40, &qword_4FE9160, &qword_4A427C0);
  qword_4FE9080 = (__int64)&unk_49DC150;
  v37 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE90D0 = 0x100000000LL;
  dword_4FE908C &= 0x8000u;
  word_4FE9090 = 0;
  qword_4FE9098 = 0;
  qword_4FE90A0 = 0;
  dword_4FE9088 = v37;
  qword_4FE90A8 = 0;
  qword_4FE90B0 = 0;
  qword_4FE90B8 = 0;
  qword_4FE90C0 = 0;
  qword_4FE90C8 = (__int64)&unk_4FE90D8;
  qword_4FE90E0 = 0;
  qword_4FE90E8 = (__int64)&unk_4FE9100;
  qword_4FE90F0 = 1;
  dword_4FE90F8 = 0;
  byte_4FE90FC = 1;
  v38 = sub_C57470();
  v39 = (unsigned int)qword_4FE90D0;
  v40 = (unsigned int)qword_4FE90D0 + 1LL;
  if ( v40 > HIDWORD(qword_4FE90D0) )
  {
    sub_C8D5F0((char *)&unk_4FE90D8 - 16, &unk_4FE90D8, v40, 8);
    v39 = (unsigned int)qword_4FE90D0;
  }
  *(_QWORD *)(qword_4FE90C8 + 8 * v39) = v38;
  LODWORD(qword_4FE90D0) = qword_4FE90D0 + 1;
  qword_4FE9108 = 0;
  qword_4FE9110 = (__int64)&unk_49D9748;
  qword_4FE9118 = 0;
  qword_4FE9080 = (__int64)&unk_49DC090;
  qword_4FE9120 = (__int64)&unk_49DC1D0;
  qword_4FE9140 = (__int64)nullsub_23;
  qword_4FE9138 = (__int64)sub_984030;
  sub_C53080(&qword_4FE9080, "memprof-salvage-stale-profile", 29);
  qword_4FE90A8 = (__int64)"Salvage stale MemProf profile";
  LOWORD(qword_4FE9118) = 256;
  LOBYTE(qword_4FE9108) = 0;
  qword_4FE90B0 = 29;
  LOBYTE(dword_4FE908C) = dword_4FE908C & 0x9F | 0x20;
  sub_C53130(&qword_4FE9080);
  __cxa_atexit(sub_984900, &qword_4FE9080, &qword_4A427C0);
  qword_4FE8FA0 = &unk_49DC150;
  v41 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4FE8FAC = word_4FE8FAC & 0x8000;
  qword_4FE8FE8[1] = 0x100000000LL;
  unk_4FE8FA8 = v41;
  unk_4FE8FB0 = 0;
  unk_4FE8FB8 = 0;
  unk_4FE8FC0 = 0;
  unk_4FE8FC8 = 0;
  unk_4FE8FD0 = 0;
  unk_4FE8FD8 = 0;
  unk_4FE8FE0 = 0;
  qword_4FE8FE8[0] = &qword_4FE8FE8[2];
  qword_4FE8FE8[3] = 0;
  qword_4FE8FE8[4] = &qword_4FE8FE8[7];
  qword_4FE8FE8[5] = 1;
  LODWORD(qword_4FE8FE8[6]) = 0;
  BYTE4(qword_4FE8FE8[6]) = 1;
  v42 = sub_C57470();
  v43 = LODWORD(qword_4FE8FE8[1]);
  if ( (unsigned __int64)LODWORD(qword_4FE8FE8[1]) + 1 > HIDWORD(qword_4FE8FE8[1]) )
  {
    sub_C8D5F0(qword_4FE8FE8, &qword_4FE8FE8[2], LODWORD(qword_4FE8FE8[1]) + 1LL, 8);
    v43 = LODWORD(qword_4FE8FE8[1]);
  }
  *(_QWORD *)(qword_4FE8FE8[0] + 8 * v43) = v42;
  qword_4FE8FE8[9] = &unk_49D9728;
  qword_4FE8FA0 = &unk_49DBF10;
  ++LODWORD(qword_4FE8FE8[1]);
  qword_4FE8FE8[15] = nullsub_24;
  qword_4FE8FE8[11] = &unk_49DC290;
  qword_4FE8FE8[8] = 0;
  qword_4FE8FE8[14] = sub_984050;
  qword_4FE8FE8[10] = 0;
  sub_C53080(&qword_4FE8FA0, "memprof-cloning-cold-threshold", 30);
  LODWORD(qword_4FE8FE8[8]) = 100;
  BYTE4(qword_4FE8FE8[10]) = 1;
  LODWORD(qword_4FE8FE8[10]) = 100;
  unk_4FE8FD0 = 59;
  LOBYTE(word_4FE8FAC) = word_4FE8FAC & 0x9F | 0x20;
  unk_4FE8FC8 = "Min percent of cold bytes to hint alloc cold during cloning";
  sub_C53130(&qword_4FE8FA0);
  __cxa_atexit(sub_984970, &qword_4FE8FA0, &qword_4A427C0);
  qword_4FE8EC0 = (__int64)&unk_49DC150;
  v44 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FE8ECC &= 0x8000u;
  word_4FE8ED0 = 0;
  qword_4FE8F10 = 0x100000000LL;
  qword_4FE8ED8 = 0;
  qword_4FE8EE0 = 0;
  qword_4FE8EE8 = 0;
  dword_4FE8EC8 = v44;
  qword_4FE8EF0 = 0;
  qword_4FE8EF8 = 0;
  qword_4FE8F00 = 0;
  qword_4FE8F08 = (__int64)&unk_4FE8F18;
  qword_4FE8F20 = 0;
  qword_4FE8F28 = (__int64)&unk_4FE8F40;
  qword_4FE8F30 = 1;
  dword_4FE8F38 = 0;
  byte_4FE8F3C = 1;
  v45 = sub_C57470();
  v46 = (unsigned int)qword_4FE8F10;
  if ( (unsigned __int64)(unsigned int)qword_4FE8F10 + 1 > HIDWORD(qword_4FE8F10) )
  {
    v51 = v45;
    sub_C8D5F0((char *)&unk_4FE8F18 - 16, &unk_4FE8F18, (unsigned int)qword_4FE8F10 + 1LL, 8);
    v46 = (unsigned int)qword_4FE8F10;
    v45 = v51;
  }
  *(_QWORD *)(qword_4FE8F08 + 8 * v46) = v45;
  qword_4FE8F50 = (__int64)&unk_49D9728;
  qword_4FE8EC0 = (__int64)&unk_49DBF10;
  qword_4FE8F80 = (__int64)nullsub_24;
  LODWORD(qword_4FE8F10) = qword_4FE8F10 + 1;
  qword_4FE8F60 = (__int64)&unk_49DC290;
  qword_4FE8F48 = 0;
  qword_4FE8F78 = (__int64)sub_984050;
  qword_4FE8F58 = 0;
  sub_C53080(&qword_4FE8EC0, "memprof-matching-cold-threshold", 31);
  LODWORD(qword_4FE8F48) = 100;
  BYTE4(qword_4FE8F58) = 1;
  LODWORD(qword_4FE8F58) = 100;
  qword_4FE8EF0 = 57;
  LOBYTE(dword_4FE8ECC) = dword_4FE8ECC & 0x9F | 0x20;
  qword_4FE8EE8 = (__int64)"Min percent of cold bytes matched to hint allocation cold";
  sub_C53130(&qword_4FE8EC0);
  return __cxa_atexit(sub_984970, &qword_4FE8EC0, &qword_4A427C0);
}
