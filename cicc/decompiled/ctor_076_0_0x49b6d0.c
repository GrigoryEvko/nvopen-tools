// Function: ctor_076_0
// Address: 0x49b6d0
//
int ctor_076_0()
{
  int v0; // edx
  __int64 v1; // rax
  __int64 v2; // rdx
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v14; // [rsp+8h] [rbp-108h]
  __int64 v15; // [rsp+8h] [rbp-108h]
  int v16; // [rsp+1Ch] [rbp-F4h] BYREF
  const char *v17; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v18; // [rsp+28h] [rbp-E8h]
  char **v19; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v20; // [rsp+38h] [rbp-D8h]
  char *v21; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v22; // [rsp+48h] [rbp-C8h]
  int v23; // [rsp+50h] [rbp-C0h]
  const char *v24; // [rsp+58h] [rbp-B8h]
  __int64 v25; // [rsp+60h] [rbp-B0h]
  char *v26; // [rsp+68h] [rbp-A8h]
  __int64 v27; // [rsp+70h] [rbp-A0h]
  int v28; // [rsp+78h] [rbp-98h]
  const char *v29; // [rsp+80h] [rbp-90h]
  __int64 v30; // [rsp+88h] [rbp-88h]
  char *v31; // [rsp+90h] [rbp-80h]
  __int64 v32; // [rsp+98h] [rbp-78h]
  int v33; // [rsp+A0h] [rbp-70h]
  const char *v34; // [rsp+A8h] [rbp-68h]
  __int64 v35; // [rsp+B0h] [rbp-60h]
  char *v36; // [rsp+B8h] [rbp-58h]
  __int64 v37; // [rsp+C0h] [rbp-50h]
  int v38; // [rsp+C8h] [rbp-48h]
  const char *v39; // [rsp+D0h] [rbp-40h]
  __int64 v40; // [rsp+D8h] [rbp-38h]

  v21 = "none";
  v24 = "do not display graphs.";
  v26 = "fraction";
  v29 = "display a graph using the fractional block frequency representation.";
  v31 = "integer";
  v34 = "display a graph using the raw integer fractional block frequency representation.";
  v36 = "count";
  v39 = "display a graph using the real profile count if available.";
  v20 = 0x400000004LL;
  v19 = &v21;
  v22 = 4;
  v23 = 0;
  v25 = 22;
  v27 = 8;
  v28 = 1;
  v30 = 68;
  v32 = 7;
  v33 = 2;
  v35 = 80;
  v37 = 5;
  v38 = 3;
  v40 = 58;
  v17 = "Pop up a window to show a dag displaying how block frequencies propagation through the CFG.";
  v18 = 91;
  v16 = 1;
  sub_FE6E00(&unk_4F8DFE0, "view-block-freq-propagation-dags", &v16, &v17, &v19);
  if ( v19 != &v21 )
    _libc_free(v19, "view-block-freq-propagation-dags");
  __cxa_atexit(sub_FDB740, &unk_4F8DFE0, &qword_4A427C0);
  qword_4F8DEE0 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8DEEC = word_4F8DEEC & 0x8000;
  qword_4F8DF28[1] = 0x100000000LL;
  unk_4F8DEE8 = v0;
  qword_4F8DF28[0] = &qword_4F8DF28[2];
  unk_4F8DEF0 = 0;
  unk_4F8DEF8 = 0;
  unk_4F8DF00 = 0;
  unk_4F8DF08 = 0;
  unk_4F8DF10 = 0;
  unk_4F8DF18 = 0;
  unk_4F8DF20 = 0;
  qword_4F8DF28[3] = 0;
  qword_4F8DF28[4] = &qword_4F8DF28[7];
  qword_4F8DF28[5] = 1;
  LODWORD(qword_4F8DF28[6]) = 0;
  BYTE4(qword_4F8DF28[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F8DF28[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8DF28[1]) + 1 > HIDWORD(qword_4F8DF28[1]) )
  {
    v14 = v1;
    sub_C8D5F0(qword_4F8DF28, &qword_4F8DF28[2], LODWORD(qword_4F8DF28[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F8DF28[1]);
    v1 = v14;
  }
  *(_QWORD *)(qword_4F8DF28[0] + 8 * v2) = v1;
  qword_4F8DF28[8] = &qword_4F8DF28[10];
  qword_4F8DF28[13] = &qword_4F8DF28[15];
  ++LODWORD(qword_4F8DF28[1]);
  qword_4F8DF28[9] = 0;
  qword_4F8DF28[12] = &unk_49DC130;
  LOBYTE(qword_4F8DF28[10]) = 0;
  LOBYTE(qword_4F8DF28[15]) = 0;
  qword_4F8DEE0 = &unk_49DC010;
  qword_4F8DF28[14] = 0;
  LOBYTE(qword_4F8DF28[17]) = 0;
  qword_4F8DF28[18] = &unk_49DC350;
  qword_4F8DF28[22] = nullsub_92;
  qword_4F8DF28[21] = sub_BC4D70;
  sub_C53080(&qword_4F8DEE0, "view-bfi-func-name", 18);
  unk_4F8DF10 = 75;
  LOBYTE(word_4F8DEEC) = word_4F8DEEC & 0x9F | 0x20;
  unk_4F8DF08 = "The option to specify the name of the function whose CFG will be displayed.";
  sub_C53130(&qword_4F8DEE0);
  __cxa_atexit(sub_BC5A40, &qword_4F8DEE0, &qword_4A427C0);
  qword_4F8DE00 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8DE0C = word_4F8DE0C & 0x8000;
  unk_4F8DE10 = 0;
  qword_4F8DE48[1] = 0x100000000LL;
  unk_4F8DE08 = v3;
  qword_4F8DE48[0] = &qword_4F8DE48[2];
  unk_4F8DE18 = 0;
  unk_4F8DE20 = 0;
  unk_4F8DE28 = 0;
  unk_4F8DE30 = 0;
  unk_4F8DE38 = 0;
  unk_4F8DE40 = 0;
  qword_4F8DE48[3] = 0;
  qword_4F8DE48[4] = &qword_4F8DE48[7];
  qword_4F8DE48[5] = 1;
  LODWORD(qword_4F8DE48[6]) = 0;
  BYTE4(qword_4F8DE48[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_4F8DE48[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8DE48[1]) + 1 > HIDWORD(qword_4F8DE48[1]) )
  {
    v15 = v4;
    sub_C8D5F0(qword_4F8DE48, &qword_4F8DE48[2], LODWORD(qword_4F8DE48[1]) + 1LL, 8);
    v5 = LODWORD(qword_4F8DE48[1]);
    v4 = v15;
  }
  *(_QWORD *)(qword_4F8DE48[0] + 8 * v5) = v4;
  ++LODWORD(qword_4F8DE48[1]);
  qword_4F8DE48[8] = 0;
  qword_4F8DE48[9] = &unk_49D9728;
  qword_4F8DE48[10] = 0;
  qword_4F8DE00 = &unk_49DBF10;
  qword_4F8DE48[11] = &unk_49DC290;
  qword_4F8DE48[15] = nullsub_24;
  qword_4F8DE48[14] = sub_984050;
  sub_C53080(&qword_4F8DE00, "view-hot-freq-percent", 21);
  BYTE4(qword_4F8DE48[10]) = 1;
  LODWORD(qword_4F8DE48[8]) = 10;
  unk_4F8DE30 = 192;
  LODWORD(qword_4F8DE48[10]) = 10;
  LOBYTE(word_4F8DE0C) = word_4F8DE0C & 0x9F | 0x20;
  unk_4F8DE28 = "An integer in percent used to specify the hot blocks/edges to be displayed in red: a block or edge whose"
                " frequency is no less than the max frequency of the function multiplied by this percent.";
  sub_C53130(&qword_4F8DE00);
  __cxa_atexit(sub_984970, &qword_4F8DE00, &qword_4A427C0);
  v19 = &v21;
  v21 = "none";
  v24 = "do not show.";
  v26 = "graph";
  v29 = "show a graph.";
  v31 = "text";
  v34 = "show in text.";
  v20 = 0x400000003LL;
  v22 = 4;
  v23 = 0;
  v25 = 12;
  v27 = 5;
  v28 = 1;
  v30 = 13;
  v32 = 4;
  v33 = 2;
  v35 = 13;
  v17 = "A boolean option to show CFG dag or text with block profile counts and branch probabilities right after PGO prof"
        "ile annotation step. The profile counts are computed using branch probabilities from the runtime profile data an"
        "d block frequency propagation algorithm. To view the raw counts from the profile, use option -pgo-view-raw-count"
        "s instead. To limit graph display to only one function, use filtering option -view-bfi-func-name.";
  v18 = 433;
  v16 = 1;
  sub_FE7220(&unk_4F8DBA0, "pgo-view-counts", &v16, &v17, &v19);
  if ( v19 != &v21 )
    _libc_free(v19, "pgo-view-counts");
  __cxa_atexit(sub_FDB7D0, &unk_4F8DBA0, &qword_4A427C0);
  qword_4F8DAC0 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8DB10 = 0x100000000LL;
  word_4F8DAD0 = 0;
  dword_4F8DACC &= 0x8000u;
  qword_4F8DAD8 = 0;
  qword_4F8DAE0 = 0;
  dword_4F8DAC8 = v6;
  qword_4F8DAE8 = 0;
  qword_4F8DAF0 = 0;
  qword_4F8DAF8 = 0;
  qword_4F8DB00 = 0;
  qword_4F8DB08 = (__int64)&unk_4F8DB18;
  qword_4F8DB20 = 0;
  qword_4F8DB28 = (__int64)&unk_4F8DB40;
  qword_4F8DB30 = 1;
  dword_4F8DB38 = 0;
  byte_4F8DB3C = 1;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_4F8DB10;
  v9 = (unsigned int)qword_4F8DB10 + 1LL;
  if ( v9 > HIDWORD(qword_4F8DB10) )
  {
    sub_C8D5F0((char *)&unk_4F8DB18 - 16, &unk_4F8DB18, v9, 8);
    v8 = (unsigned int)qword_4F8DB10;
  }
  *(_QWORD *)(qword_4F8DB08 + 8 * v8) = v7;
  LODWORD(qword_4F8DB10) = qword_4F8DB10 + 1;
  qword_4F8DB48 = 0;
  qword_4F8DB50 = (__int64)&unk_49D9748;
  qword_4F8DB58 = 0;
  qword_4F8DAC0 = (__int64)&unk_49DC090;
  qword_4F8DB60 = (__int64)&unk_49DC1D0;
  qword_4F8DB80 = (__int64)nullsub_23;
  qword_4F8DB78 = (__int64)sub_984030;
  sub_C53080(&qword_4F8DAC0, "print-bfi", 9);
  LOBYTE(qword_4F8DB48) = 0;
  LOWORD(qword_4F8DB58) = 256;
  qword_4F8DAF0 = 31;
  LOBYTE(dword_4F8DACC) = dword_4F8DACC & 0x9F | 0x20;
  qword_4F8DAE8 = (__int64)"Print the block frequency info.";
  sub_C53130(&qword_4F8DAC0);
  __cxa_atexit(sub_984900, &qword_4F8DAC0, &qword_4A427C0);
  qword_4F8D9C0 = &unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8D9CC = word_4F8D9CC & 0x8000;
  unk_4F8D9C8 = v10;
  qword_4F8DA08[1] = 0x100000000LL;
  unk_4F8D9D0 = 0;
  unk_4F8D9D8 = 0;
  unk_4F8D9E0 = 0;
  unk_4F8D9E8 = 0;
  unk_4F8D9F0 = 0;
  unk_4F8D9F8 = 0;
  unk_4F8DA00 = 0;
  qword_4F8DA08[0] = &qword_4F8DA08[2];
  qword_4F8DA08[3] = 0;
  qword_4F8DA08[4] = &qword_4F8DA08[7];
  qword_4F8DA08[5] = 1;
  LODWORD(qword_4F8DA08[6]) = 0;
  BYTE4(qword_4F8DA08[6]) = 1;
  v11 = sub_C57470();
  v12 = LODWORD(qword_4F8DA08[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8DA08[1]) + 1 > HIDWORD(qword_4F8DA08[1]) )
  {
    sub_C8D5F0(qword_4F8DA08, &qword_4F8DA08[2], LODWORD(qword_4F8DA08[1]) + 1LL, 8);
    v12 = LODWORD(qword_4F8DA08[1]);
  }
  *(_QWORD *)(qword_4F8DA08[0] + 8 * v12) = v11;
  qword_4F8DA08[8] = &qword_4F8DA08[10];
  qword_4F8DA08[13] = &qword_4F8DA08[15];
  ++LODWORD(qword_4F8DA08[1]);
  qword_4F8DA08[9] = 0;
  qword_4F8DA08[12] = &unk_49DC130;
  LOBYTE(qword_4F8DA08[10]) = 0;
  qword_4F8DA08[14] = 0;
  qword_4F8D9C0 = &unk_49DC010;
  LOBYTE(qword_4F8DA08[15]) = 0;
  LOBYTE(qword_4F8DA08[17]) = 0;
  qword_4F8DA08[18] = &unk_49DC350;
  qword_4F8DA08[22] = nullsub_92;
  qword_4F8DA08[21] = sub_BC4D70;
  sub_C53080(&qword_4F8D9C0, "print-bfi-func-name", 19);
  unk_4F8D9F0 = 85;
  LOBYTE(word_4F8D9CC) = word_4F8D9CC & 0x9F | 0x20;
  unk_4F8D9E8 = "The option to specify the name of the function whose block frequency info is printed.";
  sub_C53130(&qword_4F8D9C0);
  return __cxa_atexit(sub_BC5A40, &qword_4F8D9C0, &qword_4A427C0);
}
