// Function: ctor_115_0
// Address: 0x4ab910
//
int ctor_115_0()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax
  int v3; // eax
  int v5; // [rsp+1Ch] [rbp-F4h] BYREF
  const char *v6; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v7; // [rsp+28h] [rbp-E8h]
  char **v8; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v9; // [rsp+38h] [rbp-D8h]
  char *v10; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v11; // [rsp+48h] [rbp-C8h]
  int v12; // [rsp+50h] [rbp-C0h]
  const char *v13; // [rsp+58h] [rbp-B8h]
  __int64 v14; // [rsp+60h] [rbp-B0h]
  char *v15; // [rsp+68h] [rbp-A8h]
  __int64 v16; // [rsp+70h] [rbp-A0h]
  int v17; // [rsp+78h] [rbp-98h]
  const char *v18; // [rsp+80h] [rbp-90h]
  __int64 v19; // [rsp+88h] [rbp-88h]
  char *v20; // [rsp+90h] [rbp-80h]
  __int64 v21; // [rsp+98h] [rbp-78h]
  int v22; // [rsp+A0h] [rbp-70h]
  const char *v23; // [rsp+A8h] [rbp-68h]
  __int64 v24; // [rsp+B0h] [rbp-60h]
  char *v25; // [rsp+B8h] [rbp-58h]
  __int64 v26; // [rsp+C0h] [rbp-50h]
  int v27; // [rsp+C8h] [rbp-48h]
  const char *v28; // [rsp+D0h] [rbp-40h]
  __int64 v29; // [rsp+D8h] [rbp-38h]

  v10 = "none";
  v13 = "do not display graphs.";
  v15 = "fraction";
  v18 = "display a graph using the fractional block frequency representation.";
  v20 = "integer";
  v23 = "display a graph using the raw integer fractional block frequency representation.";
  v25 = "count";
  v28 = "display a graph using the real profile count if available.";
  v9 = 0x400000004LL;
  v8 = &v10;
  v11 = 4;
  v12 = 0;
  v14 = 22;
  v16 = 8;
  v17 = 1;
  v19 = 68;
  v21 = 7;
  v22 = 2;
  v24 = 80;
  v26 = 5;
  v27 = 3;
  v29 = 58;
  v6 = "Pop up a window to show a dag displaying how block frequencies propagation through the CFG.";
  v7 = 91;
  v5 = 1;
  sub_1369090(&unk_4F984C0, "view-block-freq-propagation-dags", &v5, &v6, &v8);
  if ( v8 != &v10 )
    _libc_free(v8, "view-block-freq-propagation-dags");
  __cxa_atexit(sub_13678E0, &unk_4F984C0, &qword_4A427C0);
  qword_4F983A0[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4F983A0[1]) &= 0xF000u;
  LODWORD(qword_4F983A0[1]) = v0;
  qword_4F983A0[2] = 0;
  qword_4F983A0[9] = &unk_4FA01C0;
  qword_4F983A0[11] = &qword_4F983A0[15];
  qword_4F983A0[12] = &qword_4F983A0[15];
  qword_4F983A0[20] = &qword_4F983A0[22];
  qword_4F983A0[25] = &qword_4F983A0[27];
  qword_4F983A0[3] = 0;
  qword_4F983A0[4] = 0;
  qword_4F983A0[24] = &unk_49EED10;
  qword_4F983A0[5] = 0;
  qword_4F983A0[6] = 0;
  qword_4F983A0[0] = &unk_49EEBF0;
  qword_4F983A0[7] = 0;
  qword_4F983A0[8] = 0;
  qword_4F983A0[30] = &unk_49EEE90;
  qword_4F983A0[31] = &qword_4F983A0[33];
  qword_4F983A0[10] = 0;
  qword_4F983A0[13] = 4;
  LODWORD(qword_4F983A0[14]) = 0;
  LOBYTE(qword_4F983A0[19]) = 0;
  qword_4F983A0[21] = 0;
  LOBYTE(qword_4F983A0[22]) = 0;
  qword_4F983A0[26] = 0;
  LOBYTE(qword_4F983A0[27]) = 0;
  LOBYTE(qword_4F983A0[29]) = 0;
  qword_4F983A0[32] = 0;
  LOBYTE(qword_4F983A0[33]) = 0;
  sub_16B8280(qword_4F983A0, "view-bfi-func-name", 18);
  qword_4F983A0[6] = 75;
  BYTE4(qword_4F983A0[1]) = BYTE4(qword_4F983A0[1]) & 0x9F | 0x20;
  qword_4F983A0[5] = "The option to specify the name of the function whose CFG will be displayed.";
  sub_16B88A0(qword_4F983A0);
  __cxa_atexit(sub_12F0C20, qword_4F983A0, &qword_4A427C0);
  qword_4F982C0[0] = &unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4F982C0[1]) &= 0xF000u;
  LODWORD(qword_4F982C0[1]) = v1;
  qword_4F982C0[2] = 0;
  qword_4F982C0[9] = &unk_4FA01C0;
  qword_4F982C0[11] = &qword_4F982C0[15];
  qword_4F982C0[12] = &qword_4F982C0[15];
  qword_4F982C0[3] = 0;
  qword_4F982C0[4] = 0;
  qword_4F982C0[21] = &unk_49E74A8;
  qword_4F982C0[5] = 0;
  qword_4F982C0[6] = 0;
  qword_4F982C0[0] = &unk_49EEAF0;
  qword_4F982C0[7] = 0;
  qword_4F982C0[8] = 0;
  qword_4F982C0[23] = &unk_49EEE10;
  qword_4F982C0[10] = 0;
  qword_4F982C0[13] = 4;
  LODWORD(qword_4F982C0[14]) = 0;
  LOBYTE(qword_4F982C0[19]) = 0;
  LODWORD(qword_4F982C0[20]) = 0;
  BYTE4(qword_4F982C0[22]) = 1;
  LODWORD(qword_4F982C0[22]) = 0;
  sub_16B8280(qword_4F982C0, "view-hot-freq-percent", 21);
  LODWORD(qword_4F982C0[20]) = 10;
  BYTE4(qword_4F982C0[22]) = 1;
  LODWORD(qword_4F982C0[22]) = 10;
  qword_4F982C0[6] = 192;
  BYTE4(qword_4F982C0[1]) = BYTE4(qword_4F982C0[1]) & 0x9F | 0x20;
  qword_4F982C0[5] = "An integer in percent used to specify the hot blocks/edges to be displayed in red: a block or edge "
                     "whose frequency is no less than the max frequency of the function multiplied by this percent.";
  sub_16B88A0(qword_4F982C0);
  __cxa_atexit(sub_12EDE60, qword_4F982C0, &qword_4A427C0);
  v8 = &v10;
  v10 = "none";
  v13 = "do not show.";
  v15 = "graph";
  v18 = "show a graph.";
  v20 = "text";
  v23 = "show in text.";
  v9 = 0x400000003LL;
  v11 = 4;
  v12 = 0;
  v14 = 12;
  v16 = 5;
  v17 = 1;
  v19 = 13;
  v21 = 4;
  v22 = 2;
  v24 = 13;
  v6 = "A boolean option to show CFG dag or text with block profile counts and branch probabilities right after PGO profi"
       "le annotation step. The profile counts are computed using branch probabilities from the runtime profile data and "
       "block frequency propagation algorithm. To view the raw counts from the profile, use option -pgo-view-raw-counts i"
       "nstead. To limit graph display to only one function, use filtering option -view-bfi-func-name.";
  v7 = 433;
  v5 = 1;
  sub_1369460(&unk_4F98060, "pgo-view-counts", &v5, &v6, &v8);
  if ( v8 != &v10 )
    _libc_free(v8, "pgo-view-counts");
  __cxa_atexit(sub_1367990, &unk_4F98060, &qword_4A427C0);
  qword_4F97F80 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F97F8C &= 0xF000u;
  qword_4F97F90 = 0;
  qword_4F97F98 = 0;
  qword_4F97FA0 = 0;
  qword_4F97FA8 = 0;
  dword_4F97F88 = v2;
  qword_4F97FB0 = 0;
  qword_4F97FC8 = (__int64)&unk_4FA01C0;
  qword_4F97FD8 = (__int64)&unk_4F97FF8;
  qword_4F97FE0 = (__int64)&unk_4F97FF8;
  qword_4F97FB8 = 0;
  qword_4F97FC0 = 0;
  qword_4F98028 = (__int64)&unk_49E74E8;
  word_4F98030 = 256;
  qword_4F97FD0 = 0;
  byte_4F98018 = 0;
  qword_4F97F80 = (__int64)&unk_49EEC70;
  qword_4F97FE8 = 4;
  byte_4F98020 = 0;
  qword_4F98038 = (__int64)&unk_49EEDB0;
  dword_4F97FF0 = 0;
  sub_16B8280(&qword_4F97F80, "print-bfi", 9);
  word_4F98030 = 256;
  byte_4F98020 = 0;
  qword_4F97FB0 = 31;
  LOBYTE(word_4F97F8C) = word_4F97F8C & 0x9F | 0x20;
  qword_4F97FA8 = (__int64)"Print the block frequency info.";
  sub_16B88A0(&qword_4F97F80);
  __cxa_atexit(sub_12EDEC0, &qword_4F97F80, &qword_4A427C0);
  qword_4F97E60[0] = &unk_49EED30;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4F97E60[1]) &= 0xF000u;
  LODWORD(qword_4F97E60[1]) = v3;
  qword_4F97E60[2] = 0;
  qword_4F97E60[9] = &unk_4FA01C0;
  qword_4F97E60[11] = &qword_4F97E60[15];
  qword_4F97E60[12] = &qword_4F97E60[15];
  qword_4F97E60[20] = &qword_4F97E60[22];
  qword_4F97E60[25] = &qword_4F97E60[27];
  qword_4F97E60[3] = 0;
  qword_4F97E60[4] = 0;
  qword_4F97E60[24] = &unk_49EED10;
  qword_4F97E60[5] = 0;
  qword_4F97E60[6] = 0;
  qword_4F97E60[0] = &unk_49EEBF0;
  qword_4F97E60[7] = 0;
  qword_4F97E60[8] = 0;
  qword_4F97E60[30] = &unk_49EEE90;
  qword_4F97E60[31] = &qword_4F97E60[33];
  qword_4F97E60[10] = 0;
  qword_4F97E60[13] = 4;
  LODWORD(qword_4F97E60[14]) = 0;
  LOBYTE(qword_4F97E60[19]) = 0;
  qword_4F97E60[21] = 0;
  LOBYTE(qword_4F97E60[22]) = 0;
  qword_4F97E60[26] = 0;
  LOBYTE(qword_4F97E60[27]) = 0;
  LOBYTE(qword_4F97E60[29]) = 0;
  qword_4F97E60[32] = 0;
  LOBYTE(qword_4F97E60[33]) = 0;
  sub_16B8280(qword_4F97E60, "print-bfi-func-name", 19);
  qword_4F97E60[6] = 85;
  BYTE4(qword_4F97E60[1]) = BYTE4(qword_4F97E60[1]) & 0x9F | 0x20;
  qword_4F97E60[5] = "The option to specify the name of the function whose block frequency info is printed.";
  sub_16B88A0(qword_4F97E60);
  return __cxa_atexit(sub_12F0C20, qword_4F97E60, &qword_4A427C0);
}
