// Function: ctor_425_0
// Address: 0x5345f0
//
int ctor_425_0()
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
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v16; // [rsp+8h] [rbp-108h]
  int v17; // [rsp+10h] [rbp-100h] BYREF
  int v18; // [rsp+14h] [rbp-FCh] BYREF
  int *v19; // [rsp+18h] [rbp-F8h] BYREF
  const char *v20; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v21; // [rsp+28h] [rbp-E8h]
  char **v22; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v23; // [rsp+38h] [rbp-D8h]
  char *v24; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+48h] [rbp-C8h]
  int v26; // [rsp+50h] [rbp-C0h]
  const char *v27; // [rsp+58h] [rbp-B8h]
  __int64 v28; // [rsp+60h] [rbp-B0h]
  char *v29; // [rsp+68h] [rbp-A8h]
  __int64 v30; // [rsp+70h] [rbp-A0h]
  int v31; // [rsp+78h] [rbp-98h]
  const char *v32; // [rsp+80h] [rbp-90h]
  __int64 v33; // [rsp+88h] [rbp-88h]
  const char *v34; // [rsp+90h] [rbp-80h]
  __int64 v35; // [rsp+98h] [rbp-78h]
  int v36; // [rsp+A0h] [rbp-70h]
  const char *v37; // [rsp+A8h] [rbp-68h]
  __int64 v38; // [rsp+B0h] [rbp-60h]
  const char *v39; // [rsp+B8h] [rbp-58h]
  __int64 v40; // [rsp+C0h] [rbp-50h]
  int v41; // [rsp+C8h] [rbp-48h]
  const char *v42; // [rsp+D0h] [rbp-40h]
  __int64 v43; // [rsp+D8h] [rbp-38h]

  qword_4FF2980 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF29D0 = 0x100000000LL;
  dword_4FF298C &= 0x8000u;
  word_4FF2990 = 0;
  qword_4FF2998 = 0;
  qword_4FF29A0 = 0;
  dword_4FF2988 = v0;
  qword_4FF29A8 = 0;
  qword_4FF29B0 = 0;
  qword_4FF29B8 = 0;
  qword_4FF29C0 = 0;
  qword_4FF29C8 = (__int64)&unk_4FF29D8;
  qword_4FF29E0 = 0;
  qword_4FF29E8 = (__int64)&unk_4FF2A00;
  qword_4FF29F0 = 1;
  dword_4FF29F8 = 0;
  byte_4FF29FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF29D0;
  v3 = (unsigned int)qword_4FF29D0 + 1LL;
  if ( v3 > HIDWORD(qword_4FF29D0) )
  {
    sub_C8D5F0((char *)&unk_4FF29D8 - 16, &unk_4FF29D8, v3, 8);
    v2 = (unsigned int)qword_4FF29D0;
  }
  *(_QWORD *)(qword_4FF29C8 + 8 * v2) = v1;
  LODWORD(qword_4FF29D0) = qword_4FF29D0 + 1;
  qword_4FF2A08 = 0;
  qword_4FF2A10 = (__int64)&unk_49DA090;
  qword_4FF2A18 = 0;
  qword_4FF2980 = (__int64)&unk_49DBF90;
  qword_4FF2A20 = (__int64)&unk_49DC230;
  qword_4FF2A40 = (__int64)nullsub_58;
  qword_4FF2A38 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FF2980, "intra-scc-cost-multiplier", 25);
  LODWORD(qword_4FF2A08) = 2;
  BYTE4(qword_4FF2A18) = 1;
  LODWORD(qword_4FF2A18) = 2;
  qword_4FF29B0 = 503;
  LOBYTE(dword_4FF298C) = dword_4FF298C & 0x9F | 0x20;
  qword_4FF29A8 = (__int64)"Cost multiplier to multiply onto inlined call sites where the new call was previously an intr"
                           "a-SCC call (not relevant when the original call was already intra-SCC). This can accumulate o"
                           "ver multiple inlinings (e.g. if a call site already had a cost multiplier and one of its inli"
                           "ned calls was also subject to this, the inlined call would have the original multiplier multi"
                           "plied by intra-scc-cost-multiplier). This is to prevent tons of inlining through a child SCC "
                           "which can cause terrible compile times";
  sub_C53130(&qword_4FF2980);
  __cxa_atexit(sub_B2B680, &qword_4FF2980, &qword_4A427C0);
  qword_4FF28A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF28F0 = 0x100000000LL;
  dword_4FF28AC &= 0x8000u;
  word_4FF28B0 = 0;
  qword_4FF28B8 = 0;
  qword_4FF28C0 = 0;
  dword_4FF28A8 = v4;
  qword_4FF28C8 = 0;
  qword_4FF28D0 = 0;
  qword_4FF28D8 = 0;
  qword_4FF28E0 = 0;
  qword_4FF28E8 = (__int64)&unk_4FF28F8;
  qword_4FF2900 = 0;
  qword_4FF2908 = (__int64)&unk_4FF2920;
  qword_4FF2910 = 1;
  dword_4FF2918 = 0;
  byte_4FF291C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF28F0;
  v7 = (unsigned int)qword_4FF28F0 + 1LL;
  if ( v7 > HIDWORD(qword_4FF28F0) )
  {
    sub_C8D5F0((char *)&unk_4FF28F8 - 16, &unk_4FF28F8, v7, 8);
    v6 = (unsigned int)qword_4FF28F0;
  }
  *(_QWORD *)(qword_4FF28E8 + 8 * v6) = v5;
  qword_4FF2930 = (__int64)&unk_49D9748;
  qword_4FF28A0 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF28F0) = qword_4FF28F0 + 1;
  qword_4FF2928 = 0;
  qword_4FF2940 = (__int64)&unk_49DC1D0;
  qword_4FF2938 = 0;
  qword_4FF2960 = (__int64)nullsub_23;
  qword_4FF2958 = (__int64)sub_984030;
  sub_C53080(&qword_4FF28A0, "keep-inline-advisor-for-printing", 32);
  LOWORD(qword_4FF2938) = 256;
  LOBYTE(qword_4FF2928) = 0;
  LOBYTE(dword_4FF28AC) = dword_4FF28AC & 0x9F | 0x20;
  sub_C53130(&qword_4FF28A0);
  __cxa_atexit(sub_984900, &qword_4FF28A0, &qword_4A427C0);
  qword_4FF27C0 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF2810 = 0x100000000LL;
  dword_4FF27CC &= 0x8000u;
  word_4FF27D0 = 0;
  qword_4FF27D8 = 0;
  qword_4FF27E0 = 0;
  dword_4FF27C8 = v8;
  qword_4FF27E8 = 0;
  qword_4FF27F0 = 0;
  qword_4FF27F8 = 0;
  qword_4FF2800 = 0;
  qword_4FF2808 = (__int64)&unk_4FF2818;
  qword_4FF2820 = 0;
  qword_4FF2828 = (__int64)&unk_4FF2840;
  qword_4FF2830 = 1;
  dword_4FF2838 = 0;
  byte_4FF283C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FF2810;
  if ( (unsigned __int64)(unsigned int)qword_4FF2810 + 1 > HIDWORD(qword_4FF2810) )
  {
    v16 = v9;
    sub_C8D5F0((char *)&unk_4FF2818 - 16, &unk_4FF2818, (unsigned int)qword_4FF2810 + 1LL, 8);
    v10 = (unsigned int)qword_4FF2810;
    v9 = v16;
  }
  *(_QWORD *)(qword_4FF2808 + 8 * v10) = v9;
  qword_4FF2850 = (__int64)&unk_49D9748;
  qword_4FF27C0 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF2810) = qword_4FF2810 + 1;
  qword_4FF2848 = 0;
  qword_4FF2860 = (__int64)&unk_49DC1D0;
  qword_4FF2858 = 0;
  qword_4FF2880 = (__int64)nullsub_23;
  qword_4FF2878 = (__int64)sub_984030;
  sub_C53080(&qword_4FF27C0, "enable-scc-inline-advisor-printing", 34);
  LOBYTE(qword_4FF2848) = 0;
  LOWORD(qword_4FF2858) = 256;
  LOBYTE(dword_4FF27CC) = dword_4FF27CC & 0x9F | 0x20;
  sub_C53130(&qword_4FF27C0);
  __cxa_atexit(sub_984900, &qword_4FF27C0, &qword_4A427C0);
  qword_4FF26C0 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF273C = 1;
  qword_4FF2710 = 0x100000000LL;
  dword_4FF26CC &= 0x8000u;
  qword_4FF26D8 = 0;
  qword_4FF26E0 = 0;
  qword_4FF26E8 = 0;
  dword_4FF26C8 = v11;
  word_4FF26D0 = 0;
  qword_4FF26F0 = 0;
  qword_4FF26F8 = 0;
  qword_4FF2700 = 0;
  qword_4FF2708 = (__int64)&unk_4FF2718;
  qword_4FF2720 = 0;
  qword_4FF2728 = (__int64)&unk_4FF2740;
  qword_4FF2730 = 1;
  dword_4FF2738 = 0;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4FF2710;
  v14 = (unsigned int)qword_4FF2710 + 1LL;
  if ( v14 > HIDWORD(qword_4FF2710) )
  {
    sub_C8D5F0((char *)&unk_4FF2718 - 16, &unk_4FF2718, v14, 8);
    v13 = (unsigned int)qword_4FF2710;
  }
  *(_QWORD *)(qword_4FF2708 + 8 * v13) = v12;
  qword_4FF2748 = (__int64)&byte_4FF2758;
  qword_4FF2770 = (__int64)&byte_4FF2780;
  LODWORD(qword_4FF2710) = qword_4FF2710 + 1;
  qword_4FF2750 = 0;
  qword_4FF2768 = (__int64)&unk_49DC130;
  byte_4FF2758 = 0;
  byte_4FF2780 = 0;
  qword_4FF26C0 = (__int64)&unk_49DC010;
  qword_4FF2778 = 0;
  byte_4FF2790 = 0;
  qword_4FF2798 = (__int64)&unk_49DC350;
  qword_4FF27B8 = (__int64)nullsub_92;
  qword_4FF27B0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FF26C0, "cgscc-inline-replay", 19);
  v22 = &v24;
  v23 = 0;
  LOBYTE(v24) = 0;
  sub_2240AE0(&qword_4FF2748, &v22);
  byte_4FF2790 = 1;
  sub_2240AE0(&qword_4FF2770, &v22);
  if ( v22 != &v24 )
    j_j___libc_free_0(v22, v24 + 1);
  qword_4FF2700 = 8;
  qword_4FF26F8 = (__int64)"filename";
  qword_4FF26E8 = (__int64)"Optimization remarks file containing inline remarks to be replayed by cgscc inlining.";
  qword_4FF26F0 = 85;
  LOBYTE(dword_4FF26CC) = dword_4FF26CC & 0x9F | 0x20;
  sub_C53130(&qword_4FF26C0);
  __cxa_atexit(sub_BC5A40, &qword_4FF26C0, &qword_4A427C0);
  v18 = 1;
  v20 = "Whether inline replay should be applied to the entire Module or just the Functions (default) that are present as"
        " callers in remarks during cgscc inlining.";
  v24 = "Function";
  v27 = "Replay on functions that have remarks associated with them (default)";
  v29 = "Module";
  v32 = "Replay on the entire module";
  v23 = 0x400000002LL;
  v21 = 154;
  v22 = &v24;
  v25 = 8;
  v26 = 0;
  v28 = 68;
  v30 = 6;
  v31 = 1;
  v33 = 27;
  v17 = 0;
  v19 = &v17;
  sub_26167B0(&unk_4FF2460, "cgscc-inline-replay-scope", &v19, &v22, &v20, &v18);
  if ( v22 != &v24 )
    _libc_free(v22, "cgscc-inline-replay-scope");
  __cxa_atexit(sub_2610AB0, &unk_4FF2460, &qword_4A427C0);
  v18 = 1;
  v20 = "How cgscc inline replay treats sites that don't come from the replay. Original: defers to original advisor, Alwa"
        "ysInline: inline all sites not in replay, NeverInline: inline no sites not in replay";
  v24 = "Original";
  v27 = "All decisions not in replay send to original advisor (default)";
  v29 = "AlwaysInline";
  v32 = "All decisions not in replay are inlined";
  v34 = "NeverInline";
  v37 = "All decisions not in replay are not inlined";
  v23 = 0x400000003LL;
  v21 = 196;
  v22 = &v24;
  v25 = 8;
  v26 = 0;
  v28 = 62;
  v30 = 12;
  v31 = 1;
  v33 = 39;
  v35 = 11;
  v36 = 2;
  v38 = 43;
  v17 = 0;
  v19 = &v17;
  sub_2616C30(&unk_4FF2200, "cgscc-inline-replay-fallback", &v19, &v22, &v20, &v18);
  if ( v22 != &v24 )
    _libc_free(v22, "cgscc-inline-replay-fallback");
  __cxa_atexit(sub_2610B40, &unk_4FF2200, &qword_4A427C0);
  v18 = 1;
  v20 = "How cgscc inline replay file is formatted";
  v24 = "Line";
  v27 = "<Line Number>";
  v29 = "LineColumn";
  v32 = "<Line Number>:<Column Number>";
  v34 = "LineDiscriminator";
  v37 = "<Line Number>.<Discriminator>";
  v39 = "LineColumnDiscriminator";
  v42 = "<Line Number>:<Column Number>.<Discriminator> (default)";
  v23 = 0x400000004LL;
  v21 = 41;
  v22 = &v24;
  v25 = 4;
  v26 = 0;
  v28 = 13;
  v30 = 10;
  v31 = 1;
  v33 = 29;
  v35 = 17;
  v36 = 2;
  v38 = 29;
  v40 = 23;
  v41 = 3;
  v43 = 55;
  v17 = 3;
  v19 = &v17;
  sub_26170B0(&unk_4FF1FA0, "cgscc-inline-replay-format", &v19, &v22, &v20, &v18);
  if ( v22 != &v24 )
    _libc_free(v22, "cgscc-inline-replay-format");
  return __cxa_atexit(sub_2610A20, &unk_4FF1FA0, &qword_4A427C0);
}
