// Function: ctor_186_0
// Address: 0x4dbec0
//
int ctor_186_0()
{
  int v0; // eax
  int v1; // edi
  int v2; // edi
  int v3; // edx
  int v4; // edx
  int v5; // edx
  int v6; // eax
  int v7; // eax
  const char *v8; // rsi
  const char *v9; // r12
  const char *v10; // r15
  __int64 v11; // rsi
  const char *v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // rbx
  __int64 v18; // rbx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // r10
  __int64 v23; // rax
  int v24; // r10d
  void *v25; // r11
  __m128i *v26; // rax
  __int64 v27; // rdi
  const __m128i *v28; // rsi
  __int8 v29; // dl
  __int64 v30; // [rsp+0h] [rbp-130h]
  __int64 v31; // [rsp+8h] [rbp-128h]
  __int64 v32; // [rsp+8h] [rbp-128h]
  unsigned int v33; // [rsp+10h] [rbp-120h]
  __int64 v34; // [rsp+10h] [rbp-120h]
  __int64 v35; // [rsp+18h] [rbp-118h]
  unsigned int v36; // [rsp+18h] [rbp-118h]
  __int64 v37; // [rsp+20h] [rbp-110h]
  int v38; // [rsp+28h] [rbp-108h]
  __int64 v39; // [rsp+28h] [rbp-108h]
  int v40; // [rsp+28h] [rbp-108h]
  const char *v41; // [rsp+38h] [rbp-F8h]
  char v42; // [rsp+43h] [rbp-EDh] BYREF
  int v43; // [rsp+44h] [rbp-ECh] BYREF
  char *v44; // [rsp+48h] [rbp-E8h]
  const char *v45; // [rsp+50h] [rbp-E0h]
  __int64 v46; // [rsp+58h] [rbp-D8h]
  _QWORD v47[2]; // [rsp+60h] [rbp-D0h] BYREF
  int v48; // [rsp+70h] [rbp-C0h]
  const char *v49; // [rsp+78h] [rbp-B8h]
  __int64 v50; // [rsp+80h] [rbp-B0h]
  char *v51; // [rsp+88h] [rbp-A8h]
  __int64 v52; // [rsp+90h] [rbp-A0h]
  int v53; // [rsp+98h] [rbp-98h]
  const char *v54; // [rsp+A0h] [rbp-90h]
  __int64 v55; // [rsp+A8h] [rbp-88h]

  v45 = "profuse for inlining";
  v44 = &v42;
  v46 = 20;
  v42 = 1;
  v43 = 1;
  sub_186ADD0(&unk_4FABD60, "profuseinline", &v43);
  __cxa_atexit(sub_12EDEC0, &unk_4FABD60, &qword_4A427C0);
  qword_4FABC80 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FABC8C &= 0xF000u;
  qword_4FABCC8 = (__int64)qword_4FA01C0;
  qword_4FABC90 = 0;
  qword_4FABC98 = 0;
  qword_4FABCA0 = 0;
  dword_4FABC88 = v0;
  qword_4FABCD8 = (__int64)&unk_4FABCF8;
  qword_4FABCE0 = (__int64)&unk_4FABCF8;
  qword_4FABCA8 = 0;
  qword_4FABCB0 = 0;
  qword_4FABD28 = (__int64)&unk_49E74C8;
  qword_4FABCB8 = 0;
  qword_4FABC80 = (__int64)&unk_49EEB70;
  qword_4FABCC0 = 0;
  qword_4FABCD0 = 0;
  qword_4FABD38 = (__int64)&unk_49EEDF0;
  qword_4FABCE8 = 4;
  dword_4FABCF0 = 0;
  byte_4FABD18 = 0;
  dword_4FABD20 = 0;
  byte_4FABD34 = 1;
  dword_4FABD30 = 0;
  sub_16B8280(&qword_4FABC80, "inline-total-budget", 19);
  dword_4FABD20 = 500000;
  qword_4FABCA8 = (__int64)"Total inlining budget";
  byte_4FABD34 = 1;
  dword_4FABD30 = 500000;
  qword_4FABCB0 = 21;
  LOBYTE(word_4FABC8C) = word_4FABC8C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FABC80);
  __cxa_atexit(sub_12EDEA0, &qword_4FABC80, &qword_4A427C0);
  v45 = "Control to inline all function calls if possible";
  v44 = &v42;
  v46 = 48;
  v42 = 0;
  v43 = 1;
  sub_186ADD0(&unk_4FABBA0, "nv-inline-all", &v43);
  __cxa_atexit(sub_12EDEC0, &unk_4FABBA0, &qword_4A427C0);
  qword_4FABAC0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FABB18 = (__int64)&unk_4FABB38;
  qword_4FABB20 = (__int64)&unk_4FABB38;
  word_4FABACC &= 0xF000u;
  qword_4FABB78 = (__int64)&unk_49EEDF0;
  qword_4FABAC0 = (__int64)&unk_49EEB70;
  dword_4FABAC8 = v1;
  qword_4FABB08 = (__int64)qword_4FA01C0;
  qword_4FABB68 = (__int64)&unk_49E74C8;
  qword_4FABAD0 = 0;
  qword_4FABAD8 = 0;
  qword_4FABAE0 = 0;
  qword_4FABAE8 = 0;
  qword_4FABAF0 = 0;
  qword_4FABAF8 = 0;
  qword_4FABB00 = 0;
  qword_4FABB10 = 0;
  qword_4FABB28 = 4;
  dword_4FABB30 = 0;
  byte_4FABB58 = 0;
  dword_4FABB60 = 0;
  byte_4FABB74 = 1;
  dword_4FABB70 = 0;
  sub_16B8280(&qword_4FABAC0, "inline-budget", 13);
  dword_4FABB60 = 20000;
  qword_4FABAE8 = (__int64)"Control the amount of inlining to perform to each caller (default = 20000)";
  byte_4FABB74 = 1;
  dword_4FABB70 = 20000;
  qword_4FABAF0 = 74;
  LOBYTE(word_4FABACC) = word_4FABACC & 0x98 | 0x21;
  sub_16B88A0(&qword_4FABAC0);
  __cxa_atexit(sub_12EDEA0, &qword_4FABAC0, &qword_4A427C0);
  qword_4FAB9E0 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FABA38 = (__int64)&unk_4FABA58;
  qword_4FABA40 = (__int64)&unk_4FABA58;
  word_4FAB9EC &= 0xF000u;
  qword_4FABA98 = (__int64)&unk_49EEDF0;
  qword_4FAB9E0 = (__int64)&unk_49EEB70;
  dword_4FAB9E8 = v2;
  qword_4FABA28 = (__int64)qword_4FA01C0;
  qword_4FABA88 = (__int64)&unk_49E74C8;
  qword_4FAB9F0 = 0;
  qword_4FAB9F8 = 0;
  qword_4FABA00 = 0;
  qword_4FABA08 = 0;
  qword_4FABA10 = 0;
  qword_4FABA18 = 0;
  qword_4FABA20 = 0;
  qword_4FABA30 = 0;
  qword_4FABA48 = 4;
  dword_4FABA50 = 0;
  byte_4FABA78 = 0;
  dword_4FABA80 = 0;
  byte_4FABA94 = 1;
  dword_4FABA90 = 0;
  sub_16B8280(&qword_4FAB9E0, "inline-adj-budget1", 18);
  dword_4FABA80 = 1;
  byte_4FABA94 = 1;
  dword_4FABA90 = 1;
  qword_4FABA10 = 66;
  qword_4FABA08 = (__int64)"Adjusted control the amount of inlining to perform to each caller)";
  LOBYTE(word_4FAB9EC) = word_4FAB9EC & 0x98 | 0x21;
  sub_16B88A0(&qword_4FAB9E0);
  __cxa_atexit(sub_12EDEA0, &qword_4FAB9E0, &qword_4A427C0);
  qword_4FAB900 = (__int64)&unk_49EED30;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAB90C &= 0xF000u;
  qword_4FAB958 = (__int64)&unk_4FAB978;
  qword_4FAB960 = (__int64)&unk_4FAB978;
  qword_4FAB910 = 0;
  qword_4FAB918 = 0;
  word_4FAB9B0 = 256;
  dword_4FAB908 = v3;
  qword_4FAB9A8 = (__int64)&unk_49E74E8;
  qword_4FAB900 = (__int64)&unk_49EEC70;
  qword_4FAB9B8 = (__int64)&unk_49EEDB0;
  qword_4FAB948 = (__int64)qword_4FA01C0;
  qword_4FAB920 = 0;
  qword_4FAB928 = 0;
  qword_4FAB930 = 0;
  qword_4FAB938 = 0;
  qword_4FAB940 = 0;
  qword_4FAB950 = 0;
  qword_4FAB968 = 4;
  dword_4FAB970 = 0;
  byte_4FAB998 = 0;
  byte_4FAB9A0 = 0;
  sub_16B8280(&qword_4FAB900, "inline-switchctrl", 17);
  byte_4FAB9A0 = 1;
  word_4FAB9B0 = 257;
  qword_4FAB928 = (__int64)"Control to tuning inline heuristic based on switches";
  qword_4FAB930 = 52;
  LOBYTE(word_4FAB90C) = word_4FAB90C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAB900);
  __cxa_atexit(sub_12EDEC0, &qword_4FAB900, &qword_4A427C0);
  qword_4FAB820 = (__int64)&unk_49EED30;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAB82C &= 0xF000u;
  qword_4FAB878 = (__int64)&unk_4FAB898;
  qword_4FAB880 = (__int64)&unk_4FAB898;
  qword_4FAB830 = 0;
  qword_4FAB8D8 = (__int64)&unk_49EEDF0;
  qword_4FAB820 = (__int64)&unk_49EEB70;
  dword_4FAB828 = v4;
  qword_4FAB868 = (__int64)qword_4FA01C0;
  qword_4FAB8C8 = (__int64)&unk_49E74C8;
  qword_4FAB838 = 0;
  qword_4FAB840 = 0;
  qword_4FAB848 = 0;
  qword_4FAB850 = 0;
  qword_4FAB858 = 0;
  qword_4FAB860 = 0;
  qword_4FAB870 = 0;
  qword_4FAB888 = 4;
  dword_4FAB890 = 0;
  byte_4FAB8B8 = 0;
  dword_4FAB8C0 = 0;
  byte_4FAB8D4 = 1;
  dword_4FAB8D0 = 0;
  sub_16B8280(&qword_4FAB820, "inline-numswitchfunc", 20);
  dword_4FAB8C0 = 5;
  byte_4FAB8D4 = 1;
  dword_4FAB8D0 = 5;
  qword_4FAB850 = 47;
  qword_4FAB848 = (__int64)"Control of inline heuristic on switch functions";
  LOBYTE(word_4FAB82C) = word_4FAB82C & 0x98 | 0x21;
  sub_16B88A0(&qword_4FAB820);
  __cxa_atexit(sub_12EDEA0, &qword_4FAB820, &qword_4A427C0);
  qword_4FAB740 = (__int64)&unk_49EED30;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAB74C &= 0xF000u;
  qword_4FAB798 = (__int64)&unk_4FAB7B8;
  qword_4FAB7A0 = (__int64)&unk_4FAB7B8;
  qword_4FAB750 = 0;
  qword_4FAB7F8 = (__int64)&unk_49EEDF0;
  qword_4FAB740 = (__int64)&unk_49EEB70;
  dword_4FAB748 = v5;
  qword_4FAB788 = (__int64)qword_4FA01C0;
  qword_4FAB7E8 = (__int64)&unk_49E74C8;
  qword_4FAB758 = 0;
  qword_4FAB760 = 0;
  qword_4FAB768 = 0;
  qword_4FAB770 = 0;
  qword_4FAB778 = 0;
  qword_4FAB780 = 0;
  qword_4FAB790 = 0;
  qword_4FAB7A8 = 4;
  dword_4FAB7B0 = 0;
  byte_4FAB7D8 = 0;
  dword_4FAB7E0 = 0;
  byte_4FAB7F4 = 1;
  dword_4FAB7F0 = 0;
  sub_16B8280(&qword_4FAB740, "inline-maxswitchcases", 21);
  dword_4FAB7E0 = 71;
  byte_4FAB7F4 = 1;
  dword_4FAB7F0 = 71;
  qword_4FAB770 = 43;
  LOBYTE(word_4FAB74C) = word_4FAB74C & 0x98 | 0x21;
  qword_4FAB768 = (__int64)"Control of inline heuristic on switch cases";
  sub_16B88A0(&qword_4FAB740);
  __cxa_atexit(sub_12EDEA0, &qword_4FAB740, &qword_4A427C0);
  qword_4FAB660 = (__int64)&unk_49EED30;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAB710 = 256;
  qword_4FAB670 = 0;
  word_4FAB66C &= 0xF000u;
  qword_4FAB708 = (__int64)&unk_49E74E8;
  qword_4FAB660 = (__int64)&unk_49EEC70;
  dword_4FAB668 = v6;
  qword_4FAB718 = (__int64)&unk_49EEDB0;
  qword_4FAB6A8 = (__int64)qword_4FA01C0;
  qword_4FAB6B8 = (__int64)&unk_4FAB6D8;
  qword_4FAB6C0 = (__int64)&unk_4FAB6D8;
  qword_4FAB678 = 0;
  qword_4FAB680 = 0;
  qword_4FAB688 = 0;
  qword_4FAB690 = 0;
  qword_4FAB698 = 0;
  qword_4FAB6A0 = 0;
  qword_4FAB6B0 = 0;
  qword_4FAB6C8 = 4;
  dword_4FAB6D0 = 0;
  byte_4FAB6F8 = 0;
  byte_4FAB700 = 0;
  sub_16B8280((char *)&unk_4FAB6D8 - 120, "disable-inlined-alloca-merging", 30);
  word_4FAB710 = 256;
  byte_4FAB700 = 0;
  LOBYTE(word_4FAB66C) = word_4FAB66C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAB660);
  __cxa_atexit(sub_12EDEC0, &qword_4FAB660, &qword_4A427C0);
  v47[1] = 5;
  v45 = (const char *)v47;
  v47[0] = "basic";
  v49 = "basic statistics";
  v51 = "verbose";
  v54 = "printing of statistics for each inlined function";
  v46 = 0x400000002LL;
  v48 = 1;
  v50 = 16;
  v52 = 7;
  v53 = 2;
  v55 = 48;
  qword_4FAB400 = (__int64)&unk_49EED30;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  v8 = "inliner-function-import-stats";
  word_4FAB40C &= 0xF000u;
  qword_4FAB410 = 0;
  qword_4FAB418 = 0;
  qword_4FAB420 = 0;
  qword_4FAB428 = 0;
  dword_4FAB408 = v7;
  qword_4FAB458 = (__int64)&unk_4FAB478;
  qword_4FAB460 = (__int64)&unk_4FAB478;
  qword_4FAB400 = (__int64)off_49F17C8;
  qword_4FAB4B8 = (__int64)&off_49F1778;
  qword_4FAB4C8 = (__int64)&unk_4FAB4D8;
  qword_4FAB448 = (__int64)qword_4FA01C0;
  qword_4FAB4D0 = 0x800000000LL;
  qword_4FAB430 = 0;
  qword_4FAB438 = 0;
  qword_4FAB440 = 0;
  qword_4FAB450 = 0;
  qword_4FAB468 = 4;
  dword_4FAB470 = 0;
  byte_4FAB498 = 0;
  dword_4FAB4A0 = 0;
  qword_4FAB4A8 = (__int64)&off_49F1758;
  byte_4FAB4B4 = 1;
  dword_4FAB4B0 = 0;
  qword_4FAB4C0 = (__int64)&qword_4FAB400;
  sub_16B8280(&qword_4FAB400, "inliner-function-import-stats", 29);
  dword_4FAB4A0 = 0;
  byte_4FAB4B4 = 1;
  v41 = v45;
  v9 = v45;
  dword_4FAB4B0 = 0;
  v10 = &v45[40 * (unsigned int)v46];
  if ( v45 != v10 )
  {
    do
    {
      LODWORD(v11) = qword_4FAB4D0;
      v12 = *(const char **)v9;
      v13 = *((_QWORD *)v9 + 1);
      v14 = *((unsigned int *)v9 + 4);
      v15 = *((_QWORD *)v9 + 3);
      v16 = *((_QWORD *)v9 + 4);
      if ( (unsigned int)qword_4FAB4D0 >= HIDWORD(qword_4FAB4D0) )
      {
        v30 = *((_QWORD *)v9 + 4);
        v31 = *((_QWORD *)v9 + 3);
        v33 = *((_DWORD *)v9 + 4);
        v20 = (((unsigned __int64)HIDWORD(qword_4FAB4D0) + 2) >> 1) | (HIDWORD(qword_4FAB4D0) + 2LL);
        v35 = *((_QWORD *)v9 + 1);
        v21 = (((v20 >> 2) | v20) >> 4) | (v20 >> 2) | v20;
        v22 = ((v21 >> 8) | v21 | (((v21 >> 8) | v21) >> 16) | (((v21 >> 8) | v21) >> 32)) + 1;
        if ( v22 > 0xFFFFFFFF )
          v22 = 0xFFFFFFFFLL;
        v38 = v22;
        v23 = malloc(48 * v22, (unsigned int)qword_4FAB4D0, v13, v14, v16, v15);
        v24 = v38;
        v11 = (unsigned int)v11;
        v14 = v33;
        v13 = v35;
        v17 = v23;
        v15 = v31;
        v16 = v30;
        if ( !v23 )
        {
          sub_16BD1C0("Allocation failed");
          v11 = (unsigned int)qword_4FAB4D0;
          v16 = v30;
          v15 = v31;
          v14 = v33;
          v13 = v35;
          v24 = v38;
        }
        v25 = (void *)qword_4FAB4C8;
        v26 = (__m128i *)v17;
        v27 = qword_4FAB4C8 + 48 * v11;
        v28 = (const __m128i *)qword_4FAB4C8;
        if ( qword_4FAB4C8 != v27 )
        {
          v39 = v13;
          do
          {
            if ( v26 )
            {
              *v26 = _mm_loadu_si128(v28);
              v26[1] = _mm_loadu_si128(v28 + 1);
              v26[2].m128i_i32[2] = v28[2].m128i_i32[2];
              v29 = v28[2].m128i_i8[12];
              v26[2].m128i_i64[0] = (__int64)&off_49F1758;
              v26[2].m128i_i8[12] = v29;
            }
            v28 += 3;
            v26 += 3;
          }
          while ( (const __m128i *)v27 != v28 );
          v13 = v39;
        }
        if ( v25 != &unk_4FAB4D8 )
        {
          v32 = v16;
          v34 = v15;
          v36 = v14;
          v37 = v13;
          v40 = v24;
          _libc_free(v25, v28);
          v16 = v32;
          v15 = v34;
          v14 = v36;
          v13 = v37;
          v24 = v40;
        }
        qword_4FAB4C8 = v17;
        LODWORD(v11) = qword_4FAB4D0;
        HIDWORD(qword_4FAB4D0) = v24;
      }
      else
      {
        v17 = qword_4FAB4C8;
      }
      v18 = 48LL * (unsigned int)v11 + v17;
      if ( v18 )
      {
        *(_QWORD *)v18 = v12;
        *(_QWORD *)(v18 + 8) = v13;
        *(_QWORD *)(v18 + 16) = v15;
        *(_QWORD *)(v18 + 24) = v16;
        *(_DWORD *)(v18 + 40) = v14;
        *(_BYTE *)(v18 + 44) = 1;
        *(_QWORD *)(v18 + 32) = &off_49F1758;
        LODWORD(v11) = qword_4FAB4D0;
      }
      v9 += 40;
      LODWORD(qword_4FAB4D0) = v11 + 1;
      v8 = v12;
      sub_16B7FD0(qword_4FAB4C0, v12, v13, v14);
    }
    while ( v10 != v9 );
  }
  qword_4FAB430 = 43;
  LOBYTE(word_4FAB40C) = word_4FAB40C & 0x9F | 0x20;
  qword_4FAB428 = (__int64)"Enable inliner stats for imported functions";
  sub_16B88A0(&qword_4FAB400);
  if ( v41 != (const char *)v47 )
    _libc_free(v41, v8);
  return __cxa_atexit(sub_186A120, &qword_4FAB400, &qword_4A427C0);
}
