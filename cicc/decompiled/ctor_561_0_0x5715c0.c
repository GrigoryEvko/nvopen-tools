// Function: ctor_561_0
// Address: 0x5715c0
//
int __fastcall ctor_561_0(__int64 a1, __int64 a2, int a3, __int64 a4, int a5, int a6)
{
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  const char *v10; // rsi
  _BYTE *v11; // r13
  _BYTE *i; // r14
  int v13; // eax
  const char *v14; // r9
  const __m128i *v15; // r10
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rsi
  int v22; // ecx
  __m128i *v23; // rax
  __m128i v24; // xmm1
  __int8 v25; // dl
  __int64 v26; // rcx
  int v27; // edx
  int v28; // r8d
  int v29; // r9d
  int v30; // edx
  __int64 v31; // rbx
  __int64 v32; // rax
  const char *v33; // rsi
  _BYTE *v34; // r15
  _BYTE *j; // r13
  int v36; // eax
  __int64 v37; // rsi
  __int64 v38; // rdx
  const char *v39; // r8
  __int64 v40; // rax
  __int64 v41; // rcx
  const __m128i *v42; // rsi
  unsigned __int64 v43; // r9
  int v44; // edx
  __int64 v45; // rax
  __m128i v46; // xmm3
  __int8 v47; // dl
  int v48; // edx
  __int64 v49; // rbx
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  __int128 v53; // [rsp-A0h] [rbp-350h]
  __int128 v54; // [rsp-A0h] [rbp-350h]
  __int128 v55; // [rsp-90h] [rbp-340h]
  __int128 v56; // [rsp-90h] [rbp-340h]
  __int128 v57; // [rsp-78h] [rbp-328h]
  __int128 v58; // [rsp-78h] [rbp-328h]
  __int128 v59; // [rsp-68h] [rbp-318h]
  __int128 v60; // [rsp-68h] [rbp-318h]
  __int128 v61; // [rsp-50h] [rbp-300h]
  __int128 v62; // [rsp-50h] [rbp-300h]
  __int128 v63; // [rsp-40h] [rbp-2F0h]
  __int128 v64; // [rsp-40h] [rbp-2F0h]
  __int128 v65; // [rsp-28h] [rbp-2D8h]
  __int128 v66; // [rsp-28h] [rbp-2D8h]
  __int128 v67; // [rsp-18h] [rbp-2C8h]
  __int128 v68; // [rsp-18h] [rbp-2C8h]
  __int64 v69; // [rsp+0h] [rbp-2B0h]
  __int64 v70; // [rsp+0h] [rbp-2B0h]
  __int64 v71; // [rsp+0h] [rbp-2B0h]
  __int64 v72; // [rsp+8h] [rbp-2A8h]
  __int64 v73; // [rsp+8h] [rbp-2A8h]
  const char *v74; // [rsp+8h] [rbp-2A8h]
  const char *v75; // [rsp+8h] [rbp-2A8h]
  __int64 v76; // [rsp+10h] [rbp-2A0h]
  const char *v77; // [rsp+18h] [rbp-298h]
  const char *v78; // [rsp+18h] [rbp-298h]
  __int64 v79; // [rsp+30h] [rbp-280h]
  __int64 v80; // [rsp+60h] [rbp-250h]
  __int64 v81; // [rsp+90h] [rbp-220h]
  __int64 v82; // [rsp+C0h] [rbp-1F0h]
  __int64 v83; // [rsp+F0h] [rbp-1C0h]
  __int64 v84; // [rsp+120h] [rbp-190h]
  __int64 v85; // [rsp+150h] [rbp-160h]
  __int64 v86; // [rsp+180h] [rbp-130h]
  const char *v87; // [rsp+1A0h] [rbp-110h] BYREF
  __int64 v88; // [rsp+1A8h] [rbp-108h]
  __int64 v89; // [rsp+1B0h] [rbp-100h]
  __int64 v90; // [rsp+1B8h] [rbp-F8h]
  void *v91; // [rsp+1C0h] [rbp-F0h]
  int v92; // [rsp+1C8h] [rbp-E8h]
  char v93; // [rsp+1CCh] [rbp-E4h]
  _BYTE *v94; // [rsp+1D0h] [rbp-E0h] BYREF
  int v95; // [rsp+1D8h] [rbp-D8h]
  _BYTE v96[208]; // [rsp+1E0h] [rbp-D0h] BYREF

  *((_QWORD *)&v67 + 1) = "display a graph using the real profile count if available.";
  LODWORD(v86) = 3;
  LODWORD(v85) = 2;
  LODWORD(v84) = 1;
  LODWORD(v83) = 0;
  *(_QWORD *)&v67 = v86;
  *((_QWORD *)&v65 + 1) = 5;
  *(_QWORD *)&v65 = "count";
  *((_QWORD *)&v63 + 1) = "display a graph using the raw integer fractional block frequency representation.";
  *(_QWORD *)&v63 = v85;
  *((_QWORD *)&v61 + 1) = 7;
  *(_QWORD *)&v61 = "integer";
  *((_QWORD *)&v59 + 1) = "display a graph using the fractional block frequency representation.";
  *(_QWORD *)&v59 = v84;
  *((_QWORD *)&v57 + 1) = 8;
  *(_QWORD *)&v57 = "fraction";
  *((_QWORD *)&v55 + 1) = "do not display graphs.";
  *(_QWORD *)&v55 = v83;
  *((_QWORD *)&v53 + 1) = 4;
  *(_QWORD *)&v53 = "none";
  sub_2273AE0(
    (unsigned int)&v94,
    (unsigned int)"do not display graphs.",
    a3,
    (unsigned int)"none",
    a5,
    a6,
    v53,
    v55,
    22,
    v57,
    v59,
    68,
    v61,
    v63,
    80,
    v65,
    v67,
    58);
  qword_501EF60 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501EFB0 = 0x100000000LL;
  dword_501EF6C &= 0x8000u;
  word_501EF70 = 0;
  qword_501EF78 = 0;
  qword_501EF80 = 0;
  dword_501EF68 = v6;
  qword_501EF88 = 0;
  qword_501EF90 = 0;
  qword_501EF98 = 0;
  qword_501EFA0 = 0;
  qword_501EFA8 = (__int64)&unk_501EFB8;
  qword_501EFC0 = 0;
  qword_501EFC8 = (__int64)&unk_501EFE0;
  qword_501EFD0 = 1;
  dword_501EFD8 = 0;
  byte_501EFDC = 1;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_501EFB0;
  v9 = (unsigned int)qword_501EFB0 + 1LL;
  if ( v9 > HIDWORD(qword_501EFB0) )
  {
    sub_C8D5F0((char *)&unk_501EFB8 - 16, &unk_501EFB8, v9, 8);
    v8 = (unsigned int)qword_501EFB0;
  }
  v10 = "view-machine-block-freq-propagation-dags";
  *(_QWORD *)(qword_501EFA8 + 8 * v8) = v7;
  LODWORD(qword_501EFB0) = qword_501EFB0 + 1;
  qword_501EFE8 = 0;
  qword_501EFF8 = 0;
  qword_501EFF0 = (__int64)&unk_49E5290;
  qword_501EF60 = (__int64)&unk_49E5300;
  qword_501F008 = (__int64)&qword_501EF60;
  qword_501F000 = (__int64)&unk_49E52B0;
  qword_501F010 = (__int64)&unk_501F020;
  qword_501F018 = 0x800000000LL;
  qword_501F1B8 = (__int64)nullsub_382;
  qword_501F1B0 = (__int64)sub_FDAD20;
  sub_C53080(&qword_501EF60, "view-machine-block-freq-propagation-dags", 40);
  qword_501EF90 = 97;
  LOBYTE(dword_501EF6C) = dword_501EF6C & 0x9F | 0x20;
  qword_501EF88 = (__int64)"Pop up a window to show a dag displaying how machine block frequencies propagate through the CFG.";
  v11 = &v94[40 * v95];
  for ( i = v94; v11 != i; i += 40 )
  {
    v13 = *((_DWORD *)i + 4);
    v14 = *(const char **)i;
    v91 = &unk_49E5290;
    v15 = (const __m128i *)&v87;
    v16 = *((_QWORD *)i + 3);
    v17 = *((_QWORD *)i + 4);
    v93 = 1;
    v18 = *((_QWORD *)i + 1);
    v92 = v13;
    v19 = (unsigned int)qword_501F018;
    v89 = v16;
    v90 = v17;
    v20 = qword_501F010;
    v21 = (unsigned int)qword_501F018 + 1LL;
    v87 = v14;
    v22 = qword_501F018;
    v88 = v18;
    if ( v21 > HIDWORD(qword_501F018) )
    {
      if ( qword_501F010 > (unsigned __int64)&v87 )
      {
        v71 = v18;
        v75 = v14;
        sub_FE6D40(&qword_501F010, v21, qword_501F010, (unsigned int)qword_501F018);
        v19 = (unsigned int)qword_501F018;
        v20 = qword_501F010;
        v15 = (const __m128i *)&v87;
        v14 = v75;
        v18 = v71;
        v22 = qword_501F018;
      }
      else
      {
        v70 = v18;
        v74 = v14;
        if ( (unsigned __int64)&v87 < qword_501F010 + 48 * (unsigned __int64)(unsigned int)qword_501F018 )
        {
          v76 = qword_501F010;
          sub_FE6D40(&qword_501F010, v21, qword_501F010, (unsigned int)qword_501F018);
          v19 = (unsigned int)qword_501F018;
          v18 = v70;
          v14 = v74;
          v20 = qword_501F010;
          v22 = qword_501F018;
          v15 = (const __m128i *)((char *)&v87 + qword_501F010 - v76);
        }
        else
        {
          sub_FE6D40(&qword_501F010, v21, qword_501F010, (unsigned int)qword_501F018);
          v19 = (unsigned int)qword_501F018;
          v20 = qword_501F010;
          v14 = v74;
          v18 = v70;
          v15 = (const __m128i *)&v87;
          v22 = qword_501F018;
        }
      }
    }
    v23 = (__m128i *)(v20 + 48 * v19);
    if ( v23 )
    {
      v24 = _mm_loadu_si128(v15 + 1);
      *v23 = _mm_loadu_si128(v15);
      v23[1] = v24;
      v23[2].m128i_i32[2] = v15[2].m128i_i32[2];
      v25 = v15[2].m128i_i8[12];
      v23[2].m128i_i64[0] = (__int64)&unk_49E5290;
      v23[2].m128i_i8[12] = v25;
      v22 = qword_501F018;
    }
    v26 = (unsigned int)(v22 + 1);
    v10 = v14;
    LODWORD(qword_501F018) = v26;
    sub_C52F90(qword_501F008, v14, v18, v26);
  }
  sub_C53130(&qword_501EF60);
  if ( v94 != v96 )
    _libc_free(v94, v10);
  __cxa_atexit(sub_FDB740, &qword_501EF60, &qword_4A427C0);
  LODWORD(v82) = 3;
  *((_QWORD *)&v68 + 1) = "display a graph using the real profile count if available.";
  *(_QWORD *)&v68 = v82;
  LODWORD(v81) = 2;
  *((_QWORD *)&v66 + 1) = 5;
  *(_QWORD *)&v66 = "count";
  LODWORD(v80) = 1;
  LODWORD(v79) = 0;
  *((_QWORD *)&v64 + 1) = "display a graph using the raw integer fractional block frequency representation.";
  *(_QWORD *)&v64 = v81;
  *((_QWORD *)&v62 + 1) = 7;
  *(_QWORD *)&v62 = "integer";
  *((_QWORD *)&v60 + 1) = "display a graph using the fractional block frequency representation.";
  *(_QWORD *)&v60 = v80;
  *((_QWORD *)&v58 + 1) = 8;
  *(_QWORD *)&v58 = "fraction";
  *((_QWORD *)&v56 + 1) = "do not display graphs.";
  *(_QWORD *)&v56 = v79;
  *((_QWORD *)&v54 + 1) = 4;
  *(_QWORD *)&v54 = "none";
  sub_2273AE0(
    (unsigned int)&v94,
    (unsigned int)"do not display graphs.",
    v27,
    (unsigned int)"none",
    v28,
    v29,
    v54,
    v56,
    22,
    v58,
    v60,
    68,
    v62,
    v64,
    80,
    v66,
    v68,
    58);
  qword_501ED00 = &unk_49DC150;
  v30 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_501ED0C = word_501ED0C & 0x8000;
  qword_501ED48[1] = 0x100000000LL;
  unk_501ED08 = v30;
  unk_501ED10 = 0;
  unk_501ED18 = 0;
  unk_501ED20 = 0;
  unk_501ED28 = 0;
  unk_501ED30 = 0;
  unk_501ED38 = 0;
  unk_501ED40 = 0;
  qword_501ED48[0] = &qword_501ED48[2];
  qword_501ED48[3] = 0;
  qword_501ED48[4] = &qword_501ED48[7];
  qword_501ED48[5] = 1;
  LODWORD(qword_501ED48[6]) = 0;
  BYTE4(qword_501ED48[6]) = 1;
  v31 = sub_C57470();
  v32 = LODWORD(qword_501ED48[1]);
  if ( (unsigned __int64)LODWORD(qword_501ED48[1]) + 1 > HIDWORD(qword_501ED48[1]) )
  {
    sub_C8D5F0(qword_501ED48, &qword_501ED48[2], LODWORD(qword_501ED48[1]) + 1LL, 8);
    v32 = LODWORD(qword_501ED48[1]);
  }
  v33 = "view-block-layout-with-bfi";
  *(_QWORD *)(qword_501ED48[0] + 8 * v32) = v31;
  ++LODWORD(qword_501ED48[1]);
  qword_501ED48[8] = 0;
  qword_501ED48[10] = 0;
  qword_501ED48[9] = &unk_49E5290;
  qword_501ED00 = &unk_49E5300;
  qword_501ED48[12] = &qword_501ED00;
  qword_501ED48[11] = &unk_49E52B0;
  qword_501ED48[13] = &qword_501ED48[15];
  qword_501ED48[14] = 0x800000000LL;
  qword_501ED48[66] = nullsub_382;
  qword_501ED48[65] = sub_FDAD20;
  sub_C53080(&qword_501ED00, "view-block-layout-with-bfi", 26);
  unk_501ED30 = 96;
  LOBYTE(word_501ED0C) = word_501ED0C & 0x9F | 0x20;
  unk_501ED28 = "Pop up a window to show a dag displaying MBP layout and associated block frequencies of the CFG.";
  v34 = &v94[40 * v95];
  for ( j = v94; v34 != j; j += 40 )
  {
    v36 = *((_DWORD *)j + 4);
    v37 = *((_QWORD *)j + 3);
    v91 = &unk_49E5290;
    v38 = *((_QWORD *)j + 4);
    v39 = *(const char **)j;
    v93 = 1;
    v92 = v36;
    v40 = LODWORD(qword_501ED48[14]);
    v41 = *((_QWORD *)j + 1);
    v89 = v37;
    v42 = (const __m128i *)&v87;
    v43 = LODWORD(qword_501ED48[14]) + 1LL;
    v90 = v38;
    v44 = qword_501ED48[14];
    v87 = v39;
    v88 = v41;
    if ( v43 > HIDWORD(qword_501ED48[14]) )
    {
      if ( qword_501ED48[13] > (unsigned __int64)&v87
        || (v69 = qword_501ED48[13],
            (unsigned __int64)&v87 >= qword_501ED48[13] + 48 * (unsigned __int64)LODWORD(qword_501ED48[14])) )
      {
        v73 = v41;
        v78 = v39;
        sub_FE6D40(&qword_501ED48[13], v43, qword_501ED48[13], v41);
        v40 = LODWORD(qword_501ED48[14]);
        v41 = v73;
        v42 = (const __m128i *)&v87;
        v39 = v78;
        v44 = qword_501ED48[14];
      }
      else
      {
        v72 = v41;
        v77 = v39;
        sub_FE6D40(&qword_501ED48[13], v43, qword_501ED48[13], v41);
        v40 = LODWORD(qword_501ED48[14]);
        v39 = v77;
        v41 = v72;
        v44 = qword_501ED48[14];
        v42 = (const __m128i *)((char *)&v87 + qword_501ED48[13] - v69);
      }
    }
    v45 = qword_501ED48[13] + 48 * v40;
    if ( v45 )
    {
      v46 = _mm_loadu_si128(v42 + 1);
      *(__m128i *)v45 = _mm_loadu_si128(v42);
      *(__m128i *)(v45 + 16) = v46;
      *(_DWORD *)(v45 + 40) = v42[2].m128i_i32[2];
      v47 = v42[2].m128i_i8[12];
      *(_QWORD *)(v45 + 32) = &unk_49E5290;
      *(_BYTE *)(v45 + 44) = v47;
      v44 = qword_501ED48[14];
    }
    v33 = v39;
    LODWORD(qword_501ED48[14]) = v44 + 1;
    sub_C52F90(qword_501ED48[12], v39, v41, v41);
  }
  sub_C53130(&qword_501ED00);
  if ( v94 != v96 )
    _libc_free(v94, v33);
  __cxa_atexit(sub_FDB740, &qword_501ED00, &qword_4A427C0);
  qword_501EC20 = (__int64)&unk_49DC150;
  v48 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501EC9C = 1;
  qword_501EC70 = 0x100000000LL;
  dword_501EC2C &= 0x8000u;
  qword_501EC38 = 0;
  qword_501EC40 = 0;
  qword_501EC48 = 0;
  dword_501EC28 = v48;
  word_501EC30 = 0;
  qword_501EC50 = 0;
  qword_501EC58 = 0;
  qword_501EC60 = 0;
  qword_501EC68 = (__int64)&unk_501EC78;
  qword_501EC80 = 0;
  qword_501EC88 = (__int64)&unk_501ECA0;
  qword_501EC90 = 1;
  dword_501EC98 = 0;
  v49 = sub_C57470();
  v50 = (unsigned int)qword_501EC70;
  v51 = (unsigned int)qword_501EC70 + 1LL;
  if ( v51 > HIDWORD(qword_501EC70) )
  {
    sub_C8D5F0((char *)&unk_501EC78 - 16, &unk_501EC78, v51, 8);
    v50 = (unsigned int)qword_501EC70;
  }
  *(_QWORD *)(qword_501EC68 + 8 * v50) = v49;
  LODWORD(qword_501EC70) = qword_501EC70 + 1;
  qword_501ECA8 = 0;
  qword_501ECB0 = (__int64)&unk_49D9748;
  qword_501ECB8 = 0;
  qword_501EC20 = (__int64)&unk_49DC090;
  qword_501ECC0 = (__int64)&unk_49DC1D0;
  qword_501ECE0 = (__int64)nullsub_23;
  qword_501ECD8 = (__int64)sub_984030;
  sub_C53080(&qword_501EC20, "print-machine-bfi", 17);
  LOBYTE(qword_501ECA8) = 0;
  LOWORD(qword_501ECB8) = 256;
  qword_501EC50 = 39;
  LOBYTE(dword_501EC2C) = dword_501EC2C & 0x9F | 0x20;
  qword_501EC48 = (__int64)"Print the machine block frequency info.";
  sub_C53130(&qword_501EC20);
  return __cxa_atexit(sub_984900, &qword_501EC20, &qword_4A427C0);
}
