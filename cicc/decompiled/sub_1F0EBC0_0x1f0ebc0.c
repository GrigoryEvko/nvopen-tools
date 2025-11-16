// Function: sub_1F0EBC0
// Address: 0x1f0ebc0
//
void __fastcall sub_1F0EBC0(__int64 a1, __int64 *a2)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // r13
  __int64 (*v26)(); // rax
  __int64 v27; // rax
  __int64 (*v28)(); // rax
  __int64 v29; // rax
  unsigned int v30; // edx
  __int64 v31; // rax
  unsigned int v32; // edx
  unsigned int v33; // ecx
  __int64 v34; // rax
  void *v35; // rdi
  size_t v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rax
  const char **v39; // rdi
  __int64 v40; // rcx
  const __m128i *v41; // rdx
  _BYTE *v42; // rsi
  __m128i *v43; // rax
  __int64 v44; // rsi
  __m128i *v45; // rcx
  __m128i *v46; // rax
  __m128i *v47; // rax
  __int8 *v48; // rax
  const __m128i *v49; // rcx
  unsigned __int64 v50; // r15
  __m128i *v51; // rax
  __m128i *v52; // rcx
  __m128i *v53; // rax
  __m128i *v54; // rax
  __int8 *v55; // rax
  __int64 (*v56)(); // rax
  __int64 v57; // rdi
  __int64 (*v58)(); // rax
  _QWORD *v59; // r14
  __int64 v60; // rcx
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 v63; // r13
  __int64 (*v64)(void); // rax
  __int64 v65; // rax
  __int64 v66; // rsi
  unsigned __int64 v67; // r15
  unsigned __int64 v68; // r15
  __int64 v69; // rax
  __int64 (*v70)(); // rdx
  __int64 (*v71)(); // rax
  __int64 v72; // rax
  __int64 v73; // rdx
  unsigned __int64 v74; // rdi
  __int64 v75; // r13
  __int64 v76; // rsi
  __int64 **v77; // rdi
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // r9
  __int64 v82; // rsi
  __int64 v83; // r9
  int v84; // eax
  char v85; // al
  unsigned int v86; // edx
  unsigned int v87; // r13d
  char v88; // al
  __int64 v89; // rdi
  __int64 v90; // rax
  bool v91; // zf
  _DWORD *v92; // rax
  __int64 v93; // rdx
  _DWORD *i; // rdx
  _QWORD *v95; // rsi
  __int64 **v96; // rdi
  __int64 v97; // rdx
  __int64 v98; // r8
  __int64 v99; // r9
  _DWORD *v100; // rdx
  __int64 v101; // rax
  _DWORD *v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // [rsp+8h] [rbp-378h]
  __int64 v106; // [rsp+10h] [rbp-370h]
  __int64 v107; // [rsp+10h] [rbp-370h]
  __int64 v108; // [rsp+20h] [rbp-360h]
  __int64 v109; // [rsp+28h] [rbp-358h]
  __int64 v110[2]; // [rsp+30h] [rbp-350h] BYREF
  __int64 v111; // [rsp+40h] [rbp-340h]
  __int64 v112; // [rsp+50h] [rbp-330h] BYREF
  _QWORD *v113; // [rsp+58h] [rbp-328h]
  _QWORD *v114; // [rsp+60h] [rbp-320h]
  __int64 v115; // [rsp+68h] [rbp-318h]
  int v116; // [rsp+70h] [rbp-310h]
  _QWORD v117[8]; // [rsp+78h] [rbp-308h] BYREF
  const __m128i *v118; // [rsp+B8h] [rbp-2C8h] BYREF
  const __m128i *v119; // [rsp+C0h] [rbp-2C0h]
  __int64 v120; // [rsp+C8h] [rbp-2B8h]
  __int64 v121[16]; // [rsp+D0h] [rbp-2B0h] BYREF
  const char *v122; // [rsp+150h] [rbp-230h] BYREF
  __int64 v123; // [rsp+158h] [rbp-228h]
  unsigned __int64 v124; // [rsp+160h] [rbp-220h]
  _BYTE v125[64]; // [rsp+178h] [rbp-208h] BYREF
  __m128i *v126; // [rsp+1B8h] [rbp-1C8h]
  __m128i *v127; // [rsp+1C0h] [rbp-1C0h]
  __int8 *v128; // [rsp+1C8h] [rbp-1B8h]
  const char *v129; // [rsp+1D0h] [rbp-1B0h] BYREF
  __int64 v130; // [rsp+1D8h] [rbp-1A8h]
  unsigned __int64 v131; // [rsp+1E0h] [rbp-1A0h]
  char v132[64]; // [rsp+1F8h] [rbp-188h] BYREF
  __m128i *v133; // [rsp+238h] [rbp-148h]
  __m128i *v134; // [rsp+240h] [rbp-140h]
  __int8 *v135; // [rsp+248h] [rbp-138h]
  __m128i v136; // [rsp+250h] [rbp-130h] BYREF
  __int64 *v137; // [rsp+260h] [rbp-120h]
  const char **v138; // [rsp+268h] [rbp-118h]
  char v139[64]; // [rsp+278h] [rbp-108h] BYREF
  __m128i *v140; // [rsp+2B8h] [rbp-C8h]
  __m128i *v141; // [rsp+2C0h] [rbp-C0h]
  __int8 *v142; // [rsp+2C8h] [rbp-B8h]
  __m128i v143; // [rsp+2D0h] [rbp-B0h] BYREF
  __int64 *v144; // [rsp+2E0h] [rbp-A0h]
  const char **v145; // [rsp+2E8h] [rbp-98h]
  char v146[64]; // [rsp+2F8h] [rbp-88h] BYREF
  __m128i *v147; // [rsp+338h] [rbp-48h]
  __m128i *v148; // [rsp+340h] [rbp-40h]
  __int8 *v149; // [rsp+348h] [rbp-38h]

  sub_1ED7320((__int64 *)(a1 + 232), (unsigned __int64)a2);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
    goto LABEL_166;
  while ( *(_UNKNOWN **)v5 != &unk_4FC62EC )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_166;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4FC62EC);
  v8 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 328) = v7;
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
    goto LABEL_166;
  while ( *(_UNKNOWN **)v9 != &unk_4FC71EC )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_166;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4FC71EC);
  v12 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 336) = v11;
  *(_QWORD *)(a1 + 352) = 0;
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
    goto LABEL_166;
  while ( *(_UNKNOWN **)v13 != &unk_4FC453D )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_166;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4FC453D);
  v16 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 360) = v15;
  v17 = *v16;
  v18 = v16[1];
  if ( v17 == v18 )
    goto LABEL_166;
  while ( *(_UNKNOWN **)v17 != &unk_4FC6A0C )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_166;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4FC6A0C);
  v20 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 368) = v19;
  v21 = *v20;
  v22 = v20[1];
  if ( v21 == v22 )
LABEL_166:
    BUG();
  while ( *(_UNKNOWN **)v21 != &unk_4FC6AEC )
  {
    v21 += 16;
    if ( v22 == v21 )
      goto LABEL_166;
  }
  v23 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(*(_QWORD *)(v21 + 8), &unk_4FC6AEC);
  v24 = *(_QWORD *)(a1 + 360);
  *(_QWORD *)(a1 + 376) = *(_QWORD *)(v23 + 232);
  *(_QWORD *)(a1 + 384) = sub_1DDC5F0(v24);
  v25 = a2[2];
  v26 = *(__int64 (**)())(*(_QWORD *)v25 + 40LL);
  if ( v26 == sub_1D00B00 )
    BUG();
  v27 = ((__int64 (__fastcall *)(__int64))v26)(a2[2]);
  *(_DWORD *)(a1 + 392) = *(_DWORD *)(v27 + 36);
  *(_DWORD *)(a1 + 396) = *(_DWORD *)(v27 + 40);
  v28 = *(__int64 (**)())(*(_QWORD *)v25 + 56LL);
  if ( v28 == sub_1D12D20 )
    BUG();
  v29 = ((__int64 (__fastcall *)(__int64))v28)(v25);
  v30 = *(_DWORD *)(a1 + 424);
  *(_DWORD *)(a1 + 400) = *(_DWORD *)(v29 + 112);
  v31 = a2[41];
  ++*(_QWORD *)(a1 + 416);
  v32 = v30 >> 1;
  *(_QWORD *)(a1 + 408) = v31;
  if ( v32 )
  {
    if ( (*(_BYTE *)(a1 + 424) & 1) == 0 )
    {
      v33 = 4 * v32;
      goto LABEL_26;
    }
LABEL_106:
    v35 = (void *)(a1 + 432);
    v36 = 64;
    goto LABEL_29;
  }
  if ( !*(_DWORD *)(a1 + 428) )
    goto LABEL_31;
  v33 = 0;
  if ( (*(_BYTE *)(a1 + 424) & 1) != 0 )
    goto LABEL_106;
LABEL_26:
  v34 = *(unsigned int *)(a1 + 440);
  if ( (unsigned int)v34 <= v33 || (unsigned int)v34 <= 0x40 )
  {
    v35 = *(void **)(a1 + 432);
    v36 = 4LL * (unsigned int)v34;
    if ( !v36 )
    {
LABEL_30:
      *(_QWORD *)(a1 + 424) &= 1uLL;
      goto LABEL_31;
    }
LABEL_29:
    memset(v35, 255, v36);
    goto LABEL_30;
  }
  if ( !v32 || (v86 = v32 - 1) == 0 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 432));
    *(_BYTE *)(a1 + 424) |= 1u;
LABEL_133:
    v91 = (*(_QWORD *)(a1 + 424) & 1LL) == 0;
    *(_QWORD *)(a1 + 424) &= 1uLL;
    if ( v91 )
    {
      v92 = *(_DWORD **)(a1 + 432);
      v93 = *(unsigned int *)(a1 + 440);
    }
    else
    {
      v92 = (_DWORD *)(a1 + 432);
      v93 = 16;
    }
    for ( i = &v92[v93]; i != v92; ++v92 )
    {
      if ( v92 )
        *v92 = -1;
    }
    goto LABEL_31;
  }
  _BitScanReverse(&v86, v86);
  v87 = 1 << (33 - (v86 ^ 0x1F));
  if ( v87 - 17 <= 0x2E )
  {
    v87 = 64;
    j___libc_free_0(*(_QWORD *)(a1 + 432));
    v88 = *(_BYTE *)(a1 + 424);
    v89 = 256;
LABEL_132:
    *(_BYTE *)(a1 + 424) = v88 & 0xFE;
    v90 = sub_22077B0(v89);
    *(_DWORD *)(a1 + 440) = v87;
    *(_QWORD *)(a1 + 432) = v90;
    goto LABEL_133;
  }
  if ( (_DWORD)v34 != v87 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 432));
    v88 = *(_BYTE *)(a1 + 424) | 1;
    *(_BYTE *)(a1 + 424) = v88;
    if ( v87 <= 0x10 )
      goto LABEL_133;
    v89 = 4LL * v87;
    goto LABEL_132;
  }
  v91 = (*(_QWORD *)(a1 + 424) & 1LL) == 0;
  *(_QWORD *)(a1 + 424) &= 1uLL;
  if ( v91 )
  {
    v100 = *(_DWORD **)(a1 + 432);
    v101 = v34;
  }
  else
  {
    v100 = (_DWORD *)(a1 + 432);
    v101 = 16;
  }
  v102 = &v100[v101];
  do
  {
    if ( v100 )
      *v100 = -1;
    ++v100;
  }
  while ( v102 != v100 );
LABEL_31:
  memset(v121, 0, sizeof(v121));
  *(_DWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 576) = a2;
  v37 = a2[41];
  v121[1] = (__int64)&v121[5];
  v121[2] = (__int64)&v121[5];
  LODWORD(v121[3]) = 8;
  v113 = v117;
  v114 = v117;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v38 = *(_QWORD *)(v37 + 88);
  v117[0] = v37;
  v143.m128i_i64[0] = v37;
  v115 = 0x100000008LL;
  v143.m128i_i64[1] = v38;
  v110[0] = 0;
  v110[1] = 0;
  v111 = 0;
  v116 = 0;
  v112 = 1;
  sub_1D530F0(&v118, 0, &v143);
  sub_1D53270((__int64)&v112);
  v39 = (const char **)&v136;
  sub_16CCCB0(&v136, (__int64)v139, (__int64)v121);
  v40 = v121[14];
  v41 = (const __m128i *)v121[13];
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v42 = (_BYTE *)(v121[14] - v121[13]);
  if ( v121[14] == v121[13] )
  {
    v44 = 0;
    v43 = 0;
  }
  else
  {
    if ( (unsigned __int64)v42 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_157;
    v108 = v121[14] - v121[13];
    v43 = (__m128i *)sub_22077B0(v121[14] - v121[13]);
    v40 = v121[14];
    v41 = (const __m128i *)v121[13];
    v44 = v108;
  }
  v140 = v43;
  v141 = v43;
  v142 = &v43->m128i_i8[v44];
  if ( (const __m128i *)v40 == v41 )
  {
    v45 = v43;
  }
  else
  {
    v45 = (__m128i *)((char *)v43 + v40 - (_QWORD)v41);
    do
    {
      if ( v43 )
        *v43 = _mm_loadu_si128(v41);
      ++v43;
      ++v41;
    }
    while ( v43 != v45 );
  }
  v141 = v45;
  sub_16CCEE0(&v143, (__int64)v146, 8, (__int64)&v136);
  v46 = v140;
  v42 = v125;
  v140 = 0;
  v147 = v46;
  v47 = v141;
  v141 = 0;
  v148 = v47;
  v48 = v142;
  v142 = 0;
  v149 = v48;
  v39 = &v122;
  sub_16CCCB0(&v122, (__int64)v125, (__int64)&v112);
  v49 = v119;
  v41 = v118;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v50 = (char *)v119 - (char *)v118;
  if ( v119 != v118 )
  {
    if ( v50 <= 0x7FFFFFFFFFFFFFF0LL )
    {
      v51 = (__m128i *)sub_22077B0((char *)v119 - (char *)v118);
      v49 = v119;
      v41 = v118;
      goto LABEL_42;
    }
LABEL_157:
    sub_4261EA(v39, v42, v41);
  }
  v51 = 0;
LABEL_42:
  v126 = v51;
  v127 = v51;
  v128 = &v51->m128i_i8[v50];
  if ( v49 == v41 )
  {
    v52 = v51;
  }
  else
  {
    v52 = (__m128i *)((char *)v51 + (char *)v49 - (char *)v41);
    do
    {
      if ( v51 )
        *v51 = _mm_loadu_si128(v41);
      ++v51;
      ++v41;
    }
    while ( v51 != v52 );
  }
  v127 = v52;
  sub_16CCEE0(&v129, (__int64)v132, 8, (__int64)&v122);
  v53 = v126;
  v126 = 0;
  v133 = v53;
  v54 = v127;
  v127 = 0;
  v134 = v54;
  v55 = v128;
  v128 = 0;
  v135 = v55;
  sub_1D533B0((__int64)&v129, (__int64)&v143, (__int64)v110);
  if ( v133 )
    j_j___libc_free_0(v133, v135 - (__int8 *)v133);
  if ( v131 != v130 )
    _libc_free(v131);
  if ( v126 )
    j_j___libc_free_0(v126, v128 - (__int8 *)v126);
  if ( v124 != v123 )
    _libc_free(v124);
  if ( v147 )
    j_j___libc_free_0(v147, v149 - (__int8 *)v147);
  if ( v144 != (__int64 *)v143.m128i_i64[1] )
    _libc_free((unsigned __int64)v144);
  if ( v140 )
    j_j___libc_free_0(v140, v142 - (__int8 *)v140);
  if ( v137 != (__int64 *)v136.m128i_i64[1] )
    _libc_free((unsigned __int64)v137);
  if ( v118 )
    j_j___libc_free_0(v118, v120 - (_QWORD)v118);
  if ( v114 != v113 )
    _libc_free((unsigned __int64)v114);
  if ( v121[13] )
    j_j___libc_free_0(v121[13], v121[15] - v121[13]);
  if ( v121[2] != v121[1] )
    _libc_free(v121[2]);
  if ( (unsigned __int8)sub_1F0CAB0(v110, *(_QWORD *)(a1 + 368)) )
  {
    v75 = a2[41];
    v76 = sub_1626D20(*a2);
    sub_15C9150((const char **)&v136, v76);
    v143.m128i_i64[0] = (__int64)&v129;
    v77 = *(__int64 ***)(a1 + 376);
    v129 = "UnsupportedIrreducibleCFG";
    v122 = "Irreducible CFGs are not supported yet.";
    v143.m128i_i64[1] = (__int64)&v136;
    v144 = v121;
    v130 = 25;
    v145 = &v122;
    v123 = 39;
    v121[0] = v75;
    sub_1F0CE40(v77, v76, v78, v79, v80, v81, (__int64 *)&v129, &v136, v121, (__int64)&v122);
    goto LABEL_108;
  }
  v56 = *(__int64 (**)())(*(_QWORD *)a2[2] + 112LL);
  if ( v56 == sub_1D00B10 )
    goto LABEL_166;
  v57 = v56();
  v58 = *(__int64 (**)())(*(_QWORD *)v57 + 280LL);
  if ( v58 == sub_1EAD560 || !((unsigned __int8 (__fastcall *)(__int64, __int64 *))v58)(v57, a2) )
  {
    v59 = 0;
  }
  else
  {
    v59 = (_QWORD *)sub_22077B0(200);
    if ( v59 )
    {
      memset(v59, 0, 0xC8u);
      v59[6] = v59 + 8;
      v59[7] = 0x200000000LL;
    }
  }
  v60 = a2[41];
  if ( (__int64 *)v60 != a2 + 40 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(v60 + 183) )
      {
        v109 = v60;
        v95 = (_QWORD *)(*(_QWORD *)(v60 + 32) + 64LL);
        sub_15C9090((__int64)&v143, v95);
        v136.m128i_i64[0] = (__int64)&v129;
        v129 = "UnsupportedEHFunclets";
        v96 = *(__int64 ***)(a1 + 376);
        v122 = "EH Funclets are not supported yet.";
        v136.m128i_i64[1] = (__int64)&v143;
        v137 = v121;
        v130 = 21;
        v138 = &v122;
        v123 = 34;
        v121[0] = v109;
        sub_1F0CE40(v96, (__int64)v95, v97, v109, v98, v99, (__int64 *)&v129, &v143, v121, (__int64)&v122);
        goto LABEL_98;
      }
      if ( *(_BYTE *)(v60 + 180) )
        goto LABEL_78;
      v82 = *(_QWORD *)(v60 + 32);
      v83 = v60 + 24;
      if ( v82 != v60 + 24 )
        break;
LABEL_81:
      v60 = *(_QWORD *)(v60 + 8);
      if ( a2 + 40 == (__int64 *)v60 )
        goto LABEL_82;
    }
    while ( 1 )
    {
      v107 = v83;
      v84 = **(unsigned __int16 **)(v82 + 16);
      if ( v84 == *(_DWORD *)(a1 + 392) )
        break;
      if ( v84 == *(_DWORD *)(a1 + 396) )
        break;
      v105 = v60;
      v85 = sub_1F0D660(a1, v82, (__int64)v59);
      v60 = v105;
      if ( v85 )
        break;
      v83 = v107;
      if ( (*(_BYTE *)v82 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v82 + 46) & 8) != 0 )
          v82 = *(_QWORD *)(v82 + 8);
      }
      v82 = *(_QWORD *)(v82 + 8);
      if ( v107 == v82 )
        goto LABEL_81;
    }
LABEL_78:
    v106 = v60;
    sub_1F0DCB0(a1, v60, (__int64)v59);
    v61 = *(_QWORD *)(a1 + 344);
    if ( *(_QWORD *)(a1 + 408) == v61 )
      goto LABEL_98;
    if ( !v61 )
      goto LABEL_98;
    v60 = v106;
    if ( !*(_QWORD *)(a1 + 352) )
      goto LABEL_98;
    goto LABEL_81;
  }
LABEL_82:
  v62 = *(_QWORD *)(a1 + 344);
  if ( *(_QWORD *)(a1 + 408) == v62 || !v62 || !*(_QWORD *)(a1 + 352) )
    goto LABEL_98;
  v63 = 0;
  v64 = *(__int64 (**)(void))(*(_QWORD *)a2[2] + 48LL);
  if ( v64 != sub_1D90020 )
  {
    v104 = v64();
    v62 = *(_QWORD *)(a1 + 344);
    v63 = v104;
  }
  while ( 1 )
  {
    v67 = *(_QWORD *)(a1 + 384);
    if ( v67 >= sub_1DDC3C0(*(_QWORD *)(a1 + 360), v62) )
    {
      v68 = *(_QWORD *)(a1 + 384);
      if ( v68 >= sub_1DDC3C0(*(_QWORD *)(a1 + 360), *(_QWORD *)(a1 + 352)) )
      {
        v69 = *(_QWORD *)v63;
        v70 = *(__int64 (**)())(*(_QWORD *)v63 + 232LL);
        if ( v70 == sub_1F0BF30 )
          goto LABEL_93;
        if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v70)(v63, *(_QWORD *)(a1 + 344)) )
          break;
      }
    }
    v65 = sub_1F0C1B0(
            *(_QWORD *)(a1 + 344),
            *(__int64 **)(*(_QWORD *)(a1 + 344) + 64LL),
            *(__int64 **)(*(_QWORD *)(a1 + 344) + 72LL),
            *(_QWORD *)(a1 + 328));
    *(_QWORD *)(a1 + 344) = v65;
    v66 = v65;
    if ( !v65 )
      goto LABEL_98;
LABEL_88:
    sub_1F0DCB0(a1, v66, (__int64)v59);
    v62 = *(_QWORD *)(a1 + 344);
    if ( !v62 || !*(_QWORD *)(a1 + 352) )
      goto LABEL_98;
  }
  v69 = *(_QWORD *)v63;
LABEL_93:
  v71 = *(__int64 (**)())(v69 + 240);
  if ( v71 != sub_1F0BF40 && !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v71)(v63, *(_QWORD *)(a1 + 352)) )
  {
    v103 = sub_1F0C4C0(
             *(_QWORD *)(a1 + 352),
             *(__int64 **)(*(_QWORD *)(a1 + 352) + 88LL),
             *(__int64 **)(*(_QWORD *)(a1 + 352) + 96LL),
             *(_QWORD *)(a1 + 336));
    *(_QWORD *)(a1 + 352) = v103;
    v66 = v103;
    if ( !v103 )
      goto LABEL_98;
    goto LABEL_88;
  }
  v72 = *(_QWORD *)(a1 + 344);
  if ( *(_QWORD *)(a1 + 408) != v72 && v72 && *(_QWORD *)(a1 + 352) )
  {
    v73 = a2[7];
    *(_QWORD *)(v73 + 664) = v72;
    *(_QWORD *)(v73 + 672) = *(_QWORD *)(a1 + 352);
  }
LABEL_98:
  if ( v59 )
  {
    _libc_free(v59[22]);
    _libc_free(v59[19]);
    _libc_free(v59[16]);
    _libc_free(v59[13]);
    v74 = v59[6];
    if ( (_QWORD *)v74 != v59 + 8 )
      _libc_free(v74);
    j_j___libc_free_0(v59, 200);
  }
LABEL_108:
  if ( v110[0] )
    j_j___libc_free_0(v110[0], v111 - v110[0]);
}
