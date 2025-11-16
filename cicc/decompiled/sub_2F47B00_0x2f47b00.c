// Function: sub_2F47B00
// Address: 0x2f47b00
//
__int64 __fastcall sub_2F47B00(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned __int32 a4,
        unsigned __int8 a5,
        __int64 a6)
{
  __int64 v7; // r12
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rsi
  _WORD *v13; // r8
  unsigned int v14; // ecx
  unsigned int v15; // eax
  __int64 v16; // rdi
  const __m128i *v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  unsigned int v21; // ecx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // r14
  _BOOL4 v25; // r15d
  unsigned __int64 v26; // rax
  unsigned int v27; // esi
  __int64 v28; // r9
  int v29; // r14d
  __int64 v30; // r8
  __int64 v31; // rcx
  _DWORD *v32; // rdx
  _DWORD *v33; // rax
  int v34; // edi
  __int64 *v35; // rbx
  __int64 v36; // r12
  int v37; // esi
  __int64 *v38; // rdi
  unsigned int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // rax
  __int64 *v42; // r13
  __int64 v43; // rax
  __int64 v44; // r14
  __int64 v45; // r13
  unsigned int v46; // esi
  unsigned __int32 v47; // eax
  __int64 *v48; // r15
  unsigned __int32 v49; // edx
  unsigned int v50; // edi
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // rsi
  __int64 *v53; // rsi
  char **v54; // rcx
  char **v55; // rdx
  __int64 *v56; // rbx
  unsigned __int64 v57; // r14
  unsigned __int64 v58; // rdi
  __int64 *v59; // r13
  __int64 v60; // r12
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  unsigned __int64 v64; // rcx
  unsigned int v65; // esi
  __int64 v66; // r8
  int v67; // r14d
  _DWORD *v68; // rax
  __int64 v69; // rcx
  _DWORD *v70; // rdx
  int v71; // edi
  __m128i *v72; // rax
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  __int64 v75; // rcx
  __m128i *v76; // rdx
  __m128i *v77; // rax
  __int64 v78; // rax
  __int64 v79; // rbx
  __int64 v80; // r13
  __int64 v81; // r15
  unsigned int v82; // r12d
  __m128i *v83; // r8
  int v84; // ecx
  __int64 *v85; // rdi
  unsigned int v86; // edx
  __int64 v87; // rsi
  int v88; // ecx
  __int64 *v89; // rdi
  unsigned int v90; // edx
  __int64 v91; // rsi
  int v92; // edi
  int v93; // edx
  _DWORD *v94; // rdx
  _QWORD *v95; // rax
  int v96; // r11d
  int v97; // r11d
  __int64 v98; // r10
  unsigned int v99; // ecx
  int v100; // edi
  _DWORD *v101; // rsi
  int v102; // r10d
  int v103; // r10d
  _DWORD *v104; // rcx
  unsigned int v105; // ebx
  int v106; // esi
  int v107; // edi
  unsigned __int64 v108; // r14
  __int64 v109; // rdi
  const void *v110; // rsi
  int v111; // edi
  int v112; // edx
  int v113; // r9d
  int v114; // r9d
  __int64 v115; // r10
  unsigned int v116; // ecx
  int v117; // r8d
  int v118; // edi
  _DWORD *v119; // rsi
  int v120; // r8d
  int v121; // r8d
  __int64 v122; // r9
  _DWORD *v123; // rcx
  unsigned int v124; // ebx
  int v125; // esi
  int v126; // edi
  __int64 *v127; // [rsp+8h] [rbp-218h]
  __int64 v128; // [rsp+20h] [rbp-200h]
  const __m128i *v129; // [rsp+28h] [rbp-1F8h]
  _QWORD *v130; // [rsp+30h] [rbp-1F0h]
  unsigned __int32 v131; // [rsp+38h] [rbp-1E8h]
  _BOOL4 v132; // [rsp+3Ch] [rbp-1E4h]
  _DWORD *v133; // [rsp+48h] [rbp-1D8h]
  __int64 v134; // [rsp+50h] [rbp-1D0h]
  __int64 v135; // [rsp+58h] [rbp-1C8h]
  __int64 *v136; // [rsp+60h] [rbp-1C0h]
  const __m128i *v137; // [rsp+60h] [rbp-1C0h]
  __int8 v138; // [rsp+68h] [rbp-1B8h]
  unsigned __int64 v139; // [rsp+68h] [rbp-1B8h]
  __int64 *v140; // [rsp+70h] [rbp-1B0h]
  __int64 v141; // [rsp+70h] [rbp-1B0h]
  __int64 v142; // [rsp+78h] [rbp-1A8h]
  unsigned __int16 v143; // [rsp+82h] [rbp-19Eh]
  __int64 v146; // [rsp+D0h] [rbp-150h] BYREF
  _BYTE *v147; // [rsp+D8h] [rbp-148h]
  __int64 v148; // [rsp+E0h] [rbp-140h]
  _BYTE v149[56]; // [rsp+E8h] [rbp-138h] BYREF
  __m128i v150; // [rsp+120h] [rbp-100h] BYREF
  __int64 v151; // [rsp+130h] [rbp-F0h] BYREF
  unsigned int v152; // [rsp+138h] [rbp-E8h]
  __int64 v153; // [rsp+140h] [rbp-E0h]
  __int64 *v154; // [rsp+150h] [rbp-D0h] BYREF
  __int64 v155; // [rsp+158h] [rbp-C8h]
  _BYTE v156[192]; // [rsp+160h] [rbp-C0h] BYREF

  v7 = a1;
  if ( *(_QWORD *)(a1 + 368) )
  {
    v9 = *(_QWORD *)(a1 + 8);
    v10 = a1 + 352;
    v11 = *(_QWORD *)(v10 - 336);
    v150.m128i_i32[0] = a4;
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __m128i *))(v7 + 376))(v10, v11, v9, &v150) )
      return 0;
  }
  v150.m128i_i64[0] = 0;
  LOBYTE(v151) = 0;
  v142 = *(_QWORD *)(a2 + 32) + 40LL * a3;
  v150.m128i_i64[1] = a4;
  v135 = a4 & 0x7FFFFFFF;
  v13 = (_WORD *)(*(_QWORD *)(v7 + 624) + 2 * v135);
  v14 = *(_DWORD *)(v7 + 424);
  v15 = (unsigned __int16)*v13;
  if ( v15 >= v14 )
    goto LABEL_84;
  v16 = *(_QWORD *)(v7 + 416);
  while ( 1 )
  {
    v17 = (const __m128i *)(v16 + 24LL * v15);
    if ( (a4 & 0x7FFFFFFF) == (v17->m128i_i32[2] & 0x7FFFFFFF) )
      break;
    v15 += 0x10000;
    if ( v14 <= v15 )
      goto LABEL_84;
  }
  if ( v17 == (const __m128i *)(v16 + 24LL * v14) )
  {
LABEL_84:
    *v13 = v14;
    v73 = *(unsigned int *)(v7 + 424);
    v74 = v73 + 1;
    if ( v73 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 428) )
    {
      v108 = *(_QWORD *)(v7 + 416);
      v109 = v7 + 416;
      v110 = (const void *)(v7 + 432);
      if ( v108 > (unsigned __int64)&v150 || (unsigned __int64)&v150 >= v108 + 24 * v73 )
      {
        sub_C8D5F0(v109, v110, v74, 0x18u, (__int64)v13, a6);
        v75 = *(_QWORD *)(v7 + 416);
        v73 = *(unsigned int *)(v7 + 424);
        v76 = &v150;
      }
      else
      {
        sub_C8D5F0(v109, v110, v74, 0x18u, (__int64)v13, a6);
        v75 = *(_QWORD *)(v7 + 416);
        v73 = *(unsigned int *)(v7 + 424);
        v76 = (__m128i *)((char *)&v150 + v75 - v108);
      }
    }
    else
    {
      v75 = *(_QWORD *)(v7 + 416);
      v76 = &v150;
    }
    v77 = (__m128i *)(v75 + 24 * v73);
    *v77 = _mm_loadu_si128(v76);
    v77[1].m128i_i64[0] = v76[1].m128i_i64[0];
    v78 = (unsigned int)(*(_DWORD *)(v7 + 424) + 1);
    *(_DWORD *)(v7 + 424) = v78;
    v17 = (const __m128i *)(*(_QWORD *)(v7 + 416) + 24 * v78 - 24);
    if ( (((*(_BYTE *)(v142 + 3) & 0x40) != 0) & (*(_BYTE *)(v142 + 3) >> 4)) == 0 )
    {
      if ( (unsigned __int8)sub_2F462B0((_QWORD *)v7, a4) )
        v17->m128i_i8[14] = 1;
      else
        *(_BYTE *)(v142 + 3) |= 0x40u;
    }
  }
  v143 = v17->m128i_u16[6];
  if ( !v143 )
  {
    sub_2F44460(v7, (_BYTE *)a2, (__int64)v17, 0, a5, a6);
    v143 = v17->m128i_u16[6];
  }
  if ( !v17->m128i_i8[15] && !v17->m128i_i8[14] )
    goto LABEL_13;
  if ( *(_WORD *)(a2 + 68) == 10 )
    goto LABEL_19;
  v23 = a2;
  if ( (*(_BYTE *)a2 & 4) == 0 && (*(_BYTE *)(a2 + 44) & 8) != 0 )
  {
    do
      v23 = *(_QWORD *)(v23 + 8);
    while ( (*(_BYTE *)(v23 + 44) & 8) != 0 );
  }
  v24 = *(__int64 **)(v23 + 8);
  v138 = v17->m128i_i8[14];
  v25 = v17->m128i_i64[0] == 0;
  v132 = v25;
  v136 = v24;
  v131 = sub_2F41760((__int64 *)v7, a4);
  (*(void (__fastcall **)(_QWORD, _QWORD, __int64 *, _QWORD, _BOOL4, _QWORD, unsigned __int64, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(v7 + 24) + 560LL))(
    *(_QWORD *)(v7 + 24),
    *(_QWORD *)(v7 + 384),
    v24,
    v143,
    v25,
    v131,
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 8) + 56LL) + 16 * v135) & 0xFFFFFFFFFFFFFFF8LL,
    *(_QWORD *)(v7 + 16),
    a4,
    0);
  v26 = sub_2E313E0(*(_QWORD *)(v7 + 384));
  v27 = *(_DWORD *)(v7 + 696);
  v130 = (_QWORD *)v26;
  if ( !v27 )
  {
    ++*(_QWORD *)(v7 + 672);
    goto LABEL_144;
  }
  v28 = v27 - 1;
  v29 = 1;
  v30 = *(_QWORD *)(v7 + 680);
  LODWORD(v31) = v28 & (37 * a4);
  v32 = (_DWORD *)(v30 + 40LL * (unsigned int)v31);
  v33 = 0;
  v34 = *v32;
  if ( *v32 != a4 )
  {
    while ( v34 != -1 )
    {
      if ( v34 == -2 && !v33 )
        v33 = v32;
      v31 = (unsigned int)v28 & ((_DWORD)v31 + v29);
      v32 = (_DWORD *)(v30 + 40 * v31);
      v34 = *v32;
      if ( *v32 == a4 )
        goto LABEL_25;
      ++v29;
    }
    v92 = *(_DWORD *)(v7 + 688);
    if ( !v33 )
      v33 = v32;
    ++*(_QWORD *)(v7 + 672);
    v93 = v92 + 1;
    if ( 4 * (v92 + 1) < 3 * v27 )
    {
      if ( v27 - *(_DWORD *)(v7 + 692) - v93 > v27 >> 3 )
      {
LABEL_140:
        *(_DWORD *)(v7 + 688) = v93;
        if ( *v33 != -1 )
          --*(_DWORD *)(v7 + 692);
        v94 = v33 + 6;
        v95 = v33 + 2;
        *v95 = v94;
        *((_DWORD *)v95 - 2) = a4;
        v95[1] = 0x200000000LL;
        v133 = v95;
        goto LABEL_26;
      }
      sub_2F46A90(v7 + 672, v27);
      v102 = *(_DWORD *)(v7 + 696);
      if ( v102 )
      {
        v103 = v102 - 1;
        v28 = *(_QWORD *)(v7 + 680);
        v104 = 0;
        v105 = v103 & (37 * a4);
        v93 = *(_DWORD *)(v7 + 688) + 1;
        v106 = 1;
        v33 = (_DWORD *)(v28 + 40LL * v105);
        v107 = *v33;
        if ( a4 != *v33 )
        {
          while ( v107 != -1 )
          {
            if ( !v104 && v107 == -2 )
              v104 = v33;
            v30 = (unsigned int)(v106 + 1);
            v105 = v103 & (v106 + v105);
            v33 = (_DWORD *)(v28 + 40LL * v105);
            v107 = *v33;
            if ( *v33 == a4 )
              goto LABEL_140;
            ++v106;
          }
          if ( v104 )
            v33 = v104;
        }
        goto LABEL_140;
      }
LABEL_214:
      ++*(_DWORD *)(v7 + 688);
      BUG();
    }
LABEL_144:
    sub_2F46A90(v7 + 672, 2 * v27);
    v96 = *(_DWORD *)(v7 + 696);
    if ( v96 )
    {
      v97 = v96 - 1;
      v98 = *(_QWORD *)(v7 + 680);
      v93 = *(_DWORD *)(v7 + 688) + 1;
      v99 = v97 & (37 * a4);
      v33 = (_DWORD *)(v98 + 40LL * v99);
      v30 = (unsigned int)*v33;
      if ( (_DWORD)v30 != a4 )
      {
        v100 = 1;
        v101 = 0;
        while ( (_DWORD)v30 != -1 )
        {
          if ( (_DWORD)v30 == -2 && !v101 )
            v101 = v33;
          v28 = (unsigned int)(v100 + 1);
          v99 = v97 & (v100 + v99);
          v33 = (_DWORD *)(v98 + 40LL * v99);
          v30 = (unsigned int)*v33;
          if ( (_DWORD)v30 == a4 )
            goto LABEL_140;
          ++v100;
        }
        if ( v101 )
          v33 = v101;
      }
      goto LABEL_140;
    }
    goto LABEL_214;
  }
LABEL_25:
  v133 = v32 + 2;
LABEL_26:
  v150.m128i_i64[0] = 0;
  v154 = (__int64 *)v156;
  v155 = 0x200000000LL;
  v150.m128i_i64[1] = 1;
  v151 = -4096;
  v153 = -4096;
  v35 = *(__int64 **)v133;
  if ( *(_QWORD *)v133 == *(_QWORD *)v133 + 8LL * (unsigned int)v133[2] )
    goto LABEL_53;
  v128 = v7;
  v36 = *(_QWORD *)v133 + 8LL * (unsigned int)v133[2];
  v129 = v17;
  do
  {
    v44 = *v35;
    v45 = *(_QWORD *)(*v35 + 16);
    if ( (v150.m128i_i8[8] & 1) != 0 )
    {
      v37 = 1;
      v38 = &v151;
    }
    else
    {
      v46 = v152;
      v38 = (__int64 *)v151;
      if ( !v152 )
      {
        v47 = v150.m128i_u32[2];
        ++v150.m128i_i64[0];
        v48 = 0;
        v49 = ((unsigned __int32)v150.m128i_i32[2] >> 1) + 1;
LABEL_38:
        v50 = 3 * v46;
        goto LABEL_39;
      }
      v37 = v152 - 1;
    }
    v39 = v37 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
    v40 = &v38[2 * v39];
    v30 = *v40;
    if ( v45 == *v40 )
    {
LABEL_30:
      v41 = *((unsigned int *)v40 + 2);
      goto LABEL_31;
    }
    v28 = 1;
    v48 = 0;
    while ( v30 != -4096 )
    {
      if ( v30 == -8192 && !v48 )
        v48 = v40;
      v39 = v37 & (v28 + v39);
      v40 = &v38[2 * v39];
      v30 = *v40;
      if ( v45 == *v40 )
        goto LABEL_30;
      v28 = (unsigned int)(v28 + 1);
    }
    if ( !v48 )
      v48 = v40;
    v47 = v150.m128i_u32[2];
    ++v150.m128i_i64[0];
    v49 = ((unsigned __int32)v150.m128i_i32[2] >> 1) + 1;
    if ( (v150.m128i_i8[8] & 1) == 0 )
    {
      v46 = v152;
      goto LABEL_38;
    }
    v50 = 6;
    v46 = 2;
LABEL_39:
    if ( 4 * v49 >= v50 )
    {
      sub_2F476E0((__int64)&v150, 2 * v46);
      if ( (v150.m128i_i8[8] & 1) != 0 )
      {
        v84 = 1;
        v85 = &v151;
      }
      else
      {
        v85 = (__int64 *)v151;
        if ( !v152 )
          goto LABEL_215;
        v84 = v152 - 1;
      }
      v47 = v150.m128i_u32[2];
      v86 = v84 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v48 = &v85[2 * v86];
      v87 = *v48;
      if ( v45 == *v48 )
        goto LABEL_41;
      v28 = 1;
      v30 = 0;
      while ( v87 != -4096 )
      {
        if ( !v30 && v87 == -8192 )
          v30 = (__int64)v48;
        v86 = v84 & (v28 + v86);
        v48 = &v85[2 * v86];
        v87 = *v48;
        if ( v45 == *v48 )
          goto LABEL_109;
        v28 = (unsigned int)(v28 + 1);
      }
    }
    else
    {
      if ( v46 - v150.m128i_i32[3] - v49 > v46 >> 3 )
        goto LABEL_41;
      sub_2F476E0((__int64)&v150, v46);
      if ( (v150.m128i_i8[8] & 1) != 0 )
      {
        v88 = 1;
        v89 = &v151;
      }
      else
      {
        v89 = (__int64 *)v151;
        if ( !v152 )
        {
LABEL_215:
          v150.m128i_i32[2] = (2 * ((unsigned __int32)v150.m128i_i32[2] >> 1) + 2) | v150.m128i_i8[8] & 1;
          BUG();
        }
        v88 = v152 - 1;
      }
      v28 = 1;
      v30 = 0;
      v47 = v150.m128i_u32[2];
      v90 = v88 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v48 = &v89[2 * v90];
      v91 = *v48;
      if ( v45 == *v48 )
        goto LABEL_41;
      while ( v91 != -4096 )
      {
        if ( v91 == -8192 && !v30 )
          v30 = (__int64)v48;
        v90 = v88 & (v28 + v90);
        v48 = &v89[2 * v90];
        v91 = *v48;
        if ( v45 == *v48 )
          goto LABEL_109;
        v28 = (unsigned int)(v28 + 1);
      }
    }
    if ( v30 )
      v48 = (__int64 *)v30;
LABEL_109:
    v47 = v150.m128i_u32[2];
LABEL_41:
    v150.m128i_i32[2] = (2 * (v47 >> 1) + 2) | v47 & 1;
    if ( *v48 != -4096 )
      --v150.m128i_i32[3];
    *v48 = v45;
    *((_DWORD *)v48 + 2) = 0;
    v51 = (unsigned int)v155;
    v146 = v45;
    v52 = (unsigned int)v155 + 1LL;
    v148 = 0x600000000LL;
    v41 = (unsigned int)v155;
    v147 = v149;
    if ( v52 > HIDWORD(v155) )
    {
      if ( v154 > &v146
        || (v127 = v154, v51 = (unsigned __int64)&v154[9 * (unsigned int)v155], (unsigned __int64)&v146 >= v51) )
      {
        sub_2F46DC0((__int64)&v154, v52, v51, HIDWORD(v155), v30, v28);
        v51 = (unsigned int)v155;
        v53 = v154;
        v54 = (char **)&v146;
        v41 = (unsigned int)v155;
      }
      else
      {
        sub_2F46DC0((__int64)&v154, v52, v51, HIDWORD(v155), v30, v28);
        v53 = v154;
        v51 = (unsigned int)v155;
        v41 = (unsigned int)v155;
        v54 = (char **)((char *)v154 + (char *)&v146 - (char *)v127);
      }
    }
    else
    {
      v53 = v154;
      v54 = (char **)&v146;
    }
    v55 = (char **)&v53[9 * v51];
    if ( v55 )
    {
      *v55 = *v54;
      v55[1] = (char *)(v55 + 3);
      v55[2] = (char *)0x600000000LL;
      v30 = *((unsigned int *)v54 + 4);
      if ( (_DWORD)v30 )
        sub_2F41600((__int64)(v55 + 1), v54 + 1, (__int64)v55, (__int64)v54, v30, v28);
      v41 = (unsigned int)v155;
    }
    LODWORD(v155) = v41 + 1;
    if ( v147 != v149 )
    {
      _libc_free((unsigned __int64)v147);
      v41 = (unsigned int)(v155 - 1);
    }
    *((_DWORD *)v48 + 2) = v41;
LABEL_31:
    v42 = &v154[9 * v41];
    v43 = *((unsigned int *)v42 + 4);
    if ( v43 + 1 > (unsigned __int64)*((unsigned int *)v42 + 5) )
    {
      sub_C8D5F0((__int64)(v42 + 1), v42 + 3, v43 + 1, 8u, v30, v28);
      v43 = *((unsigned int *)v42 + 4);
    }
    ++v35;
    *(_QWORD *)(v42[1] + 8 * v43) = v44;
    ++*((_DWORD *)v42 + 4);
  }
  while ( (__int64 *)v36 != v35 );
  v17 = v129;
  v7 = v128;
  if ( v154 != &v154[9 * (unsigned int)v155] )
  {
    v140 = &v154[9 * (unsigned int)v155];
    v59 = v154;
    do
    {
      v60 = *v59;
      if ( *(_WORD *)(*v59 + 68) != 15 )
      {
        v61 = sub_2E904C0(*(_QWORD *)(v128 + 384), v136, *v59, v131, (__int64)(v59 + 1));
        if ( v138 )
        {
          v134 = (__int64)sub_2E7B2C0(*(_QWORD **)(*(_QWORD *)(v128 + 384) + 32LL), v61);
          sub_2E31040((__int64 *)(*(_QWORD *)(v128 + 384) + 40LL), v134);
          v63 = *(_QWORD *)v134;
          v64 = *v130 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v134 + 8) = v130;
          *(_QWORD *)v134 = v64 | v63 & 7;
          *(_QWORD *)(v64 + 8) = v134;
          *v130 = *v130 & 7LL | v134;
        }
        if ( *(_WORD *)(v60 + 68) == 14 )
        {
          v62 = *(_QWORD *)(v60 + 32);
          if ( !*(_BYTE *)v62 && !*(_DWORD *)(v62 + 8) )
            sub_2E8DA60(v60, v131, 0);
        }
      }
      v59 += 9;
    }
    while ( v140 != v59 );
    v17 = v129;
    v7 = v128;
  }
LABEL_53:
  v133[2] = 0;
  v56 = v154;
  v57 = (unsigned __int64)&v154[9 * (unsigned int)v155];
  if ( v154 != (__int64 *)v57 )
  {
    do
    {
      v57 -= 72LL;
      v58 = *(_QWORD *)(v57 + 8);
      if ( v58 != v57 + 24 )
        _libc_free(v58);
    }
    while ( v56 != (__int64 *)v57 );
    v57 = (unsigned __int64)v154;
  }
  if ( (_BYTE *)v57 != v156 )
    _libc_free(v57);
  if ( (v150.m128i_i8[8] & 1) == 0 )
    sub_C7D6A0(v151, 16LL * v152, 8);
  if ( *(_WORD *)(a2 + 68) == 2 )
  {
    v139 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 8) + 56LL) + 16 * v135) & 0xFFFFFFFFFFFFFFF8LL;
    v79 = *(_QWORD *)(a2 + 32);
    if ( v79 != v79 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF) )
    {
      v137 = v17;
      v80 = v79 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
      v81 = v7;
      v82 = *(_DWORD *)(*(_QWORD *)(v7 + 392) + 4 * v135);
      do
      {
        if ( *(_BYTE *)v79 == 4 )
        {
          v141 = *(_QWORD *)(v79 + 24);
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, _BOOL4, _QWORD, unsigned __int64, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(v81 + 24) + 560LL))(
            *(_QWORD *)(v81 + 24),
            v141,
            *(_QWORD *)(v141 + 56),
            v143,
            v132,
            v82,
            v139,
            *(_QWORD *)(v81 + 16),
            a4,
            0);
          v150.m128i_i32[0] = v143;
          v150.m128i_i64[1] = -1;
          v151 = -1;
          v83 = *(__m128i **)(v141 + 192);
          if ( v83 == *(__m128i **)(v141 + 200) )
          {
            sub_2E341F0((unsigned __int64 *)(v141 + 184), *(const __m128i **)(v141 + 192), &v150);
          }
          else
          {
            if ( v83 )
            {
              *v83 = _mm_loadu_si128(&v150);
              v83[1].m128i_i64[0] = v151;
              v83 = *(__m128i **)(v141 + 192);
            }
            *(_QWORD *)(v141 + 192) = (char *)v83 + 24;
          }
        }
        v79 += 40;
      }
      while ( v80 != v79 );
      v17 = v137;
      v7 = v81;
    }
  }
  v17->m128i_i64[0] = 0;
LABEL_19:
  v17->m128i_i16[7] = 0;
LABEL_13:
  if ( *(_WORD *)(a2 + 68) == 21 )
  {
    v65 = *(_DWORD *)(v7 + 664);
    if ( v65 )
    {
      v66 = *(_QWORD *)(v7 + 648);
      v67 = 1;
      v68 = 0;
      LODWORD(v69) = (v65 - 1) & (37 * a4);
      v70 = (_DWORD *)(v66 + 32LL * (unsigned int)v69);
      v71 = *v70;
      if ( a4 == *v70 )
      {
LABEL_82:
        v72 = (__m128i *)(v70 + 2);
LABEL_83:
        *v72 = _mm_loadu_si128(v17);
        v72[1].m128i_i8[0] = v17[1].m128i_i8[0];
        goto LABEL_14;
      }
      while ( v71 != -1 )
      {
        if ( v71 == -2 && !v68 )
          v68 = v70;
        v69 = (v65 - 1) & ((_DWORD)v69 + v67);
        v70 = (_DWORD *)(v66 + 32 * v69);
        v71 = *v70;
        if ( *v70 == a4 )
          goto LABEL_82;
        ++v67;
      }
      v111 = *(_DWORD *)(v7 + 656);
      if ( !v68 )
        v68 = v70;
      ++*(_QWORD *)(v7 + 640);
      v112 = v111 + 1;
      if ( 4 * (v111 + 1) < 3 * v65 )
      {
        if ( v65 - *(_DWORD *)(v7 + 660) - v112 > v65 >> 3 )
        {
LABEL_171:
          *(_DWORD *)(v7 + 656) = v112;
          if ( *v68 != -1 )
            --*(_DWORD *)(v7 + 660);
          *((_QWORD *)v68 + 1) = 0;
          v72 = (__m128i *)(v68 + 2);
          v72->m128i_i64[1] = 0;
          v72[-1].m128i_i32[2] = a4;
          v72[1].m128i_i8[0] = 0;
          goto LABEL_83;
        }
        sub_2F42650(v7 + 640, v65);
        v120 = *(_DWORD *)(v7 + 664);
        if ( v120 )
        {
          v121 = v120 - 1;
          v122 = *(_QWORD *)(v7 + 648);
          v123 = 0;
          v124 = v121 & (37 * a4);
          v112 = *(_DWORD *)(v7 + 656) + 1;
          v125 = 1;
          v68 = (_DWORD *)(v122 + 32LL * v124);
          v126 = *v68;
          if ( *v68 != a4 )
          {
            while ( v126 != -1 )
            {
              if ( !v123 && v126 == -2 )
                v123 = v68;
              v124 = v121 & (v125 + v124);
              v68 = (_DWORD *)(v122 + 32LL * v124);
              v126 = *v68;
              if ( a4 == *v68 )
                goto LABEL_171;
              ++v125;
            }
            if ( v123 )
              v68 = v123;
          }
          goto LABEL_171;
        }
LABEL_216:
        ++*(_DWORD *)(v7 + 656);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v7 + 640);
    }
    sub_2F42650(v7 + 640, 2 * v65);
    v113 = *(_DWORD *)(v7 + 664);
    if ( v113 )
    {
      v114 = v113 - 1;
      v115 = *(_QWORD *)(v7 + 648);
      v112 = *(_DWORD *)(v7 + 656) + 1;
      v116 = v114 & (37 * a4);
      v68 = (_DWORD *)(v115 + 32LL * v116);
      v117 = *v68;
      if ( *v68 != a4 )
      {
        v118 = 1;
        v119 = 0;
        while ( v117 != -1 )
        {
          if ( v117 == -2 && !v119 )
            v119 = v68;
          v116 = v114 & (v118 + v116);
          v68 = (_DWORD *)(v115 + 32LL * v116);
          v117 = *v68;
          if ( *v68 == a4 )
            goto LABEL_171;
          ++v118;
        }
        if ( v119 )
          v68 = v119;
      }
      goto LABEL_171;
    }
    goto LABEL_216;
  }
LABEL_14:
  v18 = *(_QWORD *)(v7 + 16);
  v19 = *(_QWORD *)(v18 + 8);
  v20 = *(_DWORD *)(v19 + 24LL * v143 + 16) >> 12;
  v21 = *(_DWORD *)(v19 + 24LL * v143 + 16) & 0xFFF;
  v22 = *(_QWORD *)(v18 + 56) + 2 * v20;
  do
  {
    if ( !v22 )
      break;
    v22 += 2;
    *(_DWORD *)(*(_QWORD *)(v7 + 1112) + 4LL * v21) = *(_DWORD *)(v7 + 1104) | 1;
    v21 += *(__int16 *)(v22 - 2);
  }
  while ( *(_WORD *)(v22 - 2) );
  return sub_2F41240(v7, a2, v142, (__int64)v17);
}
