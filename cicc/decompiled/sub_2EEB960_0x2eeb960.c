// Function: sub_2EEB960
// Address: 0x2eeb960
//
void __fastcall sub_2EEB960(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r12
  int v7; // edx
  __int64 v8; // rcx
  unsigned int *v9; // rbx
  unsigned int v10; // r13d
  __int64 v11; // r11
  unsigned int v12; // r8d
  __int64 v13; // r9
  __int64 v14; // rcx
  unsigned int *v15; // r10
  __int64 v16; // rsi
  __int64 v17; // rax
  unsigned int v18; // edx
  unsigned int v19; // edi
  __int64 v20; // rdi
  int v21; // r15d
  __int64 *v22; // r9
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // rcx
  unsigned int *v26; // rax
  unsigned int v27; // r13d
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // rdi
  unsigned int v31; // r15d
  int v32; // eax
  unsigned __int64 v33; // rax
  int v34; // eax
  int v35; // eax
  int v36; // r8d
  int v37; // r8d
  __int64 v38; // r9
  unsigned int v39; // ecx
  int v40; // edx
  __int64 v41; // r11
  int v42; // edi
  __int64 *v43; // rsi
  char v44; // al
  unsigned int **v45; // r9
  __int64 v46; // rax
  __int64 v47; // r8
  __int64 v48; // r15
  __int64 v49; // r12
  __int64 v50; // r14
  unsigned int v51; // r13d
  unsigned __int8 v52; // al
  unsigned int v53; // r10d
  __int64 v54; // rax
  unsigned __int64 v55; // rdx
  char v56; // al
  __int64 v57; // rdi
  unsigned int v58; // esi
  __int64 v59; // rcx
  unsigned int v60; // eax
  __int64 v61; // rdi
  _DWORD *v62; // rdx
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rdx
  __int64 v65; // rax
  unsigned __int64 v66; // r13
  unsigned __int64 v67; // rdx
  unsigned int *v68; // rax
  unsigned int *v69; // r9
  unsigned int *v70; // r11
  unsigned int v71; // ecx
  __int16 *v72; // rsi
  unsigned int v73; // edi
  unsigned int v74; // eax
  __int64 v75; // r8
  __m128i *v76; // rdx
  __int64 v77; // rax
  const __m128i *v78; // rax
  int v79; // edx
  unsigned int *v80; // r11
  __int64 v81; // r8
  __int64 v82; // rdi
  unsigned int *v83; // r15
  const __m128i *v84; // r9
  __int64 v85; // rcx
  __int64 v86; // r12
  unsigned int v87; // r13d
  __int64 v88; // rbx
  unsigned int v89; // ecx
  _BYTE *v90; // r10
  unsigned int v91; // eax
  __int64 v92; // rsi
  __int64 v93; // rdx
  __int64 v94; // rdx
  unsigned __int64 v95; // r10
  __int64 v96; // rcx
  const __m128i *v97; // rax
  __m128i *v98; // rdx
  __int64 v99; // rax
  __int64 v100; // rax
  unsigned __int64 v101; // rdx
  const void *v102; // rsi
  int v103; // esi
  int v104; // edi
  int v105; // edi
  __int64 v106; // r9
  __int64 *v107; // rcx
  unsigned int v108; // ebx
  int v109; // esi
  __int64 v110; // r8
  int v111; // edx
  __int64 v112; // [rsp+0h] [rbp-190h]
  const __m128i *v113; // [rsp+8h] [rbp-188h]
  const __m128i *v114; // [rsp+8h] [rbp-188h]
  __int64 v115; // [rsp+10h] [rbp-180h]
  unsigned int *v116; // [rsp+10h] [rbp-180h]
  unsigned int *v117; // [rsp+10h] [rbp-180h]
  unsigned int v118; // [rsp+10h] [rbp-180h]
  unsigned int **v119; // [rsp+10h] [rbp-180h]
  __int64 v120; // [rsp+18h] [rbp-178h]
  __int64 v121; // [rsp+18h] [rbp-178h]
  __int64 v122; // [rsp+20h] [rbp-170h]
  unsigned int *v123; // [rsp+38h] [rbp-158h]
  __int64 v124; // [rsp+40h] [rbp-150h]
  int v126; // [rsp+40h] [rbp-150h]
  unsigned int v128; // [rsp+50h] [rbp-140h] BYREF
  __int64 v129; // [rsp+54h] [rbp-13Ch]
  __int64 v130; // [rsp+5Ch] [rbp-134h]
  unsigned int *v131; // [rsp+70h] [rbp-120h] BYREF
  __int64 v132; // [rsp+78h] [rbp-118h]
  _BYTE v133[32]; // [rsp+80h] [rbp-110h] BYREF
  unsigned int *v134; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v135; // [rsp+A8h] [rbp-E8h]
  _BYTE v136[32]; // [rsp+B0h] [rbp-E0h] BYREF
  unsigned int *v137; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v138; // [rsp+D8h] [rbp-B8h]
  _BYTE v139[176]; // [rsp+E0h] [rbp-B0h] BYREF

  v4 = a3;
  v5 = a1;
  v7 = *(unsigned __int16 *)(a3 + 68);
  v8 = *(_QWORD *)(a1 + 440);
  v137 = (unsigned int *)v139;
  v138 = 0x800000000LL;
  if ( (_WORD)v7 == 68 || !v7 )
  {
    if ( *(_QWORD *)a2 )
    {
      sub_2EE7A10(v4, (__int64)&v137, *(_QWORD *)a2, *(_QWORD *)(v8 + 24));
      goto LABEL_5;
    }
    goto LABEL_38;
  }
  if ( (unsigned __int16)(v7 - 14) <= 4u )
  {
LABEL_38:
    v12 = *(_DWORD *)(a1 + 400);
    v11 = *(_QWORD *)(a1 + 384);
    v10 = 0;
    v20 = a1 + 376;
    if ( v12 )
      goto LABEL_15;
LABEL_39:
    ++*(_QWORD *)(v5 + 376);
    goto LABEL_40;
  }
  v44 = sub_2EE7830(v4, (__int64)&v137, *(_QWORD *)(v8 + 24));
  v45 = &v137;
  if ( !v44 )
    goto LABEL_5;
  v46 = *(_QWORD *)(a1 + 440);
  v47 = *(_QWORD *)(v4 + 32);
  v132 = 0x800000000LL;
  v135 = 0x800000000LL;
  v48 = *(_QWORD *)(v46 + 16);
  v131 = (unsigned int *)v133;
  v134 = (unsigned int *)v136;
  if ( v47 == v47 + 40LL * (*(_DWORD *)(v4 + 40) & 0xFFFFFF) )
    goto LABEL_98;
  v49 = v47 + 40LL * (*(_DWORD *)(v4 + 40) & 0xFFFFFF);
  v122 = v4;
  v50 = v47;
  do
  {
    if ( *(_BYTE *)v50 )
      goto LABEL_70;
    v51 = *(_DWORD *)(v50 + 8);
    if ( v51 - 1 > 0x3FFFFFFE )
      goto LABEL_70;
    v52 = *(_BYTE *)(v50 + 3);
    if ( (v52 & 0x10) != 0 )
    {
      if ( (((*(_BYTE *)(v50 + 3) & 0x40) != 0) & (v52 >> 4)) == 0 )
      {
        v53 = sub_2EAB0A0(v50);
        v54 = (unsigned int)v135;
        v55 = (unsigned int)v135 + 1LL;
        if ( v55 > HIDWORD(v135) )
        {
          v118 = v53;
          sub_C8D5F0((__int64)&v134, v136, v55, 4u, v47, (__int64)v45);
          v54 = (unsigned int)v135;
          v53 = v118;
        }
        v134[v54] = v53;
        LODWORD(v135) = v135 + 1;
        goto LABEL_57;
      }
    }
    else if ( (*(_BYTE *)(v50 + 3) & 0x40) == 0 )
    {
      goto LABEL_57;
    }
    v100 = (unsigned int)v132;
    v101 = (unsigned int)v132 + 1LL;
    if ( v101 > HIDWORD(v132) )
    {
      sub_C8D5F0((__int64)&v131, v133, v101, 4u, v47, (__int64)v45);
      v100 = (unsigned int)v132;
    }
    v131[v100] = v51;
    LODWORD(v132) = v132 + 1;
LABEL_57:
    v56 = *(_BYTE *)(v50 + 4);
    if ( (v56 & 1) == 0 && (v56 & 2) == 0 && ((*(_BYTE *)(v50 + 3) & 0x10) == 0 || (*(_DWORD *)v50 & 0xFFF00) != 0) )
    {
      v57 = *(_QWORD *)(v48 + 8);
      v58 = *(_DWORD *)(v57 + 24LL * v51 + 16) & 0xFFF;
      v45 = (unsigned int **)(*(_QWORD *)(v48 + 56) + 2LL * (*(_DWORD *)(v57 + 24LL * v51 + 16) >> 12));
      while ( v45 )
      {
        v59 = *(unsigned int *)(a4 + 8);
        v60 = *(unsigned __int8 *)(*(_QWORD *)(a4 + 208) + v58);
        if ( v60 < (unsigned int)v59 )
        {
          v61 = *(_QWORD *)a4;
          while ( 1 )
          {
            v62 = (_DWORD *)(v61 + 24LL * v60);
            if ( v58 == *v62 )
              break;
            v60 += 256;
            if ( (unsigned int)v59 <= v60 )
              goto LABEL_134;
          }
          if ( v62 != (_DWORD *)(v61 + 24 * v59) )
          {
            v115 = v61 + 24LL * v60;
            v63 = sub_2EAB0A0(v50);
            v45 = *(unsigned int ***)(v115 + 8);
            v64 = (v63 << 32) | *(unsigned int *)(v115 + 16);
            v65 = (unsigned int)v138;
            v66 = v64;
            v67 = (unsigned int)v138 + 1LL;
            if ( v67 > HIDWORD(v138) )
            {
              v119 = *(unsigned int ***)(v115 + 8);
              sub_C8D5F0((__int64)&v137, v139, v67, 0x10u, v47, (__int64)v45);
              v65 = (unsigned int)v138;
              v45 = v119;
            }
            v68 = &v137[4 * v65];
            *(_QWORD *)v68 = v45;
            *((_QWORD *)v68 + 1) = v66;
            LODWORD(v138) = v138 + 1;
            break;
          }
        }
LABEL_134:
        v111 = *(__int16 *)v45;
        v45 = (unsigned int **)((char *)v45 + 2);
        v58 += v111;
        if ( !(_WORD)v111 )
          break;
      }
    }
LABEL_70:
    v50 += 40;
  }
  while ( v49 != v50 );
  v69 = v131;
  v5 = a1;
  v4 = v122;
  v70 = &v131[(unsigned int)v132];
  if ( v131 != v70 )
  {
    do
    {
      v71 = *(_DWORD *)(*(_QWORD *)(v48 + 8) + 24LL * *v69 + 16) & 0xFFF;
      v72 = (__int16 *)(*(_QWORD *)(v48 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v48 + 8) + 24LL * *v69 + 16) >> 12));
      do
      {
        if ( !v72 )
          break;
        v73 = *(_DWORD *)(a4 + 8);
        v74 = *(unsigned __int8 *)(*(_QWORD *)(a4 + 208) + v71);
        if ( v74 < v73 )
        {
          v75 = *(_QWORD *)a4;
          while ( 1 )
          {
            v76 = (__m128i *)(v75 + 24LL * v74);
            if ( v71 == v76->m128i_i32[0] )
              break;
            v74 += 256;
            if ( v73 <= v74 )
              goto LABEL_82;
          }
          v77 = 24LL * v73;
          if ( v76 != (__m128i *)(v75 + v77) )
          {
            v78 = (const __m128i *)(v75 + v77 - 24);
            if ( v76 != v78 )
            {
              *v76 = _mm_loadu_si128(v78);
              v76[1].m128i_i32[0] = v78[1].m128i_i32[0];
              *(_BYTE *)(*(_QWORD *)(a4 + 208) + *(unsigned int *)(*(_QWORD *)a4 + 24LL * *(unsigned int *)(a4 + 8) - 24)) = -85 * (((__int64)v76->m128i_i64 - *(_QWORD *)a4) >> 3);
              v73 = *(_DWORD *)(a4 + 8);
            }
            *(_DWORD *)(a4 + 8) = v73 - 1;
          }
        }
LABEL_82:
        v79 = *v72++;
        v71 += v79;
      }
      while ( (_WORD)v79 );
      ++v69;
    }
    while ( v70 != v69 );
  }
  v80 = &v134[(unsigned int)v135];
  if ( v134 != v80 )
  {
    v81 = v48;
    v82 = a4;
    v83 = v134;
    v84 = (const __m128i *)&v128;
    do
    {
      v85 = *(_QWORD *)(v81 + 8);
      v86 = *v83;
      v87 = *(_DWORD *)(v85 + 24LL * *(unsigned int *)(*(_QWORD *)(v122 + 32) + 40 * v86 + 8) + 16) & 0xFFF;
      v88 = *(_QWORD *)(v81 + 56)
          + 2LL * (*(_DWORD *)(v85 + 24LL * *(unsigned int *)(*(_QWORD *)(v122 + 32) + 40 * v86 + 8) + 16) >> 12);
      do
      {
        if ( !v88 )
          break;
        v89 = *(_DWORD *)(v82 + 8);
        v90 = (_BYTE *)(*(_QWORD *)(v82 + 208) + v87);
        v128 = v87;
        v129 = 0;
        v91 = (unsigned __int8)*v90;
        v130 = 0;
        if ( v91 >= v89 )
          goto LABEL_100;
        v92 = *(_QWORD *)v82;
        while ( 1 )
        {
          v93 = v92 + 24LL * v91;
          if ( v87 == *(_DWORD *)v93 )
            break;
          v91 += 256;
          if ( v89 <= v91 )
            goto LABEL_100;
        }
        if ( v93 == v92 + 24LL * v89 )
        {
LABEL_100:
          *v90 = v89;
          v94 = *(unsigned int *)(v82 + 8);
          v95 = v94 + 1;
          if ( v94 + 1 > (unsigned __int64)*(unsigned int *)(v82 + 12) )
          {
            v102 = (const void *)(v82 + 16);
            if ( *(_QWORD *)v82 > (unsigned __int64)v84
              || (v112 = *(_QWORD *)v82, (unsigned __int64)v84 >= *(_QWORD *)v82 + 24 * v94) )
            {
              v114 = v84;
              v117 = v80;
              v121 = v81;
              sub_C8D5F0(v82, v102, v95, 0x18u, v81, (__int64)v84);
              v84 = v114;
              v80 = v117;
              v81 = v121;
              v96 = *(_QWORD *)v82;
              v94 = *(unsigned int *)(v82 + 8);
              v97 = v114;
            }
            else
            {
              v113 = v84;
              v116 = v80;
              v120 = v81;
              sub_C8D5F0(v82, v102, v95, 0x18u, v81, (__int64)v84);
              v84 = v113;
              v81 = v120;
              v96 = *(_QWORD *)v82;
              v80 = v116;
              v97 = (const __m128i *)((char *)v113 + *(_QWORD *)v82 - v112);
              v94 = *(unsigned int *)(v82 + 8);
            }
          }
          else
          {
            v96 = *(_QWORD *)v82;
            v97 = v84;
          }
          v98 = (__m128i *)(v96 + 24 * v94);
          *v98 = _mm_loadu_si128(v97);
          v98[1].m128i_i64[0] = v97[1].m128i_i64[0];
          v99 = (unsigned int)(*(_DWORD *)(v82 + 8) + 1);
          *(_DWORD *)(v82 + 8) = v99;
          v93 = *(_QWORD *)v82 + 24 * v99 - 24;
        }
        *(_QWORD *)(v93 + 8) = v122;
        v88 += 2;
        *(_DWORD *)(v93 + 16) = v86;
        v87 += *(__int16 *)(v88 - 2);
      }
      while ( *(_WORD *)(v88 - 2) );
      ++v83;
    }
    while ( v80 != v83 );
    v5 = a1;
    v80 = v134;
  }
  if ( v80 != (unsigned int *)v136 )
    _libc_free((unsigned __int64)v80);
LABEL_98:
  if ( v131 != (unsigned int *)v133 )
    _libc_free((unsigned __int64)v131);
LABEL_5:
  v9 = v137;
  v10 = 0;
  v11 = *(_QWORD *)(v5 + 384);
  v12 = *(_DWORD *)(v5 + 400);
  if ( &v137[4 * (unsigned int)v138] == v137 )
    goto LABEL_14;
  v13 = 0x800000000000C09LL;
  v14 = v4;
  v15 = &v137[4 * (unsigned int)v138];
  while ( 2 )
  {
    v16 = *(_QWORD *)v9;
    v17 = *(_QWORD *)(v5 + 8) + 88LL * *(int *)(*(_QWORD *)(*(_QWORD *)v9 + 24LL) + 24LL);
    v18 = *(_DWORD *)(v17 + 24);
    if ( v18 != -1 )
    {
      v19 = *(_DWORD *)(a2 + 24);
      if ( v19 != -1 && *(_DWORD *)(v17 + 16) == *(_DWORD *)(a2 + 16) && *(_BYTE *)(v17 + 32) && v18 <= v19 )
      {
        if ( v12 )
        {
          v28 = (v12 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v29 = (__int64 *)(v11 + 16LL * v28);
          v30 = *v29;
          if ( v16 == *v29 )
          {
LABEL_25:
            v31 = *((_DWORD *)v29 + 2);
LABEL_26:
            v32 = *(unsigned __int16 *)(v16 + 68);
            if ( (_WORD)v32 )
            {
              v33 = (unsigned int)(v32 - 9);
              if ( ((unsigned __int16)v33 > 0x3Bu || !_bittest64(&v13, v33))
                && (*(_BYTE *)(*(_QWORD *)(v16 + 16) + 24LL) & 0x10) == 0 )
              {
                v123 = v15;
                v124 = v14;
                v34 = sub_2FF8170(*(_QWORD *)(v5 + 440) + 40LL, v16, v9[2], v14, v9[3]);
                v11 = *(_QWORD *)(v5 + 384);
                v12 = *(_DWORD *)(v5 + 400);
                v13 = 0x800000000000C09LL;
                v15 = v123;
                v14 = v124;
                v31 += v34;
              }
            }
            if ( v10 < v31 )
              v10 = v31;
            goto LABEL_12;
          }
          v35 = 1;
          while ( v30 != -4096 )
          {
            v28 = (v12 - 1) & (v35 + v28);
            v126 = v35 + 1;
            v29 = (__int64 *)(v11 + 16LL * v28);
            v30 = *v29;
            if ( v16 == *v29 )
              goto LABEL_25;
            v35 = v126;
          }
        }
        v31 = 0;
        goto LABEL_26;
      }
    }
LABEL_12:
    v9 += 4;
    if ( v15 != v9 )
      continue;
    break;
  }
  v4 = v14;
LABEL_14:
  v20 = v5 + 376;
  if ( !v12 )
    goto LABEL_39;
LABEL_15:
  v21 = 1;
  v22 = 0;
  v23 = (v12 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v24 = (__int64 *)(v11 + 16LL * v23);
  v25 = *v24;
  if ( v4 == *v24 )
    goto LABEL_16;
  while ( v25 != -4096 )
  {
    if ( v25 == -8192 && !v22 )
      v22 = v24;
    v23 = (v12 - 1) & (v21 + v23);
    v24 = (__int64 *)(v11 + 16LL * v23);
    v25 = *v24;
    if ( v4 == *v24 )
      goto LABEL_16;
    ++v21;
  }
  v103 = *(_DWORD *)(v5 + 392);
  if ( v22 )
    v24 = v22;
  ++*(_QWORD *)(v5 + 376);
  v40 = v103 + 1;
  if ( 4 * (v103 + 1) >= 3 * v12 )
  {
LABEL_40:
    sub_2EEB780(v20, 2 * v12);
    v36 = *(_DWORD *)(v5 + 400);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(v5 + 384);
      v39 = v37 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v40 = *(_DWORD *)(v5 + 392) + 1;
      v24 = (__int64 *)(v38 + 16LL * v39);
      v41 = *v24;
      if ( v4 != *v24 )
      {
        v42 = 1;
        v43 = 0;
        while ( v41 != -4096 )
        {
          if ( v41 == -8192 && !v43 )
            v43 = v24;
          v39 = v37 & (v42 + v39);
          v24 = (__int64 *)(v38 + 16LL * v39);
          v41 = *v24;
          if ( v4 == *v24 )
            goto LABEL_121;
          ++v42;
        }
        if ( v43 )
          v24 = v43;
      }
      goto LABEL_121;
    }
    goto LABEL_146;
  }
  if ( v12 - (v40 + *(_DWORD *)(v5 + 396)) <= v12 >> 3 )
  {
    sub_2EEB780(v20, v12);
    v104 = *(_DWORD *)(v5 + 400);
    if ( v104 )
    {
      v105 = v104 - 1;
      v106 = *(_QWORD *)(v5 + 384);
      v107 = 0;
      v108 = v105 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v40 = *(_DWORD *)(v5 + 392) + 1;
      v109 = 1;
      v24 = (__int64 *)(v106 + 16LL * v108);
      v110 = *v24;
      if ( v4 != *v24 )
      {
        while ( v110 != -4096 )
        {
          if ( !v107 && v110 == -8192 )
            v107 = v24;
          v108 = v105 & (v109 + v108);
          v24 = (__int64 *)(v106 + 16LL * v108);
          v110 = *v24;
          if ( v4 == *v24 )
            goto LABEL_121;
          ++v109;
        }
        if ( v107 )
          v24 = v107;
      }
      goto LABEL_121;
    }
LABEL_146:
    ++*(_DWORD *)(v5 + 392);
    BUG();
  }
LABEL_121:
  *(_DWORD *)(v5 + 392) = v40;
  if ( *v24 != -4096 )
    --*(_DWORD *)(v5 + 396);
  *v24 = v4;
  v24[1] = 0;
LABEL_16:
  v26 = (unsigned int *)(v24 + 1);
  *v26 = v10;
  if ( *(_BYTE *)(a2 + 33) )
  {
    v27 = v26[1] + v10;
    if ( v27 < *(_DWORD *)(a2 + 36) )
      v27 = *(_DWORD *)(a2 + 36);
    *(_DWORD *)(a2 + 36) = v27;
  }
  if ( v137 != (unsigned int *)v139 )
    _libc_free((unsigned __int64)v137);
}
