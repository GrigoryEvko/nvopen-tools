// Function: sub_27A8390
// Address: 0x27a8390
//
void __fastcall sub_27A8390(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  int v7; // edi
  __int64 v8; // r13
  __int64 *v9; // rax
  int v10; // r8d
  int v11; // r8d
  __int64 v12; // rdx
  __int64 v13; // r9
  unsigned int v14; // eax
  unsigned __int64 v15; // rdi
  int v16; // eax
  __int64 *v17; // r13
  __int64 v18; // rax
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  _QWORD *v23; // rbx
  _QWORD *v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v26; // rcx
  unsigned __int64 v27; // r13
  _BYTE *v28; // rbx
  __int64 v29; // r14
  __int64 *v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 *v34; // rax
  __int64 v35; // rax
  int v36; // r12d
  const __m128i *v37; // r10
  __int64 v38; // r9
  unsigned int v39; // edi
  _QWORD *v40; // rbx
  __int64 v41; // rcx
  __int64 v42; // rax
  unsigned __int64 v43; // r14
  _QWORD *v44; // r13
  unsigned __int64 v45; // rsi
  unsigned __int64 v46; // r8
  __m128i *v47; // rcx
  __m128i *v48; // rax
  __int64 v49; // r15
  __int64 v50; // rdx
  __int64 v51; // r13
  __int64 v52; // rcx
  int v53; // edx
  _QWORD *v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rax
  __int64 v57; // r14
  int v58; // ebx
  __int64 v59; // r13
  __int64 v60; // rax
  __int64 v61; // r8
  char v62; // al
  int v63; // r10d
  unsigned int v64; // ecx
  _QWORD *v65; // r15
  __int64 *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rdx
  unsigned __int64 v69; // r9
  __m128i **v70; // rax
  __m128i *v71; // rcx
  unsigned __int64 v72; // r10
  __m128i *v73; // rdx
  int v74; // edx
  const void *v75; // rsi
  __int64 v76; // rsi
  __int64 v77; // rdi
  __int64 v78; // rsi
  int v79; // r11d
  __int64 *v80; // r10
  int v81; // r11d
  __int64 v82; // rdi
  __int64 v83; // rsi
  const void *v84; // rsi
  __int64 v85; // rdx
  _QWORD *v86; // r9
  __int64 v87; // r14
  int v88; // ebx
  __int64 v89; // rsi
  int v90; // eax
  const __m128i *v91; // rax
  const __m128i *v92; // r13
  const __m128i *v93; // rbx
  __m128i *v94; // rax
  __m128i *v95; // rsi
  __int64 v96; // rbx
  unsigned __int64 v97; // rax
  __m128i *v98; // rbx
  const __m128i *v99; // rdi
  int v100; // ebx
  _QWORD *v101; // r14
  __m128i *v105; // [rsp+30h] [rbp-1B0h]
  __int64 v106; // [rsp+40h] [rbp-1A0h]
  const __m128i *v107; // [rsp+50h] [rbp-190h]
  __int64 *v108; // [rsp+50h] [rbp-190h]
  const __m128i *v109; // [rsp+50h] [rbp-190h]
  int v110; // [rsp+50h] [rbp-190h]
  const __m128i *v111; // [rsp+50h] [rbp-190h]
  const __m128i *v112; // [rsp+58h] [rbp-188h]
  __int64 *v113; // [rsp+68h] [rbp-178h]
  __int64 v114; // [rsp+70h] [rbp-170h]
  unsigned int v115; // [rsp+70h] [rbp-170h]
  __m128i *v116; // [rsp+80h] [rbp-160h] BYREF
  __m128i *v117; // [rsp+88h] [rbp-158h]
  __m128i *v118; // [rsp+90h] [rbp-150h]
  __int64 *v119; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v120; // [rsp+A8h] [rbp-138h]
  _BYTE v121[16]; // [rsp+B0h] [rbp-130h] BYREF
  __int64 v122; // [rsp+C0h] [rbp-120h] BYREF
  _QWORD *v123; // [rsp+C8h] [rbp-118h]
  __int64 v124; // [rsp+D0h] [rbp-110h]
  unsigned int v125; // [rsp+D8h] [rbp-108h]
  __int64 v126; // [rsp+E0h] [rbp-100h] BYREF
  _QWORD *v127; // [rsp+E8h] [rbp-F8h]
  __int64 v128; // [rsp+F0h] [rbp-F0h]
  unsigned int v129; // [rsp+F8h] [rbp-E8h]
  __m128i v130; // [rsp+100h] [rbp-E0h] BYREF
  __int128 v131; // [rsp+110h] [rbp-D0h]
  __int64 *v132[2]; // [rsp+120h] [rbp-C0h] BYREF
  char v133; // [rsp+130h] [rbp-B0h]
  __int64 *v134; // [rsp+140h] [rbp-A0h]
  _BYTE *v135; // [rsp+150h] [rbp-90h] BYREF
  __int64 v136; // [rsp+158h] [rbp-88h]
  _BYTE v137[32]; // [rsp+160h] [rbp-80h] BYREF
  __int64 v138; // [rsp+180h] [rbp-60h] BYREF
  __int64 *v139; // [rsp+188h] [rbp-58h]
  __int64 v140; // [rsp+190h] [rbp-50h]
  int v141; // [rsp+198h] [rbp-48h]
  char v142; // [rsp+19Ch] [rbp-44h]
  char v143; // [rsp+1A0h] [rbp-40h] BYREF

  v7 = *(_DWORD *)(a2 + 16);
  v116 = 0;
  v117 = 0;
  v118 = 0;
  if ( !v7 )
    goto LABEL_2;
  v91 = *(const __m128i **)(a2 + 8);
  v92 = &v91[4 * (unsigned __int64)*(unsigned int *)(a2 + 24)];
  if ( v91 == v92 )
    goto LABEL_2;
  while ( 1 )
  {
    v93 = v91;
    if ( v91->m128i_i32[0] != -1 )
      break;
    if ( v91->m128i_i64[1] != -1 )
      goto LABEL_126;
LABEL_155:
    v91 += 4;
    if ( v92 == v91 )
      goto LABEL_2;
  }
  if ( v91->m128i_i32[0] == -2 && v91->m128i_i64[1] == -2 )
    goto LABEL_155;
LABEL_126:
  if ( v92 != v91 )
  {
    v94 = 0;
    v95 = 0;
LABEL_131:
    if ( v94 == v95 )
    {
      sub_27A37F0((unsigned __int64 *)&v116, v95, v93);
      v95 = v117;
    }
    else
    {
      if ( v95 )
      {
        *v95 = _mm_loadu_si128(v93);
        v95 = v117;
      }
      v117 = ++v95;
    }
    v93 += 4;
    if ( v93 == v92 )
    {
LABEL_139:
      v8 = (__int64)v116;
      if ( v116 == v95 )
      {
        v105 = v95;
      }
      else
      {
        v96 = (char *)v95 - (char *)v116;
        _BitScanReverse64(&v97, v95 - v116);
        sub_27A80C0((__int64)v116, (unsigned __int64)v95, 2LL * (int)(63 - (v97 ^ 0x3F)), a1, a2, a6);
        if ( v96 <= 256 )
        {
          sub_27A7E00(v8, v95->m128i_i32, a1, a2);
        }
        else
        {
          v98 = (__m128i *)(v8 + 256);
          sub_27A7E00(v8, (int *)(v8 + 256), a1, a2);
          if ( (__m128i *)(v8 + 256) != v95 )
          {
            do
            {
              v99 = v98++;
              sub_27A7D80(v99, a1, a2);
            }
            while ( v95 != v98 );
          }
        }
        v8 = (__int64)v116;
        v105 = v117;
      }
      goto LABEL_3;
    }
    while ( 1 )
    {
      if ( v93->m128i_i32[0] == -1 )
      {
        if ( v93->m128i_i64[1] != -1 )
          goto LABEL_129;
      }
      else if ( v93->m128i_i32[0] != -2 || v93->m128i_i64[1] != -2 )
      {
LABEL_129:
        if ( v92 == v93 )
          goto LABEL_139;
        v94 = v118;
        goto LABEL_131;
      }
      v93 += 4;
      if ( v92 == v93 )
        goto LABEL_139;
    }
  }
LABEL_2:
  v105 = 0;
  v8 = 0;
LABEL_3:
  v132[1] = 0;
  v119 = (__int64 *)v121;
  v120 = 0x200000000LL;
  v9 = *(__int64 **)(a1 + 224);
  v133 = 0;
  v132[0] = v9;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v112 = (const __m128i *)v8;
  if ( v105 == (__m128i *)v8 )
    goto LABEL_16;
  while ( 2 )
  {
    v10 = *(_DWORD *)(a2 + 24);
    if ( !v10 )
      goto LABEL_15;
    v11 = v10 - 1;
    v12 = v112->m128i_u32[0];
    v13 = 1;
    v14 = v11
        & (((0xBF58476D1CE4E5B9LL
           * ((unsigned int)((0xBF58476D1CE4E5B9LL * v112->m128i_i64[1]) >> 31) ^ (484763065 * v112->m128i_i32[2])
            | ((unsigned __int64)(unsigned int)(37 * v12) << 32))) >> 31)
         ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v112->m128i_i64[1]) >> 31) ^ (484763065 * v112->m128i_i32[2]))));
    while ( 1 )
    {
      v15 = *(_QWORD *)(a2 + 8) + ((unsigned __int64)v14 << 6);
      if ( *(_DWORD *)v15 == (_DWORD)v12 && *(_QWORD *)(v15 + 8) == v112->m128i_i64[1] )
        break;
      if ( *(_DWORD *)v15 == -1 )
      {
        if ( *(_QWORD *)(v15 + 8) == -1 )
          goto LABEL_15;
        v90 = v13 + v14;
        v13 = (unsigned int)(v13 + 1);
        v14 = v11 & v90;
      }
      else
      {
        v16 = v13 + v14;
        v13 = (unsigned int)(v13 + 1);
        v14 = v11 & v16;
      }
    }
    v26 = *(unsigned int *)(v15 + 24);
    v135 = v137;
    v136 = 0x400000000LL;
    if ( !(_DWORD)v26 )
      goto LABEL_15;
    sub_27A1010((__int64)&v135, v15 + 16, v12, v26, (__int64)&v135, v13);
    if ( (unsigned int)v136 <= 1uLL )
      goto LABEL_13;
    v27 = (unsigned __int64)v135;
    v138 = 0;
    v139 = (__int64 *)&v143;
    v140 = 2;
    v28 = &v135[8 * (unsigned int)v136];
    v141 = 0;
    v142 = 1;
    while ( 2 )
    {
      while ( 2 )
      {
        v29 = *(_QWORD *)(*(_QWORD *)v27 + 40LL);
        if ( (unsigned __int8)sub_27A53C0(a1, v29) )
        {
LABEL_41:
          v27 += 8LL;
          if ( v28 == (_BYTE *)v27 )
            goto LABEL_49;
          continue;
        }
        break;
      }
      if ( !v142 )
        goto LABEL_61;
      v34 = v139;
      v31 = HIDWORD(v140);
      v30 = &v139[HIDWORD(v140)];
      if ( v139 != v30 )
      {
        while ( v29 != *v34 )
        {
          if ( v30 == ++v34 )
            goto LABEL_47;
        }
        goto LABEL_41;
      }
LABEL_47:
      if ( HIDWORD(v140) >= (unsigned int)v140 )
      {
LABEL_61:
        sub_C8CC70((__int64)&v138, v29, (__int64)v30, v31, v32, v33);
        goto LABEL_41;
      }
      v31 = (unsigned int)(HIDWORD(v140) + 1);
      v27 += 8LL;
      ++HIDWORD(v140);
      *v30 = v29;
      ++v138;
      if ( v28 != (_BYTE *)v27 )
        continue;
      break;
    }
LABEL_49:
    v134 = &v138;
    LODWORD(v120) = 0;
    sub_271EE20(v132, (__int64)&v119, (__int64)v30, v31, v32, v33);
    v35 = 0;
    if ( !(_DWORD)v136 )
      goto LABEL_63;
    v36 = 0;
    v114 = a1;
    v37 = v112;
    while ( 2 )
    {
      v49 = 8 * v35;
      v50 = *(_QWORD *)&v135[8 * v35];
      v51 = *(_QWORD *)(v50 + 40);
      if ( !v129 )
      {
        ++v126;
        goto LABEL_56;
      }
      v38 = v129 - 1;
      v39 = v38 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v40 = &v127[9 * v39];
      v41 = *v40;
      if ( v51 == *v40 )
      {
LABEL_52:
        v42 = *((unsigned int *)v40 + 4);
        v43 = v40[1];
        v44 = v40 + 1;
        *(_QWORD *)&v131 = v50;
        v45 = *((unsigned int *)v40 + 5);
        v46 = v42 + 1;
        v47 = (__m128i *)(v43 + 24 * v42);
        v130 = _mm_loadu_si128(v37);
        v48 = &v130;
        if ( v45 < v46 )
        {
          v109 = v37;
          v84 = v40 + 3;
          if ( v43 > (unsigned __int64)&v130 || v47 <= &v130 )
          {
            sub_C8D5F0((__int64)(v40 + 1), v84, v46, 0x18u, v46, v38);
            v37 = v109;
            v47 = (__m128i *)(v40[1] + 24LL * *((unsigned int *)v40 + 4));
            v48 = &v130;
          }
          else
          {
            sub_C8D5F0((__int64)(v40 + 1), v84, v46, 0x18u, v46, v38);
            v85 = v40[1];
            v37 = v109;
            v48 = (__m128i *)((char *)&v130 + v85 - v43);
            v47 = (__m128i *)(v85 + 24LL * *((unsigned int *)v40 + 4));
          }
        }
      }
      else
      {
        v110 = 1;
        v54 = 0;
        while ( v41 != -4096 )
        {
          if ( v41 == -8192 && !v54 )
            v54 = v40;
          v39 = v38 & (v110 + v39);
          v40 = &v127[9 * v39];
          v41 = *v40;
          if ( v51 == *v40 )
            goto LABEL_52;
          ++v110;
        }
        if ( !v54 )
          v54 = v40;
        ++v126;
        v53 = v128 + 1;
        if ( 4 * ((int)v128 + 1) >= 3 * v129 )
        {
LABEL_56:
          v107 = v37;
          sub_27A41B0((__int64)&v126, 2 * v129);
          if ( !v129 )
            goto LABEL_180;
          v37 = v107;
          LODWORD(v52) = (v129 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
          v53 = v128 + 1;
          v54 = &v127[9 * (unsigned int)v52];
          v55 = *v54;
          if ( v51 != *v54 )
          {
            v100 = 1;
            v101 = 0;
            while ( v55 != -4096 )
            {
              if ( !v101 && v55 == -8192 )
                v101 = v54;
              v52 = (v129 - 1) & ((_DWORD)v52 + v100);
              v54 = &v127[9 * v52];
              v55 = *v54;
              if ( v51 == *v54 )
                goto LABEL_58;
              ++v100;
            }
            if ( v101 )
              v54 = v101;
          }
        }
        else if ( v129 - HIDWORD(v128) - v53 <= v129 >> 3 )
        {
          v111 = v37;
          sub_27A41B0((__int64)&v126, v129);
          if ( !v129 )
          {
LABEL_180:
            LODWORD(v128) = v128 + 1;
            BUG();
          }
          v86 = 0;
          LODWORD(v87) = (v129 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
          v37 = v111;
          v53 = v128 + 1;
          v88 = 1;
          v54 = &v127[9 * (unsigned int)v87];
          v89 = *v54;
          if ( v51 != *v54 )
          {
            while ( v89 != -4096 )
            {
              if ( !v86 && v89 == -8192 )
                v86 = v54;
              v87 = (v129 - 1) & ((_DWORD)v87 + v88);
              v54 = &v127[9 * v87];
              v89 = *v54;
              if ( v51 == *v54 )
                goto LABEL_58;
              ++v88;
            }
            if ( v86 )
              v54 = v86;
          }
        }
LABEL_58:
        LODWORD(v128) = v53;
        if ( *v54 != -4096 )
          --HIDWORD(v128);
        v47 = (__m128i *)(v54 + 3);
        *v54 = v51;
        v44 = v54 + 1;
        v54[1] = v54 + 3;
        v54[2] = 0x200000000LL;
        v56 = *(_QWORD *)&v135[v49];
        v130 = _mm_loadu_si128(v37);
        *(_QWORD *)&v131 = v56;
        v48 = &v130;
      }
      *v47 = _mm_loadu_si128(v48);
      v47[1].m128i_i64[0] = v48[1].m128i_i64[0];
      v35 = (unsigned int)(v36 + 1);
      ++*((_DWORD *)v44 + 2);
      v36 = v35;
      if ( (unsigned int)v35 < (unsigned int)v136 )
        continue;
      break;
    }
    a1 = v114;
LABEL_63:
    v131 = 0;
    v17 = v119;
    v108 = &v119[(unsigned int)v120];
    v130 = _mm_loadu_si128(v112);
    if ( v108 != v119 )
    {
      while ( 1 )
      {
        v57 = *v17;
        if ( (_DWORD)v136 )
          break;
LABEL_10:
        if ( v108 == ++v17 )
          goto LABEL_11;
      }
      v58 = 0;
      v113 = v17;
      v59 = a1;
      v60 = 0;
      v115 = ((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4);
      while ( 2 )
      {
        sub_B196A0(*(_QWORD *)(v59 + 216), v57, *(_QWORD *)(*(_QWORD *)&v135[8 * v60] + 40LL));
        if ( v62 )
        {
          if ( v125 )
          {
            v63 = 1;
            v64 = (v125 - 1) & v115;
            v65 = &v123[11 * v64];
            v66 = 0;
            v67 = *v65;
            if ( v57 == *v65 )
            {
LABEL_70:
              v68 = *((unsigned int *)v65 + 4);
              v69 = v65[1];
              v70 = (__m128i **)(v65 + 1);
              v71 = &v130;
              v72 = v68 + 1;
              v73 = (__m128i *)(v69 + 32 * v68);
              if ( v72 > *((unsigned int *)v65 + 5) )
              {
                v106 = v65[1];
                v75 = v65 + 3;
                if ( v69 > (unsigned __int64)&v130 || v73 <= &v130 )
                {
                  sub_C8D5F0((__int64)(v65 + 1), v75, v72, 0x20u, v61, v69);
                  v70 = (__m128i **)(v65 + 1);
                  v71 = &v130;
                  v73 = (__m128i *)(v65[1] + 32LL * *((unsigned int *)v65 + 4));
                }
                else
                {
                  sub_C8D5F0((__int64)(v65 + 1), v75, v72, 0x20u, v61, v69);
                  v76 = v65[1];
                  v70 = (__m128i **)(v65 + 1);
                  v73 = (__m128i *)(v76 + 32LL * *((unsigned int *)v65 + 4));
                  v71 = (__m128i *)((char *)&v130 + v76 - v106);
                }
              }
              goto LABEL_71;
            }
            while ( v67 != -4096 )
            {
              if ( v67 == -8192 && !v66 )
                v66 = v65;
              v61 = (unsigned int)(v63 + 1);
              v64 = (v125 - 1) & (v63 + v64);
              v65 = &v123[11 * v64];
              v67 = *v65;
              if ( v57 == *v65 )
                goto LABEL_70;
              ++v63;
            }
            if ( !v66 )
              v66 = v65;
            ++v122;
            v74 = v124 + 1;
            if ( 4 * ((int)v124 + 1) < 3 * v125 )
            {
              if ( v125 - HIDWORD(v124) - v74 <= v125 >> 3 )
              {
                sub_27A44C0((__int64)&v122, v125);
                if ( !v125 )
                {
LABEL_181:
                  LODWORD(v124) = v124 + 1;
                  BUG();
                }
                v80 = 0;
                v81 = 1;
                LODWORD(v82) = (v125 - 1) & v115;
                v66 = &v123[11 * (unsigned int)v82];
                v83 = *v66;
                v74 = v124 + 1;
                if ( v57 != *v66 )
                {
                  while ( v83 != -4096 )
                  {
                    if ( !v80 && v83 == -8192 )
                      v80 = v66;
                    v82 = (v125 - 1) & ((_DWORD)v82 + v81);
                    v66 = &v123[11 * v82];
                    v83 = *v66;
                    if ( v57 == *v66 )
                      goto LABEL_82;
                    ++v81;
                  }
                  goto LABEL_102;
                }
              }
              goto LABEL_82;
            }
          }
          else
          {
            ++v122;
          }
          sub_27A44C0((__int64)&v122, 2 * v125);
          if ( !v125 )
            goto LABEL_181;
          LODWORD(v77) = (v125 - 1) & v115;
          v66 = &v123[11 * (unsigned int)v77];
          v78 = *v66;
          v74 = v124 + 1;
          if ( v57 != *v66 )
          {
            v79 = 1;
            v80 = 0;
            while ( v78 != -4096 )
            {
              if ( !v80 && v78 == -8192 )
                v80 = v66;
              v77 = (v125 - 1) & ((_DWORD)v77 + v79);
              v66 = &v123[11 * v77];
              v78 = *v66;
              if ( v57 == *v66 )
                goto LABEL_82;
              ++v79;
            }
LABEL_102:
            if ( v80 )
              v66 = v80;
          }
LABEL_82:
          LODWORD(v124) = v74;
          if ( *v66 != -4096 )
            --HIDWORD(v124);
          v73 = (__m128i *)(v66 + 3);
          *v66 = v57;
          v71 = &v130;
          v70 = (__m128i **)(v66 + 1);
          *v70 = v73;
          v70[1] = (__m128i *)0x200000000LL;
LABEL_71:
          *v73 = _mm_loadu_si128(v71);
          v73[1] = _mm_loadu_si128(v71 + 1);
          ++*((_DWORD *)v70 + 2);
        }
        v60 = (unsigned int)(v58 + 1);
        v58 = v60;
        if ( (unsigned int)v60 >= (unsigned int)v136 )
        {
          a1 = v59;
          v17 = v113;
          goto LABEL_10;
        }
        continue;
      }
    }
LABEL_11:
    if ( !v142 )
      _libc_free((unsigned __int64)v139);
LABEL_13:
    if ( v135 != v137 )
      _libc_free((unsigned __int64)v135);
LABEL_15:
    if ( v105 != ++v112 )
      continue;
    break;
  }
LABEL_16:
  sub_27A7450(a1, (__int64)&v126, (__int64)&v122);
  sub_27A6600(a1, (__int64)&v122, a4, a3);
  v18 = v129;
  if ( v129 )
  {
    v19 = v127;
    v20 = &v127[9 * v129];
    do
    {
      if ( *v19 != -4096 && *v19 != -8192 )
      {
        v21 = v19[1];
        if ( (_QWORD *)v21 != v19 + 3 )
          _libc_free(v21);
      }
      v19 += 9;
    }
    while ( v20 != v19 );
    v18 = v129;
  }
  sub_C7D6A0((__int64)v127, 72 * v18, 8);
  v22 = v125;
  if ( v125 )
  {
    v23 = v123;
    v24 = &v123[11 * v125];
    do
    {
      if ( *v23 != -8192 && *v23 != -4096 )
      {
        v25 = v23[1];
        if ( (_QWORD *)v25 != v23 + 3 )
          _libc_free(v25);
      }
      v23 += 11;
    }
    while ( v24 != v23 );
    v22 = v125;
  }
  sub_C7D6A0((__int64)v123, 88 * v22, 8);
  if ( v119 != (__int64 *)v121 )
    _libc_free((unsigned __int64)v119);
  if ( v116 )
    j_j___libc_free_0((unsigned __int64)v116);
}
