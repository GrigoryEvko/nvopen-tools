// Function: sub_37DF4F0
// Address: 0x37df4f0
//
__int64 __fastcall sub_37DF4F0(__int64 a1, unsigned __int64 a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v4; // r15
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int8 v8; // dl
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // r8
  __int64 v13; // r9
  const __m128i *v14; // r10
  const __m128i *v15; // rbx
  const __m128i *i; // r13
  __int64 v17; // rax
  __m128i v18; // xmm5
  unsigned int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // r11
  const __m128i *v27; // rbx
  __int64 v28; // rax
  const __m128i *v29; // r15
  __m128i *v30; // r8
  int v31; // ecx
  unsigned __int64 v32; // r9
  unsigned __int64 v33; // rax
  unsigned int v34; // eax
  __m128i *v35; // rdx
  int v36; // r12d
  _QWORD *v37; // r12
  __int64 v38; // rax
  __int64 v39; // r9
  __int64 v40; // rbx
  __int64 v41; // r13
  __int64 v42; // rdx
  __int64 v43; // r12
  __int64 v44; // r14
  __int64 v45; // r8
  __int64 v46; // rbx
  __int64 v47; // r13
  __int64 v48; // rax
  unsigned __int64 v49; // rcx
  _QWORD *v50; // rax
  __int64 v52; // rdi
  unsigned int *v53; // r12
  unsigned int *v54; // r14
  __m128i *v55; // rdi
  unsigned __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rdx
  const __m128i *v59; // rdx
  __int64 v60; // rdx
  unsigned __int64 v61; // rdx
  __m128i *v62; // rbx
  __m128i *v63; // rdx
  unsigned int v64; // edx
  __int64 v65; // rdx
  char *v66; // rbx
  __int64 v67; // rdx
  const __m128i *v68; // r12
  const __m128i *v69; // r13
  __int64 v70; // rdx
  __m128i *v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // r12
  __int64 v77; // rdx
  _QWORD *v78; // rsi
  unsigned __int64 v79; // rax
  _QWORD *v80; // rdi
  int v81; // esi
  unsigned __int64 v82; // rdx
  unsigned __int64 v83; // rdx
  __int64 v84; // rbx
  __int64 v85; // r10
  unsigned __int8 v86; // r14
  __int64 v87; // rax
  unsigned int v88; // esi
  char *v89; // rax
  char *v90; // rsi
  __int64 v91; // rsi
  __int32 v92; // eax
  __m128i *v93; // r12
  __int64 v94; // rax
  __m128i *v95; // rax
  __int64 v96; // rcx
  __int64 v97; // rax
  __int64 v98; // rsi
  int v99; // ebx
  __int64 v100; // rcx
  __int8 *v101; // r12
  int v102; // edx
  char v103; // r9
  __int64 v104; // rsi
  __m128i *v105; // rcx
  __m128i *v106; // rax
  __int64 v107; // rdi
  __int64 v108; // r8
  __m128i *v109; // rdx
  __int64 v110; // rsi
  unsigned __int64 v111; // r8
  __int64 v112; // rax
  unsigned int v113; // eax
  __int64 v114; // r13
  int v115; // r11d
  __m128i *v116; // rax
  __int64 v117; // rsi
  __int64 v118; // [rsp+8h] [rbp-278h]
  __int64 v119; // [rsp+20h] [rbp-260h]
  __int64 v120; // [rsp+28h] [rbp-258h]
  __int64 v121; // [rsp+28h] [rbp-258h]
  __int64 v122; // [rsp+28h] [rbp-258h]
  int v123; // [rsp+28h] [rbp-258h]
  unsigned __int64 v124; // [rsp+28h] [rbp-258h]
  unsigned int v125; // [rsp+30h] [rbp-250h]
  unsigned __int64 v126; // [rsp+30h] [rbp-250h]
  __int64 v127; // [rsp+38h] [rbp-248h]
  int v128; // [rsp+38h] [rbp-248h]
  __int64 v129; // [rsp+38h] [rbp-248h]
  unsigned int v132; // [rsp+48h] [rbp-238h]
  unsigned int v133; // [rsp+48h] [rbp-238h]
  __int32 v134; // [rsp+54h] [rbp-22Ch] BYREF
  __int64 v135; // [rsp+58h] [rbp-228h] BYREF
  __m128i v136; // [rsp+60h] [rbp-220h] BYREF
  __m128i v137; // [rsp+70h] [rbp-210h] BYREF
  unsigned __int16 v138; // [rsp+80h] [rbp-200h]
  char v139; // [rsp+88h] [rbp-1F8h]
  __int64 v140; // [rsp+90h] [rbp-1F0h]
  __m128i v141; // [rsp+A0h] [rbp-1E0h] BYREF
  __m128i v142; // [rsp+B0h] [rbp-1D0h]
  __int64 v143; // [rsp+C0h] [rbp-1C0h]
  __m128i v144; // [rsp+D0h] [rbp-1B0h] BYREF
  __m128i v145; // [rsp+E0h] [rbp-1A0h]
  __int64 v146; // [rsp+F0h] [rbp-190h]
  char v147; // [rsp+F8h] [rbp-188h]
  unsigned int *v148; // [rsp+100h] [rbp-180h] BYREF
  __int64 v149; // [rsp+108h] [rbp-178h]
  _BYTE v150[48]; // [rsp+110h] [rbp-170h] BYREF
  __m128i *v151; // [rsp+140h] [rbp-140h] BYREF
  __int64 v152; // [rsp+148h] [rbp-138h]
  _BYTE v153[48]; // [rsp+150h] [rbp-130h] BYREF
  _BYTE *v154; // [rsp+180h] [rbp-100h] BYREF
  __int64 j; // [rsp+188h] [rbp-F8h]
  _BYTE v156[48]; // [rsp+190h] [rbp-F0h] BYREF
  __m128i v157; // [rsp+1C0h] [rbp-C0h] BYREF
  __m128i v158; // [rsp+1D0h] [rbp-B0h] BYREF
  __m128i v159; // [rsp+1E0h] [rbp-A0h] BYREF
  __m128i v160; // [rsp+200h] [rbp-80h] BYREF
  __m128i v161; // [rsp+210h] [rbp-70h] BYREF
  __m128i v162; // [rsp+220h] [rbp-60h]
  _BYTE v163[48]; // [rsp+250h] [rbp-30h] BYREF

  v4 = a1;
  v6 = sub_2E89170(a2);
  v127 = sub_2E891C0(a2);
  v119 = a2 + 56;
  v7 = sub_B10CD0(a2 + 56);
  v8 = *(_BYTE *)(v7 - 16);
  if ( (v8 & 2) != 0 )
  {
    if ( *(_DWORD *)(v7 - 24) != 2 )
    {
LABEL_3:
      v9 = 0;
      goto LABEL_4;
    }
    v23 = *(_QWORD *)(v7 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(v7 - 16) >> 6) & 0xF) != 2 )
      goto LABEL_3;
    v23 = v7 - 16 - 8LL * ((v8 >> 2) & 0xF);
  }
  v9 = *(_QWORD *)(v23 + 8);
LABEL_4:
  v137.m128i_i64[0] = v6;
  if ( v127 )
    sub_AF47B0((__int64)&v137.m128i_i64[1], *(unsigned __int64 **)(v127 + 16), *(unsigned __int64 **)(v127 + 24));
  else
    v139 = 0;
  v140 = v9;
  v10 = sub_B10CD0(v119);
  if ( sub_35051D0((_QWORD *)(a1 + 128), v10) )
  {
    v148 = (unsigned int *)v150;
    v149 = 0xC00000000LL;
    v14 = *(const __m128i **)(a2 + 32);
    v15 = (const __m128i *)((char *)v14 + 40);
    if ( *(_WORD *)(a2 + 68) != 14 )
    {
      v15 = (const __m128i *)((char *)v14 + 40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF));
      v14 += 5;
    }
    for ( i = v14; v15 != i; LODWORD(v149) = v149 + 1 )
    {
      if ( i->m128i_i8[0] == 20 )
      {
        v21 = sub_37DCC20(a1, i[1].m128i_u32[2], i[1].m128i_u32[3], a2, a3, a4);
        j = v22;
        v154 = (_BYTE *)v21;
        if ( !(_BYTE)v22 )
        {
          LODWORD(v149) = 0;
          break;
        }
        v157.m128i_i64[0] = (__int64)v154;
        if ( (_BYTE *)qword_5051170 == v154 )
        {
          v19 = dword_5051178[0];
        }
        else
        {
          v160.m128i_i64[0] = (__int64)v154;
          v19 = sub_37C5950(a1 + 2168, (__int64)v154);
        }
      }
      else
      {
        v160 = _mm_loadu_si128(i);
        v161 = _mm_loadu_si128(i + 1);
        v17 = i[2].m128i_i64[0];
        v159.m128i_i8[8] = 1;
        v159.m128i_i64[0] = v17;
        v18 = _mm_load_si128(&v159);
        v157 = v160;
        v158 = v161;
        v162 = v18;
        v19 = sub_37C5390(a1 + 2168, v160.m128i_i32);
      }
      v20 = (unsigned int)v149;
      v12 = (unsigned int)v149 + 1LL;
      if ( v12 > HIDWORD(v149) )
      {
        v125 = v19;
        sub_C8D5F0((__int64)&v148, v150, (unsigned int)v149 + 1LL, 4u, v12, v13);
        v20 = (unsigned int)v149;
        v19 = v125;
      }
      v11 = (__int64)v148;
      i = (const __m128i *)((char *)i + 40);
      v148[v20] = v19;
    }
    v52 = *(_QWORD *)(a1 + 424);
    v136.m128i_i16[4] = 256;
    v136.m128i_i64[0] = v127;
    if ( v52 )
      sub_37DE9A0(v52, a2, &v136, (__int64)&v148);
    if ( *(_QWORD *)(v4 + 432) )
    {
      v53 = v148;
      v152 = 0x100000000LL;
      v54 = &v148[(unsigned int)v149];
      v151 = (__m128i *)v153;
      if ( v54 == v148 )
      {
        v68 = (const __m128i *)v153;
        v69 = (const __m128i *)v153;
        goto LABEL_72;
      }
      v11 = 0;
      v55 = (__m128i *)v153;
      v12 = (unsigned __int64)&qword_5051170;
      v56 = 1;
      v57 = v4;
      while ( 1 )
      {
        v64 = *v53;
        if ( *v53 == dword_5051178[0] )
          break;
        v13 = v64 >> 1;
        if ( (v64 & 1) != 0 )
        {
          v58 = 5 * v13;
          v13 = *(_QWORD *)(v57 + 2184);
          v59 = (const __m128i *)(v13 + 8 * v58);
          v157 = _mm_loadu_si128(v59);
          v158 = _mm_loadu_si128(v59 + 1);
          v60 = v59[2].m128i_i64[0];
          v162.m128i_i8[8] = 1;
          v159.m128i_i64[0] = v60;
          v162.m128i_i64[0] = v60;
          v160 = v157;
          v161 = v158;
        }
        else
        {
          v67 = *(_QWORD *)(*(_QWORD *)(v57 + 2168) + 8 * v13);
          v162.m128i_i8[8] = 0;
          v160.m128i_i64[0] = v67;
        }
        v61 = v11 + 1;
        v62 = &v160;
        if ( v11 + 1 > v56 )
          goto LABEL_67;
LABEL_63:
        ++v53;
        v63 = &v55[3 * v11];
        *v63 = _mm_loadu_si128(v62);
        v63[1] = _mm_loadu_si128(v62 + 1);
        v11 = (unsigned int)(v152 + 1);
        v63[2] = _mm_loadu_si128(v62 + 2);
        LODWORD(v152) = v11;
        if ( v54 == v53 )
        {
          v68 = v151;
          v4 = v57;
          v69 = &v151[3 * v11];
LABEL_72:
          v160.m128i_i64[0] = 0;
          v160.m128i_i64[1] = 1;
          v70 = qword_5051170;
          v71 = &v161;
          do
          {
            v71->m128i_i64[0] = v70;
            ++v71;
          }
          while ( v71 != (__m128i *)v163 );
          v154 = v156;
          for ( j = 0x600000000LL; v69 != v68; LODWORD(j) = j + 1 )
          {
            while ( 1 )
            {
              if ( !v68[2].m128i_i8[8] )
              {
                v72 = v68->m128i_i64[0];
                v144.m128i_i32[2] = 0;
                v144.m128i_i64[0] = v72;
                sub_37D2FB0((__int64)&v157, (__int64)&v160, v144.m128i_i64, &v144.m128i_i32[2]);
                if ( v159.m128i_i8[0] )
                  break;
              }
              v68 += 3;
              if ( v69 == v68 )
                goto LABEL_82;
            }
            v73 = (unsigned int)j;
            v11 = HIDWORD(j);
            v12 = v68->m128i_i64[0];
            v74 = (unsigned int)j + 1LL;
            if ( v74 > HIDWORD(j) )
            {
              v124 = v68->m128i_i64[0];
              sub_C8D5F0((__int64)&v154, v156, v74, 8u, v12, v13);
              v73 = (unsigned int)j;
              v12 = v124;
            }
            v68 += 3;
            *(_QWORD *)&v154[8 * v73] = v12;
          }
LABEL_82:
          v75 = *(_QWORD *)(v4 + 408);
          v25 = *(unsigned int *)(v75 + 40);
          if ( (_DWORD)v25 )
          {
            v26 = (unsigned int)v25;
            v76 = 0;
            while ( 1 )
            {
              v11 = (__int64)v154;
              v77 = 8LL * (unsigned int)j;
              v78 = &v154[v77];
              v79 = *(_QWORD *)(*(_QWORD *)(v75 + 32) + 8 * v76);
              v12 = v77 >> 3;
              v25 = v77 >> 5;
              if ( v25 )
              {
                v80 = v154;
                v25 = (__int64)&v154[32 * v25];
                while ( *v80 != v79 )
                {
                  if ( v80[1] == v79 )
                  {
                    ++v80;
                    goto LABEL_91;
                  }
                  if ( v80[2] == v79 )
                  {
                    v80 += 2;
                    goto LABEL_91;
                  }
                  if ( v80[3] == v79 )
                  {
                    v80 += 3;
                    goto LABEL_91;
                  }
                  v80 += 4;
                  if ( (_QWORD *)v25 == v80 )
                  {
                    v12 = v78 - v80;
                    goto LABEL_111;
                  }
                }
                goto LABEL_91;
              }
              v80 = v154;
LABEL_111:
              if ( v12 == 2 )
                goto LABEL_125;
              if ( v12 == 3 )
                break;
              if ( v12 != 1 )
                goto LABEL_103;
LABEL_114:
              if ( *v80 != v79 )
                goto LABEL_103;
LABEL_91:
              if ( v78 == v80 )
                goto LABEL_103;
              v12 = v160.m128i_i8[8] & 1;
              if ( (v160.m128i_i8[8] & 1) != 0 )
              {
                v13 = (__int64)&v161;
                v81 = 3;
              }
              else
              {
                v91 = v161.m128i_u32[2];
                v13 = v161.m128i_i64[0];
                if ( !v161.m128i_i32[2] )
                  goto LABEL_128;
                v81 = v161.m128i_i32[2] - 1;
              }
              v82 = 0x9DDFEA08EB382D69LL * (HIDWORD(v79) ^ (((8 * v79) & 0x7FFFFFFF8LL) + 12995744));
              v83 = 0x9DDFEA08EB382D69LL * (v82 ^ HIDWORD(v79) ^ (v82 >> 47));
              v25 = v81 & (-348639895 * ((v83 >> 47) ^ (unsigned int)v83));
              v84 = v13 + 16LL * (unsigned int)v25;
              v85 = *(_QWORD *)v84;
              if ( v79 == *(_QWORD *)v84 )
                goto LABEL_95;
              v99 = 1;
              while ( qword_5051170 != v85 )
              {
                v25 = v81 & (unsigned int)(v99 + v25);
                v123 = v99 + 1;
                v84 = v13 + 16LL * (unsigned int)v25;
                v85 = *(_QWORD *)v84;
                if ( *(_QWORD *)v84 == v79 )
                  goto LABEL_95;
                v99 = v123;
              }
              if ( (_BYTE)v12 )
              {
                v98 = 64;
                goto LABEL_129;
              }
              v91 = v161.m128i_u32[2];
LABEL_128:
              v98 = 16 * v91;
LABEL_129:
              v84 = v13 + v98;
LABEL_95:
              v86 = *(_BYTE *)(v84 + 11);
              if ( v86 <= 2u )
              {
                v12 = *(_QWORD *)(v4 + 432);
                v87 = *(_QWORD *)(v12 + 16);
                v25 = *(_QWORD *)(v87 + 88);
                v88 = *(_DWORD *)(v25 + 4 * v76);
                if ( v88 >= *(_DWORD *)(v87 + 284) )
                {
                  *(_DWORD *)(v84 + 8) = v76 & 0xFFFFFF | *(_DWORD *)(v84 + 8) & 0xFF000000;
                  *(_BYTE *)(v84 + 11) = 3;
                  v24 = j;
                  v25 = v11 + 8LL * (unsigned int)j;
                  if ( (_QWORD *)v25 != v80 + 1 )
                  {
                    v120 = v26;
                    memmove(v80, v80 + 1, v25 - (_QWORD)(v80 + 1));
                    v24 = j;
                    v26 = v120;
                  }
                  LODWORD(j) = v24 - 1;
                  if ( v24 == 1 )
                    goto LABEL_25;
                }
                else if ( v86 != 2 )
                {
                  v118 = v26;
                  v122 = *(_QWORD *)(v4 + 432);
                  v89 = sub_E922F0(*(_QWORD **)(v12 + 3616), v88);
                  v26 = v118;
                  v90 = &v89[2 * v25];
                  if ( v89 == v90 )
                  {
LABEL_149:
                    if ( !v86 )
                    {
                      v25 = v76 & 0xFFFFFF;
                      *(_DWORD *)(v84 + 8) = v25 | *(_DWORD *)(v84 + 8) & 0xFF000000;
                      *(_BYTE *)(v84 + 11) = 1;
                    }
                  }
                  else
                  {
                    while ( 1 )
                    {
                      v11 = *(unsigned __int16 *)v89;
                      v12 = (unsigned __int64)(unsigned __int16)v11 >> 6;
                      v25 = *(_QWORD *)(**(_QWORD **)(v122 + 3624) + 8 * v12) & (1LL << v11);
                      if ( v25 )
                        break;
                      v89 += 2;
                      if ( v90 == v89 )
                        goto LABEL_149;
                    }
                    v25 = v76 & 0xFFFFFF;
                    *(_DWORD *)(v84 + 8) = v25 | *(_DWORD *)(v84 + 8) & 0xFF000000;
                    *(_BYTE *)(v84 + 11) = 2;
                  }
                }
              }
LABEL_103:
              if ( v26 == ++v76 )
                goto LABEL_25;
              v75 = *(_QWORD *)(v4 + 408);
            }
            if ( *v80 == v79 )
              goto LABEL_91;
            ++v80;
LABEL_125:
            if ( *v80 == v79 )
              goto LABEL_91;
            ++v80;
            goto LABEL_114;
          }
LABEL_25:
          v27 = v151;
          v157.m128i_i64[0] = (__int64)&v158;
          v157.m128i_i64[1] = 0x100000000LL;
          v28 = 3LL * (unsigned int)v152;
          if ( v151 != &v151[v28] )
          {
            v121 = v4;
            v29 = &v151[v28];
            while ( v27[2].m128i_i8[8] )
            {
              v141 = _mm_loadu_si128(v27);
              v142 = _mm_loadu_si128(v27 + 1);
              v97 = v27[2].m128i_i64[0];
              v147 = 1;
              v143 = v97;
              v146 = v97;
              v144 = v141;
              v145 = v142;
              sub_37BC120((__int64)&v157, &v144, v25, v11, v12, v13);
LABEL_118:
              v27 += 3;
              if ( v29 == v27 )
              {
                v4 = v121;
                goto LABEL_33;
              }
            }
            if ( (v160.m128i_i8[8] & 1) != 0 )
            {
              v30 = &v161;
              v31 = 3;
              goto LABEL_30;
            }
            v96 = v161.m128i_u32[2];
            v30 = (__m128i *)v161.m128i_i64[0];
            if ( v161.m128i_i32[2] )
            {
              v31 = v161.m128i_i32[2] - 1;
LABEL_30:
              v32 = HIDWORD(v27->m128i_i64[0]);
              v33 = 0x9DDFEA08EB382D69LL * (v32 ^ (((8 * v27->m128i_i64[0]) & 0x7FFFFFFF8LL) + 12995744));
              v34 = v31
                  & (-348639895
                   * (((unsigned int)((0x9DDFEA08EB382D69LL * (v33 ^ v32 ^ (v33 >> 47))) >> 32) >> 15)
                    ^ (-348639895 * (v33 ^ v32 ^ (v33 >> 47)))));
              v35 = &v30[v34];
              v13 = v35->m128i_i64[0];
              if ( v35->m128i_i64[0] == v27->m128i_i64[0] )
              {
LABEL_31:
                if ( !v35->m128i_i8[11] )
                {
                  v157.m128i_i32[2] = 0;
                  v4 = v121;
                  goto LABEL_33;
                }
                v92 = v35->m128i_i32[2];
                v11 = v157.m128i_u32[3];
                v93 = &v144;
                v147 = 0;
                v25 = v157.m128i_i64[0];
                v144.m128i_i32[0] = v92 & 0xFFFFFF;
                v94 = v157.m128i_u32[2];
                v12 = v157.m128i_u32[2] + 1LL;
                if ( v12 > v157.m128i_u32[3] )
                {
                  if ( v157.m128i_i64[0] > (unsigned __int64)&v144
                    || (unsigned __int64)&v144 >= v157.m128i_i64[0] + 48 * (unsigned __int64)v157.m128i_u32[2] )
                  {
                    v93 = &v144;
                    sub_C8D5F0((__int64)&v157, &v158, v157.m128i_u32[2] + 1LL, 0x30u, v12, v13);
                    v25 = v157.m128i_i64[0];
                    v94 = v157.m128i_u32[2];
                  }
                  else
                  {
                    v101 = &v144.m128i_i8[-v157.m128i_i64[0]];
                    sub_C8D5F0((__int64)&v157, &v158, v157.m128i_u32[2] + 1LL, 0x30u, v12, v13);
                    v25 = v157.m128i_i64[0];
                    v94 = v157.m128i_u32[2];
                    v93 = (__m128i *)&v101[v157.m128i_i64[0]];
                  }
                }
                v95 = (__m128i *)(v25 + 48 * v94);
                *v95 = _mm_loadu_si128(v93);
                v95[1] = _mm_loadu_si128(v93 + 1);
                v95[2] = _mm_loadu_si128(v93 + 2);
                ++v157.m128i_i32[2];
                goto LABEL_118;
              }
              v102 = 1;
              while ( qword_5051170 != v13 )
              {
                v115 = v102 + 1;
                v34 = v31 & (v102 + v34);
                v35 = &v30[v34];
                v13 = v35->m128i_i64[0];
                if ( v27->m128i_i64[0] == v35->m128i_i64[0] )
                  goto LABEL_31;
                v102 = v115;
              }
              if ( (v160.m128i_i8[8] & 1) != 0 )
              {
                v100 = 4;
                goto LABEL_137;
              }
              v96 = v161.m128i_u32[2];
            }
            v100 = v96;
LABEL_137:
            v35 = &v30[v100];
            goto LABEL_31;
          }
LABEL_33:
          sub_37CD070(*(_QWORD *)(v4 + 432), a2, (__int64)&v136, (char **)&v157);
          if ( !(_DWORD)v152 || (v36 = v157.m128i_i32[2]) != 0 )
          {
LABEL_35:
            v37 = *(_QWORD **)(v4 + 408);
            v38 = sub_B10CD0(v119);
            sub_37BA660(v37, (__int64)&v157, (__int64)&v137, v38, (__int64)&v136);
            v40 = *(unsigned int *)(v4 + 2288);
            v41 = *(_QWORD *)(v4 + 2272);
            v43 = v42;
            if ( (_DWORD)v40 )
            {
              v144.m128i_i64[0] = 0;
              v145.m128i_i8[8] = 0;
              v146 = 0;
              v134 = 0;
              if ( v139 )
                v134 = v138 | (v137.m128i_i32[2] << 16);
              v141.m128i_i64[0] = v140;
              v135 = v137.m128i_i64[0];
              v128 = 1;
              v132 = (v40 - 1) & sub_F11290(&v135, &v134, v141.m128i_i64);
              while ( 1 )
              {
                v44 = v41 + 48LL * v132;
                if ( sub_F34140((__int64)&v137, v44) )
                  break;
                if ( sub_F34140(v44, (__int64)&v144) )
                  goto LABEL_153;
                v132 = (v40 - 1) & (v128 + v132);
                ++v128;
              }
              v45 = v41 + 48LL * v132;
              if ( v44 )
                goto LABEL_41;
LABEL_153:
              v41 = *(_QWORD *)(v4 + 2272);
              v40 = *(unsigned int *)(v4 + 2288);
            }
            v45 = v41 + 48 * v40;
LABEL_41:
            v46 = *(_QWORD *)(v4 + 432);
            v47 = *(unsigned int *)(v45 + 40);
            v48 = *(unsigned int *)(v46 + 3480);
            v49 = *(unsigned int *)(v46 + 3484);
            if ( v48 + 1 > v49 )
            {
              sub_C8D5F0(v46 + 3472, (const void *)(v46 + 3488), v48 + 1, 0x10u, v45, v39);
              v48 = *(unsigned int *)(v46 + 3480);
            }
            v50 = (_QWORD *)(*(_QWORD *)(v46 + 3472) + 16 * v48);
            *v50 = v47;
            v50[1] = v43;
            ++*(_DWORD *)(v46 + 3480);
            sub_37C43E0(*(_QWORD *)(v4 + 432), a2, 0, v49, v45, v39);
            if ( (__m128i *)v157.m128i_i64[0] != &v158 )
              _libc_free(v157.m128i_u64[0]);
            if ( v154 != v156 )
              _libc_free((unsigned __int64)v154);
            if ( (v160.m128i_i8[8] & 1) == 0 )
              sub_C7D6A0(v161.m128i_i64[0], 16LL * v161.m128i_u32[2], 8);
            if ( v151 != (__m128i *)v153 )
              _libc_free((unsigned __int64)v151);
            goto LABEL_51;
          }
          v103 = v160.m128i_i8[8] & 1;
          if ( !((unsigned __int32)v160.m128i_i32[2] >> 1) )
          {
            if ( v103 )
            {
              v116 = &v161;
              v117 = 4;
            }
            else
            {
              v116 = (__m128i *)v161.m128i_i64[0];
              v117 = v161.m128i_u32[2];
            }
            v106 = &v116[v117];
            v109 = v106;
            goto LABEL_161;
          }
          if ( v103 )
          {
            v109 = (__m128i *)v163;
            v107 = qword_5051170;
            v108 = qword_5051168;
            v106 = &v161;
            goto LABEL_158;
          }
          v104 = v161.m128i_u32[2];
          v105 = (__m128i *)v161.m128i_i64[0];
          v106 = (__m128i *)v161.m128i_i64[0];
          v107 = qword_5051170;
          v108 = qword_5051168;
          v109 = (__m128i *)(v161.m128i_i64[0] + 16LL * v161.m128i_u32[2]);
          if ( (__m128i *)v161.m128i_i64[0] == v109 )
          {
LABEL_163:
            v110 = v104;
          }
          else
          {
            do
            {
LABEL_158:
              if ( v106->m128i_i64[0] != v107 && v106->m128i_i64[0] != v108 )
                break;
              ++v106;
            }
            while ( v106 != v109 );
LABEL_161:
            if ( !v103 )
            {
              v105 = (__m128i *)v161.m128i_i64[0];
              v104 = v161.m128i_u32[2];
              goto LABEL_163;
            }
            v105 = &v161;
            v110 = 4;
          }
          v111 = 0;
          if ( v106 != &v105[v110] )
          {
            do
            {
              if ( !v106->m128i_i8[11] )
              {
                if ( *(_DWORD *)(v4 + 416) != (v106->m128i_i32[0] & 0xFFFFF)
                  || *(_DWORD *)(v4 + 420) >= (((unsigned __int64)v106->m128i_i64[0] >> 20) & 0xFFFFF) )
                {
                  goto LABEL_35;
                }
                if ( v111 < (((unsigned __int64)v106->m128i_i64[0] >> 20) & 0xFFFFF) )
                  v111 = ((unsigned __int64)v106->m128i_i64[0] >> 20) & 0xFFFFF;
              }
              for ( ++v106; v106 != v109; ++v106 )
              {
                if ( qword_5051170 != v106->m128i_i64[0] && qword_5051168 != v106->m128i_i64[0] )
                  break;
              }
            }
            while ( v106 != &v105[v110] );
            v36 = v111;
          }
          v112 = sub_B10CD0(v119);
          v113 = sub_37CCE30(v4 + 2264, &v137, v112);
          v114 = *(_QWORD *)(v4 + 432);
          v133 = v113;
          v144.m128i_i64[0] = sub_2E891C0(a2);
          v144.m128i_i16[4] = 256;
          sub_37CF130(v114, v133, &v144, (__int64)&v151, v36);
          goto LABEL_35;
        }
        v56 = HIDWORD(v152);
        v55 = v151;
      }
      v65 = *(_QWORD *)v12;
      v162.m128i_i8[8] = 0;
      v62 = &v160;
      v160.m128i_i64[0] = v65;
      v61 = v11 + 1;
      if ( v11 + 1 <= v56 )
        goto LABEL_63;
LABEL_67:
      v126 = v12;
      v129 = v57;
      if ( v55 > &v160 || &v160 >= &v55[3 * v11] )
      {
        v62 = &v160;
        sub_C8D5F0((__int64)&v151, v153, v61, 0x30u, v12, v13);
        v55 = v151;
        v11 = (unsigned int)v152;
        v12 = v126;
        v57 = v129;
      }
      else
      {
        v66 = (char *)((char *)&v160 - (char *)v55);
        sub_C8D5F0((__int64)&v151, v153, v61, 0x30u, v12, v13);
        v55 = v151;
        v11 = (unsigned int)v152;
        v57 = v129;
        v12 = v126;
        v62 = (__m128i *)&v66[(_QWORD)v151];
      }
      goto LABEL_63;
    }
LABEL_51:
    if ( v148 != (unsigned int *)v150 )
      _libc_free((unsigned __int64)v148);
  }
  return 1;
}
