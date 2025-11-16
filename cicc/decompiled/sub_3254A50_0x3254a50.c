// Function: sub_3254A50
// Address: 0x3254a50
//
__int64 __fastcall sub_3254A50(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 *a5)
{
  __int64 v5; // rbx
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v10; // rax
  char v11; // r10
  int v12; // r15d
  unsigned int v13; // esi
  __int64 v14; // r8
  unsigned int v15; // r13d
  unsigned int v16; // r14d
  __int64 v17; // r10
  unsigned int *v18; // rdi
  int v19; // r11d
  int v20; // r15d
  unsigned int i; // eax
  unsigned int *v22; // r12
  unsigned int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // rdx
  __m128i *v36; // rcx
  __m128i *v37; // rax
  __int64 v38; // rbx
  int v39; // ecx
  __int64 v40; // r15
  __int64 v41; // r13
  __int64 v42; // r9
  unsigned int v43; // edi
  __int64 *v44; // r12
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rsi
  int v48; // edx
  __int64 v49; // rsi
  __int64 v50; // rdx
  __int32 v51; // edi
  __int64 v52; // rsi
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // r9
  __int64 v55; // r11
  unsigned __int64 v56; // r11
  __m128i *v57; // r12
  __m128i *v58; // rdx
  unsigned __int64 v59; // rdx
  unsigned __int64 v60; // r10
  __m128i *v61; // rbx
  unsigned __int64 v62; // rcx
  unsigned __int64 v63; // rax
  int v64; // eax
  char v65; // al
  char v66; // al
  __int64 v67; // rax
  int v68; // edx
  __int64 v69; // rcx
  int v70; // edx
  unsigned int v71; // r14d
  __int64 *v72; // rax
  __int64 v73; // rsi
  unsigned int v74; // r13d
  unsigned __int64 v75; // rax
  __int64 v76; // r12
  unsigned __int64 v77; // rdx
  __m128i v78; // xmm2
  int v79; // edx
  __int64 v80; // rdx
  unsigned __int64 v81; // rdi
  unsigned __int64 v82; // rcx
  __m128i *v83; // rsi
  unsigned __int64 v84; // r8
  __m128i *v85; // rcx
  __int64 v86; // rax
  __int64 v87; // rcx
  int v88; // r12d
  int v89; // r11d
  int v90; // eax
  const void *v91; // rsi
  __int8 *v92; // r12
  const void *v93; // rsi
  __int64 v95; // rbx
  const void *v96; // rsi
  int v97; // edi
  const void *v98; // rsi
  __int8 *v99; // rbx
  int v100; // ecx
  int v101; // edi
  int v102; // ecx
  unsigned int *v103; // rdx
  unsigned int v104; // esi
  unsigned int v105; // esi
  unsigned int v106; // edx
  int v107; // eax
  int v108; // eax
  __int64 v109; // rax
  unsigned __int64 v110; // rcx
  unsigned __int64 v111; // rdx
  __int64 v112; // rcx
  __m128i *v113; // rdx
  __m128i *v114; // rax
  unsigned __int64 v115; // r13
  __int64 v116; // rdi
  const void *v117; // rsi
  int v118; // esi
  int v119; // esi
  __int64 v120; // rdi
  int v121; // ecx
  unsigned int j; // r15d
  _DWORD *v123; // rdx
  unsigned int v124; // r15d
  unsigned int v125; // eax
  int v126; // eax
  __int64 v127; // [rsp+8h] [rbp-D8h]
  char v128; // [rsp+10h] [rbp-D0h]
  char v129; // [rsp+10h] [rbp-D0h]
  char v130; // [rsp+10h] [rbp-D0h]
  __int64 v131; // [rsp+10h] [rbp-D0h]
  char v132; // [rsp+10h] [rbp-D0h]
  char v133; // [rsp+10h] [rbp-D0h]
  __int64 v134; // [rsp+18h] [rbp-C8h]
  __int64 v135; // [rsp+18h] [rbp-C8h]
  __int64 v136; // [rsp+18h] [rbp-C8h]
  __int8 *v137; // [rsp+18h] [rbp-C8h]
  __int64 v138; // [rsp+18h] [rbp-C8h]
  __int64 v139; // [rsp+18h] [rbp-C8h]
  __int64 v140; // [rsp+18h] [rbp-C8h]
  unsigned __int8 v144; // [rsp+43h] [rbp-9Dh]
  unsigned __int8 v145; // [rsp+43h] [rbp-9Dh]
  char v146; // [rsp+43h] [rbp-9Dh]
  int v147; // [rsp+44h] [rbp-9Ch]
  __int64 v149; // [rsp+50h] [rbp-90h]
  __int64 v150; // [rsp+58h] [rbp-88h]
  __int64 v151; // [rsp+58h] [rbp-88h]
  __int64 v152; // [rsp+58h] [rbp-88h]
  __int64 v153; // [rsp+58h] [rbp-88h]
  __int64 v154; // [rsp+58h] [rbp-88h]
  __int64 v155; // [rsp+58h] [rbp-88h]
  __int64 v156; // [rsp+58h] [rbp-88h]
  __int64 v157; // [rsp+60h] [rbp-80h] BYREF
  __int64 v158; // [rsp+68h] [rbp-78h]
  __int64 v159; // [rsp+70h] [rbp-70h]
  unsigned int v160; // [rsp+78h] [rbp-68h]
  __m128i v161; // [rsp+80h] [rbp-60h] BYREF
  __m128i v162; // [rsp+90h] [rbp-50h] BYREF
  __int64 v163; // [rsp+A0h] [rbp-40h]
  char v164; // [rsp+A8h] [rbp-38h]

  v157 = 0;
  v158 = 0;
  v159 = 0;
  v160 = 0;
  sub_32546F0(a1, (__int64)a4, (__int64)&v157);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_QWORD *)(v5 + 536);
  v147 = *(_DWORD *)(*(_QWORD *)(v5 + 208) + 336LL);
  v7 = *(_QWORD *)(v5 + 232);
  v8 = *(_QWORD *)(v7 + 328);
  v127 = v7 + 320;
  if ( v8 != v7 + 320 )
  {
    v10 = *(_QWORD *)(v7 + 328);
    v11 = 0;
    v12 = 0;
    v149 = v8;
    while ( 2 )
    {
      if ( v149 != v10 && !*(_BYTE *)(v149 + 260) )
        goto LABEL_15;
      v13 = *(_DWORD *)(v5 + 328);
      v14 = v5 + 304;
      v15 = *(_DWORD *)(v149 + 252);
      v16 = *(_DWORD *)(v149 + 256);
      if ( !v13 )
      {
        ++*(_QWORD *)(v5 + 304);
        goto LABEL_109;
      }
      v17 = *(_QWORD *)(v5 + 312);
      v18 = 0;
      v19 = 1;
      v20 = ((0xBF58476D1CE4E5B9LL * ((37 * v16) | ((unsigned __int64)(37 * v15) << 32))) >> 31) ^ (756364221 * v16);
      for ( i = v20 & (v13 - 1); ; i = (v13 - 1) & v23 )
      {
        v22 = (unsigned int *)(v17 + 12LL * i);
        if ( v15 == *v22 && v16 == v22[1] )
        {
          v24 = v22[2];
          v25 = v5;
          goto LABEL_12;
        }
        if ( !*v22 )
          break;
LABEL_9:
        v23 = v19 + i;
        ++v19;
      }
      v106 = v22[1];
      if ( v106 != -1 )
      {
        if ( !v18 && v106 == -2 )
          v18 = (unsigned int *)(v17 + 12LL * i);
        goto LABEL_9;
      }
      v107 = *(_DWORD *)(v5 + 320);
      if ( v18 )
        v22 = v18;
      ++*(_QWORD *)(v5 + 304);
      v108 = v107 + 1;
      if ( 4 * v108 < 3 * v13 )
      {
        if ( v13 - *(_DWORD *)(v5 + 324) - v108 > v13 >> 3 )
          goto LABEL_124;
        v156 = a2;
        sub_31E5EB0(v5 + 304, v13);
        v118 = *(_DWORD *)(v5 + 328);
        if ( v118 )
        {
          v119 = v118 - 1;
          v121 = 1;
          v22 = 0;
          a2 = v156;
          for ( j = v119 & v20; ; j = v119 & v124 )
          {
            v120 = *(_QWORD *)(v5 + 312);
            v123 = (_DWORD *)(v120 + 12LL * j);
            if ( v15 == *v123 && v16 == v123[1] )
            {
              v22 = (unsigned int *)(v120 + 12LL * j);
              v108 = *(_DWORD *)(v5 + 320) + 1;
              goto LABEL_124;
            }
            if ( !*v123 )
            {
              v126 = v123[1];
              if ( v126 == -1 )
              {
                v108 = *(_DWORD *)(v5 + 320) + 1;
                if ( !v22 )
                  v22 = (unsigned int *)(v120 + 12LL * j);
                goto LABEL_124;
              }
              if ( v126 == -2 && !v22 )
                v22 = (unsigned int *)(v120 + 12LL * j);
            }
            v124 = v121 + j;
            ++v121;
          }
        }
LABEL_160:
        ++*(_DWORD *)(v5 + 320);
        BUG();
      }
LABEL_109:
      v154 = a2;
      sub_31E5EB0(v5 + 304, 2 * v13);
      v100 = *(_DWORD *)(v5 + 328);
      if ( !v100 )
        goto LABEL_160;
      v14 = *(_QWORD *)(v5 + 312);
      a2 = v154;
      v101 = v100 - 1;
      v102 = 1;
      v103 = 0;
      v104 = v101
           & (((0xBF58476D1CE4E5B9LL * ((37 * v16) | ((unsigned __int64)(37 * v15) << 32))) >> 31)
            ^ (756364221 * v16));
      while ( 2 )
      {
        v22 = (unsigned int *)(v14 + 12LL * v104);
        if ( v15 == *v22 && v16 == v22[1] )
        {
          v108 = *(_DWORD *)(v5 + 320) + 1;
          goto LABEL_124;
        }
        if ( *v22 )
        {
LABEL_113:
          v105 = v102 + v104;
          ++v102;
          v104 = v101 & v105;
          continue;
        }
        break;
      }
      v125 = v22[1];
      if ( v125 != -1 )
      {
        if ( !v103 && v125 == -2 )
          v103 = (unsigned int *)(v14 + 12LL * v104);
        goto LABEL_113;
      }
      v108 = *(_DWORD *)(v5 + 320) + 1;
      if ( v103 )
        v22 = v103;
LABEL_124:
      *(_DWORD *)(v5 + 320) = v108;
      if ( *v22 || v22[1] != -1 )
        --*(_DWORD *)(v5 + 324);
      *v22 = v15;
      v22[1] = v16;
      v22[2] = 0;
      v109 = *(unsigned int *)(v5 + 344);
      v110 = *(unsigned int *)(v5 + 348);
      v161 = (__m128i)__PAIR64__(v16, v15);
      v111 = v109 + 1;
      v162.m128i_i64[0] = 0;
      if ( v109 + 1 > v110 )
      {
        v115 = *(_QWORD *)(v5 + 336);
        v155 = a2;
        v116 = v5 + 336;
        v117 = (const void *)(v5 + 352);
        if ( v115 > (unsigned __int64)&v161 || (unsigned __int64)&v161 >= v115 + 24 * v109 )
        {
          sub_C8D5F0(v116, v117, v111, 0x18u, v14, a2);
          v112 = *(_QWORD *)(v5 + 336);
          v109 = *(unsigned int *)(v5 + 344);
          v113 = &v161;
          a2 = v155;
        }
        else
        {
          sub_C8D5F0(v116, v117, v111, 0x18u, v14, a2);
          v112 = *(_QWORD *)(v5 + 336);
          v109 = *(unsigned int *)(v5 + 344);
          a2 = v155;
          v113 = (__m128i *)((char *)&v161 + v112 - v115);
        }
      }
      else
      {
        v112 = *(_QWORD *)(v5 + 336);
        v113 = &v161;
      }
      v114 = (__m128i *)(v112 + 24 * v109);
      *v114 = _mm_loadu_si128(v113);
      v114[1].m128i_i64[0] = v113[1].m128i_i64[0];
      v24 = *(unsigned int *)(v5 + 344);
      *(_DWORD *)(v5 + 344) = v24 + 1;
      v22[2] = v24;
      v25 = *(_QWORD *)(a1 + 8);
LABEL_12:
      v150 = a2;
      v26 = *(_QWORD *)(v5 + 336) + 24 * v24;
      v27 = *(_QWORD *)(v26 + 8);
      v28 = *(_QWORD *)(v26 + 16);
      v161.m128i_i64[0] = v27;
      v161.m128i_i64[1] = v28;
      v29 = sub_31E4810(v25, v149);
      a2 = v150;
      v163 = 0;
      v162.m128i_i64[0] = v29;
      v31 = *(unsigned int *)(v150 + 8);
      v32 = *(unsigned int *)(a3 + 12);
      v164 = 0;
      v162.m128i_i64[1] = v31;
      v33 = *(unsigned int *)(a3 + 8);
      v34 = v33 + 1;
      if ( v33 + 1 > v32 )
      {
        v95 = *(_QWORD *)a3;
        v96 = (const void *)(a3 + 16);
        if ( *(_QWORD *)a3 > (unsigned __int64)&v161 || (unsigned __int64)&v161 >= v95 + 48 * v33 )
        {
          sub_C8D5F0(a3, v96, v34, 0x30u, v30, v150);
          v35 = *(_QWORD *)a3;
          v33 = *(unsigned int *)(a3 + 8);
          v36 = &v161;
          a2 = v150;
        }
        else
        {
          sub_C8D5F0(a3, v96, v34, 0x30u, v30, v150);
          v35 = *(_QWORD *)a3;
          v33 = *(unsigned int *)(a3 + 8);
          a2 = v150;
          v36 = (__m128i *)((char *)&v161 + *(_QWORD *)a3 - v95);
        }
      }
      else
      {
        v35 = *(_QWORD *)a3;
        v36 = &v161;
      }
      v12 = 0;
      v11 = 0;
      v6 = 0;
      v37 = (__m128i *)(v35 + 48 * v33);
      *v37 = _mm_loadu_si128(v36);
      v37[1] = _mm_loadu_si128(v36 + 1);
      v37[2] = _mm_loadu_si128(v36 + 2);
      ++*(_DWORD *)(a3 + 8);
LABEL_15:
      if ( *(_BYTE *)(v149 + 216) )
        *(_BYTE *)(*(_QWORD *)a3 + 48LL * *(unsigned int *)(a3 + 8) - 8) = 1;
      v38 = *(_QWORD *)(v149 + 56);
      v151 = v149 + 48;
      if ( v38 != v149 + 48 )
      {
        v39 = v12;
        v40 = a2;
        while ( 1 )
        {
          if ( *(_WORD *)(v38 + 68) != 4 )
          {
            v64 = *(_DWORD *)(v38 + 44);
            if ( (v64 & 4) != 0 || (v64 & 8) == 0 )
            {
              v65 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v38 + 16) + 24LL) >> 7;
            }
            else
            {
              v144 = v39;
              v128 = v11;
              v134 = v6;
              v65 = sub_2E88A90(v38, 128, 1);
              v6 = v134;
              v11 = v128;
              v39 = v144;
            }
            if ( v65 )
            {
              v145 = v39;
              v129 = v11;
              v135 = v6;
              v66 = sub_3253050(v38);
              v6 = v135;
              v39 = v145;
              v11 = v66 ^ 1 | v129;
            }
            goto LABEL_37;
          }
          v41 = *(_QWORD *)(*(_QWORD *)(v38 + 32) + 24LL);
          if ( v41 == v6 )
            v11 = 0;
          if ( v160 )
          {
            v42 = v160 - 1;
            v43 = v42 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
            v44 = (__int64 *)(v158 + 16LL * v43);
            v45 = *v44;
            if ( v41 != *v44 )
            {
              v88 = 1;
              while ( v45 != -4096 )
              {
                v89 = v88 + 1;
                v43 = v42 & (v88 + v43);
                v44 = (__int64 *)(v158 + 16LL * v43);
                v45 = *v44;
                if ( v41 == *v44 )
                  goto LABEL_25;
                v88 = v89;
              }
              goto LABEL_37;
            }
LABEL_25:
            if ( v44 != (__int64 *)(v158 + 16LL * v160) )
              break;
          }
LABEL_37:
          if ( (*(_BYTE *)v38 & 4) != 0 )
          {
            v38 = *(_QWORD *)(v38 + 8);
            if ( v151 == v38 )
              goto LABEL_39;
          }
          else
          {
            while ( (*(_BYTE *)(v38 + 44) & 8) != 0 )
              v38 = *(_QWORD *)(v38 + 8);
            v38 = *(_QWORD *)(v38 + 8);
            if ( v151 == v38 )
            {
LABEL_39:
              a2 = v40;
              v12 = v39;
              goto LABEL_40;
            }
          }
        }
        v46 = *(_QWORD *)(*a4 + 8LL * *((unsigned int *)v44 + 2));
        if ( v11 )
        {
          v47 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 208LL);
          v48 = *(_DWORD *)(v47 + 336);
          if ( (v48 & 0xFFFFFFFD) != 1 && v48 != 7 )
          {
            if ( v48 == 4 )
            {
              v79 = *(_DWORD *)(v47 + 344);
              if ( !v79 || v79 == 6 )
                goto LABEL_31;
            }
            else if ( v48 != 6 )
            {
              goto LABEL_31;
            }
          }
          v80 = *(unsigned int *)(v40 + 8);
          v81 = *(unsigned int *)(v40 + 12);
          v161.m128i_i64[0] = v6;
          v82 = *(_QWORD *)v40;
          v83 = &v161;
          v161.m128i_i64[1] = v41;
          v84 = v80 + 1;
          v162.m128i_i64[0] = 0;
          v162.m128i_i32[2] = 0;
          if ( v80 + 1 > v81 )
          {
            v93 = (const void *)(v40 + 16);
            if ( v82 > (unsigned __int64)&v161 || (unsigned __int64)&v161 >= v82 + 32 * v80 )
            {
              v132 = v11;
              v138 = v46;
              sub_C8D5F0(v40, v93, v84, 0x20u, v84, v42);
              v82 = *(_QWORD *)v40;
              v80 = *(unsigned int *)(v40 + 8);
              v83 = &v161;
              v11 = v132;
              v46 = v138;
            }
            else
            {
              v146 = v11;
              v137 = &v161.m128i_i8[-v82];
              v131 = v46;
              sub_C8D5F0(v40, v93, v84, 0x20u, v84, v42);
              v82 = *(_QWORD *)v40;
              v80 = *(unsigned int *)(v40 + 8);
              v46 = v131;
              v11 = v146;
              v83 = (__m128i *)&v137[*(_QWORD *)v40];
            }
          }
          v85 = (__m128i *)(32 * v80 + v82);
          *v85 = _mm_loadu_si128(v83);
          v85[1] = _mm_loadu_si128(v83 + 1);
          v39 = 0;
          ++*(_DWORD *)(v40 + 8);
        }
LABEL_31:
        v6 = *(_QWORD *)(*(_QWORD *)(v46 + 32) + 8LL * *((unsigned int *)v44 + 3));
        if ( !*(_QWORD *)(v46 + 88) )
        {
          v39 = 0;
          goto LABEL_37;
        }
        v49 = *((unsigned int *)v44 + 2);
        v161.m128i_i64[0] = v41;
        v161.m128i_i64[1] = v6;
        v50 = *a5;
        v162.m128i_i64[0] = v46;
        v51 = *(_DWORD *)(v50 + 4 * v49);
        v162.m128i_i32[2] = v51;
        LOBYTE(v39) = (v147 != 2) & v39;
        if ( (_BYTE)v39 )
        {
          v52 = *(unsigned int *)(v40 + 8);
          v53 = *(_QWORD *)v40;
          v54 = 32 * v52;
          v55 = *(_QWORD *)v40 + 32 * v52 - 32;
          if ( *(_QWORD *)(v55 + 16) == v46 && v51 == *(_DWORD *)(v55 + 24) )
          {
            *(_QWORD *)(v55 + 8) = v6;
            goto LABEL_37;
          }
          goto LABEL_34;
        }
        if ( v147 != 2 )
        {
          v52 = *(unsigned int *)(v40 + 8);
          v53 = *(_QWORD *)v40;
          v54 = 32 * v52;
LABEL_34:
          v56 = v52 + 1;
          v57 = &v161;
          if ( v52 + 1 > (unsigned __int64)*(unsigned int *)(v40 + 12) )
          {
            v130 = v11;
            v91 = (const void *)(v40 + 16);
            v136 = v6;
            if ( v53 > (unsigned __int64)&v161 || (v54 += v53, (unsigned __int64)&v161 >= v54) )
            {
              sub_C8D5F0(v40, v91, v56, 0x20u, v6, v54);
              v53 = *(_QWORD *)v40;
              v57 = &v161;
              v11 = v130;
              v6 = v136;
              v54 = 32LL * *(unsigned int *)(v40 + 8);
            }
            else
            {
              v92 = &v161.m128i_i8[-v53];
              sub_C8D5F0(v40, v91, v56, 0x20u, v6, v54);
              v53 = *(_QWORD *)v40;
              v6 = v136;
              v11 = v130;
              v57 = (__m128i *)&v92[*(_QWORD *)v40];
              v54 = 32LL * *(unsigned int *)(v40 + 8);
            }
          }
          v58 = (__m128i *)(v54 + v53);
          *v58 = _mm_loadu_si128(v57);
          v58[1] = _mm_loadu_si128(v57 + 1);
          ++*(_DWORD *)(v40 + 8);
          goto LABEL_36;
        }
        v67 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL);
        v68 = *(_DWORD *)(v67 + 544);
        v69 = *(_QWORD *)(v67 + 528);
        if ( v68 )
        {
          v70 = v68 - 1;
          v71 = v70 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
          v72 = (__int64 *)(v69 + 16LL * v71);
          v73 = *v72;
          if ( v41 == *v72 )
          {
LABEL_61:
            v74 = *((_DWORD *)v72 + 2);
            v75 = *(unsigned int *)(v40 + 8);
            v76 = 32LL * (v74 - 1);
            if ( (unsigned int)v75 < v74 && v74 != v75 )
            {
              if ( v74 >= v75 )
              {
                if ( v74 > (unsigned __int64)*(unsigned int *)(v40 + 12) )
                {
                  v133 = v11;
                  v139 = v6;
                  sub_C8D5F0(v40, (const void *)(v40 + 16), v74, 0x20u, v6, v42);
                  v75 = *(unsigned int *)(v40 + 8);
                  v11 = v133;
                  v6 = v139;
                }
                v77 = *(_QWORD *)v40;
                v86 = *(_QWORD *)v40 + 32 * v75;
                v87 = *(_QWORD *)v40 + 32LL * v74;
                if ( v86 != v87 )
                {
                  do
                  {
                    if ( v86 )
                    {
                      *(_QWORD *)v86 = 0;
                      *(_QWORD *)(v86 + 8) = 0;
                      *(_QWORD *)(v86 + 16) = 0;
                      *(_DWORD *)(v86 + 24) = 0;
                    }
                    v86 += 32;
                  }
                  while ( v87 != v86 );
                  v77 = *(_QWORD *)v40;
                }
                *(_DWORD *)(v40 + 8) = v74;
                goto LABEL_63;
              }
              *(_DWORD *)(v40 + 8) = v74;
            }
            v77 = *(_QWORD *)v40;
LABEL_63:
            v78 = _mm_loadu_si128(&v162);
            *(__m128i *)(v77 + v76) = _mm_loadu_si128(&v161);
            *(__m128i *)(v77 + v76 + 16) = v78;
LABEL_36:
            v39 = 1;
            goto LABEL_37;
          }
          v90 = 1;
          while ( v73 != -4096 )
          {
            v97 = v90 + 1;
            v71 = v70 & (v90 + v71);
            v72 = (__int64 *)(v69 + 16LL * v71);
            v73 = *v72;
            if ( v41 == *v72 )
              goto LABEL_61;
            v90 = v97;
          }
        }
        v76 = 0x1FFFFFFFE0LL;
        v77 = *(_QWORD *)v40;
        goto LABEL_63;
      }
LABEL_40:
      if ( v149 == (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL) + 320LL) & 0xFFFFFFFFFFFFFFF8LL)
        || *(_BYTE *)(v149 + 261) )
      {
        v59 = *(unsigned int *)(a2 + 8);
        if ( v147 != 2 && v11 )
        {
          v60 = v59 + 1;
          v161.m128i_i64[0] = v6;
          v61 = &v161;
          v162 = 0;
          v62 = *(unsigned int *)(a2 + 12);
          v161.m128i_i64[1] = *(_QWORD *)(*(_QWORD *)a3 + 48LL * *(unsigned int *)(a3 + 8) - 40);
          v63 = *(_QWORD *)a2;
          if ( v59 + 1 > v62 )
          {
            v140 = v6;
            v98 = (const void *)(a2 + 16);
            if ( v63 > (unsigned __int64)&v161 || (unsigned __int64)&v161 >= v63 + 32 * v59 )
            {
              v153 = a2;
              sub_C8D5F0(a2, v98, v60, 0x20u, v6, a2);
              a2 = v153;
              v61 = &v161;
              v6 = v140;
              v63 = *(_QWORD *)v153;
              v59 = *(unsigned int *)(v153 + 8);
            }
            else
            {
              v152 = a2;
              v99 = &v161.m128i_i8[-v63];
              sub_C8D5F0(a2, v98, v60, 0x20u, v6, a2);
              a2 = v152;
              v6 = v140;
              v63 = *(_QWORD *)v152;
              v59 = *(unsigned int *)(v152 + 8);
              v61 = (__m128i *)&v99[*(_QWORD *)v152];
            }
          }
          v11 = 0;
          v59 = v63 + 32 * v59;
          *(__m128i *)v59 = _mm_loadu_si128(v61);
          *(__m128i *)(v59 + 16) = _mm_loadu_si128(v61 + 1);
          LODWORD(v59) = *(_DWORD *)(a2 + 8) + 1;
          *(_DWORD *)(a2 + 8) = v59;
        }
        *(_QWORD *)(*(_QWORD *)a3 + 48LL * *(unsigned int *)(a3 + 8) - 16) = (unsigned int)v59;
      }
      v149 = *(_QWORD *)(v149 + 8);
      if ( v127 != v149 )
      {
        v5 = *(_QWORD *)(a1 + 8);
        v10 = *(_QWORD *)(*(_QWORD *)(v5 + 232) + 328LL);
        continue;
      }
      break;
    }
  }
  return sub_C7D6A0(v158, 16LL * v160, 8);
}
