// Function: sub_1518180
// Address: 0x1518180
//
_BYTE *__fastcall sub_1518180(_BYTE *a1, __m128i *a2)
{
  __int64 v2; // r13
  const __m128i *v3; // rbx
  __int64 *v4; // rdi
  __int64 v5; // rsi
  char *v6; // rdx
  _QWORD *v7; // rdi
  unsigned __int64 v8; // r8
  __int64 v9; // r8
  _QWORD *v10; // r13
  __int64 v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rbx
  volatile signed __int32 *v14; // r12
  signed __int32 v15; // eax
  signed __int32 v16; // eax
  __int64 v17; // rax
  char *v18; // rbx
  __int64 v19; // r12
  unsigned __int64 i; // r15
  unsigned __int64 v21; // r14
  _QWORD *v22; // rax
  char *v23; // rsi
  __int64 v24; // rcx
  unsigned int *v25; // r15
  unsigned __int64 v26; // rax
  char v27; // al
  unsigned __int64 v28; // rdi
  unsigned __int64 v30; // r12
  unsigned __int64 v31; // rbx
  __int64 v32; // r15
  _QWORD *v33; // rbx
  __int64 v34; // r14
  unsigned __int64 v35; // r12
  int v36; // eax
  const char *v37; // rax
  __int64 v38; // rax
  __int64 v39; // r8
  _QWORD *v40; // r13
  __int64 v41; // rdi
  __int64 v42; // r15
  __int64 v43; // rbx
  volatile signed __int32 *v44; // r12
  signed __int32 v45; // eax
  signed __int32 v46; // eax
  __int64 v47; // rax
  __int64 v48; // rax
  char v49; // al
  _QWORD *v50; // r15
  __int64 v51; // r14
  __int64 v52; // r12
  int v53; // eax
  _QWORD *v54; // rsi
  _QWORD *v55; // rdi
  const char *v56; // rax
  unsigned __int64 v57; // rsi
  unsigned int v58; // ebx
  unsigned __int8 *v59; // rbx
  __int64 v60; // rax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rdi
  unsigned int v63; // eax
  __int64 v64; // rax
  const void *v65; // r8
  __int64 v66; // rax
  __int64 v67; // rdx
  unsigned int *v68; // r12
  unsigned int *v69; // rcx
  __int64 *v70; // rdx
  unsigned int *v71; // rbx
  _BYTE *v72; // rsi
  unsigned __int8 *v73; // rax
  unsigned __int64 v74; // rsi
  unsigned int v75; // ebx
  _QWORD *v76; // rax
  __int64 v77; // rdx
  char v78; // al
  unsigned __int64 v79; // rax
  unsigned __int64 v80; // rsi
  unsigned int v81; // ebx
  const __m128i *v82; // rcx
  unsigned int *v83; // r11
  unsigned __int64 v84; // rsi
  const __m128i *v85; // rax
  const __m128i *v86; // rdi
  __m128i *v87; // r8
  signed __int64 v88; // r12
  __int64 v89; // rax
  __m128i *v90; // rdx
  unsigned __int64 v91; // rax
  unsigned __int64 v92; // rsi
  unsigned int v93; // ebx
  __int64 v94; // rbx
  unsigned int *v95; // r12
  int v96; // eax
  char *v97; // rcx
  __int64 v98; // rax
  unsigned int v99; // r9d
  unsigned int v100; // r8d
  __int64 v101; // r12
  unsigned __int64 v102; // rbx
  unsigned __int64 v103; // rax
  unsigned __int64 v104; // rdi
  _QWORD *v105; // r11
  unsigned int v106; // eax
  unsigned __int64 v107; // rdx
  unsigned __int64 v108; // rsi
  unsigned __int64 v109; // rbx
  __int64 v110; // r12
  __int64 v111; // rax
  __int64 v112; // r15
  _BYTE *v113; // rax
  _BYTE *v114; // rsi
  __int64 v115; // rax
  char *v116; // r12
  signed __int64 v117; // rdx
  unsigned __int64 v118; // rdx
  unsigned __int64 *v119; // r11
  unsigned __int64 v120; // rax
  unsigned int v121; // edx
  unsigned __int64 v122; // rsi
  unsigned __int64 *v123; // r12
  unsigned __int64 v124; // rdx
  unsigned int v125; // esi
  unsigned __int64 v126; // rdx
  unsigned __int64 *v127; // r11
  unsigned __int64 v128; // rax
  unsigned int v129; // edx
  unsigned __int64 v130; // rdx
  unsigned __int64 *v131; // r11
  unsigned __int64 v132; // rax
  unsigned int v133; // edx
  unsigned __int64 v134; // rdx
  unsigned __int64 *v135; // r11
  unsigned __int64 v136; // rax
  unsigned int v137; // edx
  unsigned __int64 v138; // rdx
  unsigned int v139; // r10d
  __int64 v140; // rax
  __int64 v141; // rsi
  __int64 v142; // rdx
  char v143; // cl
  __int64 v144; // rsi
  unsigned int v145; // edx
  __int64 v146; // rdi
  __int64 v147; // r8
  char v148; // cl
  unsigned __int64 v149; // rsi
  unsigned int v150; // edx
  __int64 v151; // rdi
  __int64 v152; // r8
  char v153; // cl
  unsigned __int64 v154; // rsi
  unsigned int v155; // edx
  __int64 v156; // rdi
  __int64 v157; // r8
  char v158; // cl
  unsigned __int64 v159; // rsi
  unsigned int v160; // esi
  __int64 v161; // r8
  __int64 v162; // r9
  char v163; // cl
  unsigned __int64 v164; // rdi
  unsigned int v165; // edx
  __int64 v166; // rdi
  __int64 v167; // r8
  char v168; // cl
  unsigned __int64 v169; // rsi
  __int64 v170; // [rsp+18h] [rbp-2B8h]
  unsigned __int32 v171; // [rsp+24h] [rbp-2ACh]
  __int64 v172; // [rsp+28h] [rbp-2A8h]
  const void *v173; // [rsp+28h] [rbp-2A8h]
  const __m128i *v175; // [rsp+38h] [rbp-298h]
  unsigned int v176; // [rsp+38h] [rbp-298h]
  unsigned int v177; // [rsp+38h] [rbp-298h]
  unsigned int *v178; // [rsp+38h] [rbp-298h]
  __int64 v179; // [rsp+38h] [rbp-298h]
  __int64 m128i_i64; // [rsp+40h] [rbp-290h]
  _QWORD *v181; // [rsp+48h] [rbp-288h]
  __int64 v182; // [rsp+48h] [rbp-288h]
  __int64 *v183; // [rsp+48h] [rbp-288h]
  __m128i *v184; // [rsp+48h] [rbp-288h]
  __int64 v185; // [rsp+48h] [rbp-288h]
  _QWORD *v186; // [rsp+50h] [rbp-280h]
  __int64 v187; // [rsp+50h] [rbp-280h]
  __int64 v188; // [rsp+60h] [rbp-270h] BYREF
  __int64 v189; // [rsp+68h] [rbp-268h] BYREF
  __int64 v190; // [rsp+70h] [rbp-260h] BYREF
  unsigned __int64 v191; // [rsp+78h] [rbp-258h]
  char v192; // [rsp+80h] [rbp-250h] BYREF
  char v193; // [rsp+81h] [rbp-24Fh]
  unsigned int *v194; // [rsp+90h] [rbp-240h] BYREF
  __int64 v195; // [rsp+98h] [rbp-238h]
  _BYTE v196[560]; // [rsp+A0h] [rbp-230h] BYREF

  v2 = (__int64)a2;
  v3 = (const __m128i *)a2[14].m128i_i64[1];
  m128i_i64 = (__int64)a2[18].m128i_i64;
  v4 = &a2[20].m128i_i64[1];
  v175 = v3;
  a2[18] = _mm_loadu_si128(v3);
  a2[19] = _mm_loadu_si128(v3 + 1);
  a2[20].m128i_i32[0] = v3[2].m128i_i32[0];
  a2[20].m128i_i32[1] = v3[2].m128i_i32[1];
  v5 = (__int64)&v3[2].m128i_i64[1];
  sub_1514C40(v4, &v3[2].m128i_i64[1]);
  v170 = v2 + 352;
  if ( (const __m128i *)(v2 + 352) != &v3[4] )
  {
    v7 = *(_QWORD **)(v2 + 352);
    v8 = *(unsigned int *)(v2 + 360);
    v171 = v3[4].m128i_u32[2];
    v172 = v171;
    v186 = v7;
    v181 = v7;
    if ( v171 > v8 )
    {
      if ( v171 <= (unsigned __int64)*(unsigned int *)(v2 + 364) )
      {
        if ( *(_DWORD *)(v2 + 360) )
        {
          v32 = 32 * v8;
          v33 = v7 + 1;
          v34 = v175[4].m128i_i64[0] + 8;
          v35 = v34 + 32 * v8;
          do
          {
            v36 = *(_DWORD *)(v34 - 8);
            v5 = v34;
            v7 = v33;
            v34 += 32;
            v33 += 4;
            *((_DWORD *)v33 - 10) = v36;
            sub_1514C40(v7, (_QWORD *)v5);
          }
          while ( v34 != v35 );
          v8 = v32;
          v172 = v175[4].m128i_u32[2];
          v186 = *(_QWORD **)(v2 + 352);
        }
      }
      else
      {
        v9 = 4 * v8;
        if ( &v7[v9] != v7 )
        {
          v182 = v2;
          v10 = &v7[v9];
          do
          {
            v11 = *(v10 - 3);
            v12 = *(v10 - 2);
            v10 -= 4;
            v13 = v11;
            if ( v12 != v11 )
            {
              do
              {
                while ( 1 )
                {
                  v14 = *(volatile signed __int32 **)(v13 + 8);
                  if ( v14 )
                  {
                    if ( &_pthread_key_create )
                    {
                      v15 = _InterlockedExchangeAdd(v14 + 2, 0xFFFFFFFF);
                    }
                    else
                    {
                      v15 = *((_DWORD *)v14 + 2);
                      *((_DWORD *)v14 + 2) = v15 - 1;
                    }
                    if ( v15 == 1 )
                    {
                      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v14 + 16LL))(v14);
                      if ( &_pthread_key_create )
                      {
                        v16 = _InterlockedExchangeAdd(v14 + 3, 0xFFFFFFFF);
                      }
                      else
                      {
                        v16 = *((_DWORD *)v14 + 3);
                        *((_DWORD *)v14 + 3) = v16 - 1;
                      }
                      if ( v16 == 1 )
                        break;
                    }
                  }
                  v13 += 16;
                  if ( v12 == v13 )
                    goto LABEL_17;
                }
                v13 += 16;
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v14 + 24LL))(v14);
              }
              while ( v12 != v13 );
LABEL_17:
              v11 = v10[1];
            }
            if ( v11 )
              j_j___libc_free_0(v11, v10[3] - v11);
          }
          while ( v10 != v186 );
          v2 = v182;
        }
        v5 = v171;
        v7 = (_QWORD *)v170;
        *(_DWORD *)(v2 + 360) = 0;
        sub_14F2B60(v170, v171);
        v8 = 0;
        v172 = v175[4].m128i_u32[2];
        v186 = *(_QWORD **)(v2 + 352);
      }
      v17 = v175[4].m128i_i64[0];
      v18 = (char *)v186 + v8;
      v19 = v17 + 32 * v172;
      for ( i = v17 + v8; v19 != i; v18 += 32 )
      {
        if ( v18 )
        {
          *(_DWORD *)v18 = *(_DWORD *)i;
          v21 = *(_QWORD *)(i + 16) - *(_QWORD *)(i + 8);
          *((_QWORD *)v18 + 1) = 0;
          *((_QWORD *)v18 + 2) = 0;
          *((_QWORD *)v18 + 3) = 0;
          if ( v21 )
          {
            if ( v21 > 0x7FFFFFFFFFFFFFF0LL )
              sub_4261EA(v7, v5, v6);
            v7 = (_QWORD *)v21;
            v22 = (_QWORD *)sub_22077B0(v21);
          }
          else
          {
            v21 = 0;
            v22 = 0;
          }
          *((_QWORD *)v18 + 1) = v22;
          *((_QWORD *)v18 + 2) = v22;
          *((_QWORD *)v18 + 3) = (char *)v22 + v21;
          v23 = *(char **)(i + 16);
          v6 = *(char **)(i + 8);
          if ( v23 == v6 )
          {
            v5 = (__int64)v22;
          }
          else
          {
            v5 = (__int64)v22 + v23 - v6;
            do
            {
              if ( v22 )
              {
                *v22 = *(_QWORD *)v6;
                v24 = *((_QWORD *)v6 + 1);
                v22[1] = v24;
                if ( v24 )
                {
                  if ( &_pthread_key_create )
                    _InterlockedAdd((volatile signed __int32 *)(v24 + 8), 1u);
                  else
                    ++*(_DWORD *)(v24 + 8);
                }
              }
              v22 += 2;
              v6 += 16;
            }
            while ( (_QWORD *)v5 != v22 );
          }
          *((_QWORD *)v18 + 2) = v5;
        }
        i += 32LL;
      }
      goto LABEL_37;
    }
    v38 = *(_QWORD *)(v2 + 352);
    if ( v171 )
    {
      v50 = v7 + 1;
      v51 = (__int64)&v7[4 * v171 + 1];
      v52 = v3[4].m128i_i64[0] + 8;
      do
      {
        v53 = *(_DWORD *)(v52 - 8);
        v54 = (_QWORD *)v52;
        v55 = v50;
        v50 += 4;
        v52 += 32;
        *((_DWORD *)v50 - 10) = v53;
        sub_1514C40(v55, v54);
      }
      while ( (_QWORD *)v51 != v50 );
      v38 = *(_QWORD *)(v2 + 352);
      v181 = &v186[4 * v171];
      v8 = *(unsigned int *)(v2 + 360);
    }
    v39 = 32 * v8;
    if ( (_QWORD *)(v38 + v39) == v181 )
    {
LABEL_37:
      *(_DWORD *)(v2 + 360) = v171;
      goto LABEL_38;
    }
    v187 = v2;
    v40 = (_QWORD *)(v38 + v39);
    do
    {
      v41 = *(v40 - 3);
      v42 = *(v40 - 2);
      v40 -= 4;
      v43 = v41;
      if ( v42 != v41 )
      {
        do
        {
          while ( 1 )
          {
            v44 = *(volatile signed __int32 **)(v43 + 8);
            if ( v44 )
            {
              if ( &_pthread_key_create )
              {
                v45 = _InterlockedExchangeAdd(v44 + 2, 0xFFFFFFFF);
              }
              else
              {
                v45 = *((_DWORD *)v44 + 2);
                *((_DWORD *)v44 + 2) = v45 - 1;
              }
              if ( v45 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v44 + 16LL))(v44);
                if ( &_pthread_key_create )
                {
                  v46 = _InterlockedExchangeAdd(v44 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v46 = *((_DWORD *)v44 + 3);
                  *((_DWORD *)v44 + 3) = v46 - 1;
                }
                if ( v46 == 1 )
                  break;
              }
            }
            v43 += 16;
            if ( v42 == v43 )
              goto LABEL_73;
          }
          v43 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v44 + 24LL))(v44);
        }
        while ( v42 != v43 );
LABEL_73:
        v41 = v40[1];
      }
      if ( v41 )
        j_j___libc_free_0(v41, v40[3] - v41);
    }
    while ( v181 != v40 );
    v2 = v187;
    *(_DWORD *)(v187 + 360) = v171;
  }
LABEL_38:
  v25 = (unsigned int *)v196;
  v194 = (unsigned int *)v196;
  *(_QWORD *)(v2 + 624) = v175[21].m128i_i64[0];
  v195 = 0x4000000000LL;
  do
  {
LABEL_39:
    v26 = sub_14ED070(m128i_i64, 1);
    if ( (_DWORD)v26 == 2 )
      goto LABEL_56;
    if ( (unsigned int)v26 <= 2 )
    {
      if ( (_DWORD)v26 )
      {
        v27 = a1[8];
        *a1 = 1;
        a1[8] = v27 & 0xFC | 2;
        goto LABEL_43;
      }
LABEL_56:
      v193 = 1;
      v37 = "Malformed block";
LABEL_57:
      v190 = (__int64)v37;
      v192 = 3;
      sub_1514BE0(&v189, (__int64)&v190);
      a1[8] |= 3u;
      *(_QWORD *)a1 = v189 & 0xFFFFFFFFFFFFFFFELL;
      v28 = (unsigned __int64)v194;
      if ( v194 != v25 )
        goto LABEL_44;
      return a1;
    }
  }
  while ( (_DWORD)v26 != 3 );
  v30 = HIDWORD(v26);
  v31 = 8LL * *(_QWORD *)(v2 + 304) - *(unsigned int *)(v2 + 320);
  switch ( (unsigned int)sub_150F8E0(m128i_i64, SHIDWORD(v26)) )
  {
    case 1u:
    case 2u:
    case 3u:
    case 5u:
    case 6u:
    case 7u:
    case 8u:
    case 9u:
    case 0xCu:
    case 0xDu:
    case 0xEu:
    case 0xFu:
    case 0x10u:
    case 0x11u:
    case 0x12u:
    case 0x13u:
    case 0x14u:
    case 0x15u:
    case 0x16u:
    case 0x17u:
    case 0x18u:
    case 0x19u:
    case 0x1Au:
    case 0x1Bu:
    case 0x1Cu:
    case 0x1Du:
    case 0x1Eu:
    case 0x1Fu:
    case 0x20u:
    case 0x21u:
    case 0x22u:
    case 0x25u:
    case 0x28u:
    case 0x29u:
    case 0x2Au:
    case 0x2Bu:
    case 0x2Cu:
      v47 = *(_QWORD *)(v2 + 632);
      if ( v47 != *(_QWORD *)(v2 + 640) )
        *(_QWORD *)(v2 + 640) = v47;
      v48 = *(_QWORD *)(v2 + 656);
      if ( v48 != *(_QWORD *)(v2 + 664) )
        *(_QWORD *)(v2 + 664) = v48;
      v49 = a1[8];
      *a1 = 0;
      a1[8] = v49 & 0xFC | 2;
      goto LABEL_43;
    case 4u:
      *(_DWORD *)(v2 + 320) = 0;
      v92 = (v31 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v2 + 304) = v92;
      v93 = v31 & 0x3F;
      if ( !v93 )
        goto LABEL_126;
      v126 = *(_QWORD *)(v2 + 296);
      if ( v92 >= v126 )
        goto LABEL_188;
      v127 = (unsigned __int64 *)(v92 + *(_QWORD *)(v2 + 288));
      if ( v126 < v92 + 8 )
      {
        *(_QWORD *)(v2 + 312) = 0;
        v150 = v126 - v92;
        if ( !v150 )
          goto LABEL_188;
        v151 = 0;
        v128 = 0;
        do
        {
          v152 = *((unsigned __int8 *)v127 + v151);
          v153 = 8 * v151++;
          v128 |= v152 << v153;
          *(_QWORD *)(v2 + 312) = v128;
        }
        while ( v150 != v151 );
        v154 = v150 + v92;
        v129 = 8 * v150;
        *(_QWORD *)(v2 + 304) = v154;
        *(_DWORD *)(v2 + 320) = v129;
        if ( v93 > v129 )
          goto LABEL_188;
      }
      else
      {
        v128 = *v127;
        *(_QWORD *)(v2 + 304) = v92 + 8;
        v129 = 64;
      }
      *(_DWORD *)(v2 + 320) = v129 - v93;
      *(_QWORD *)(v2 + 312) = v128 >> v93;
LABEL_126:
      LODWORD(v195) = 0;
      sub_1510D70(m128i_i64, v30, (__int64)&v194, 0);
      v94 = (unsigned int)v195;
      v95 = v194;
      v190 = (__int64)&v192;
      v191 = 0x800000000LL;
      v96 = v195;
      if ( (unsigned int)v195 > 8uLL )
      {
        sub_16CD150(&v190, &v192, (unsigned int)v195, 1);
        v97 = (char *)(v190 + (unsigned int)v191);
      }
      else
      {
        if ( !(8LL * (unsigned int)v195) )
          goto LABEL_132;
        v97 = &v192;
      }
      v98 = 0;
      do
      {
        v97[v98] = *(_QWORD *)&v95[2 * v98];
        ++v98;
      }
      while ( v94 != v98 );
      v96 = v94 + v191;
LABEL_132:
      v99 = *(_DWORD *)(v2 + 324);
      v100 = *(_DWORD *)(v2 + 320);
      LODWORD(v191) = v96;
      if ( v99 <= v100 )
      {
        v138 = *(_QWORD *)(v2 + 312);
        *(_DWORD *)(v2 + 320) = v100 - v99;
        *(_QWORD *)(v2 + 312) = v138 >> v99;
        LODWORD(v108) = v138 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v99));
      }
      else
      {
        v101 = 0;
        if ( v100 )
          v101 = *(_QWORD *)(v2 + 312);
        v102 = *(_QWORD *)(v2 + 304);
        v176 = v99 - v100;
        v103 = *(_QWORD *)(v2 + 296);
        if ( v102 >= v103 )
          goto LABEL_188;
        v104 = v102 + 8;
        v105 = (_QWORD *)(v102 + *(_QWORD *)(v2 + 288));
        if ( v103 < v102 + 8 )
        {
          *(_QWORD *)(v2 + 312) = 0;
          v139 = v103 - v102;
          if ( (_DWORD)v103 == (_DWORD)v102 )
          {
            *(_DWORD *)(v2 + 320) = 0;
LABEL_188:
            sub_16BD130("Unexpected end of file", 1);
          }
          v140 = 0;
          v141 = 0;
          do
          {
            v142 = *((unsigned __int8 *)v105 + v140);
            v143 = 8 * v140++;
            v141 |= v142 << v143;
            *(_QWORD *)(v2 + 312) = v141;
          }
          while ( v139 != v140 );
          v106 = 8 * v139;
          v104 = v102 + v139;
        }
        else
        {
          *(_QWORD *)(v2 + 312) = *v105;
          v106 = 64;
        }
        *(_QWORD *)(v2 + 304) = v104;
        *(_DWORD *)(v2 + 320) = v106;
        if ( v176 > v106 )
          goto LABEL_188;
        v107 = *(_QWORD *)(v2 + 312);
        *(_DWORD *)(v2 + 320) = v100 - v99 + v106;
        *(_QWORD *)(v2 + 312) = v107 >> v176;
        v108 = v101 | ((v107 & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v100 - (unsigned __int8)v99 + 64))) << v100);
      }
      v109 = 0;
      LODWORD(v195) = 0;
      sub_1510D70(m128i_i64, v108, (__int64)&v194, 0);
      v177 = v195;
      v110 = sub_1632440(*(_QWORD *)(v2 + 248), v190, (unsigned int)v191);
      v111 = 8LL * v177;
      if ( v177 )
      {
        v178 = v25;
        v112 = v111;
        do
        {
          v113 = (_BYTE *)sub_1517EB0(v2, *(_QWORD *)&v194[v109 / 4]);
          v114 = v113;
          if ( v113 && (unsigned __int8)(*v113 - 4) >= 0x1Fu )
            v114 = 0;
          v109 += 8LL;
          sub_1623CA0(v110, v114);
        }
        while ( v112 != v109 );
        v25 = v178;
      }
      if ( (char *)v190 != &v192 )
        _libc_free(v190);
      goto LABEL_39;
    case 0x23u:
      *(_DWORD *)(v2 + 320) = 0;
      v80 = (v31 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v2 + 304) = v80;
      v81 = v31 & 0x3F;
      if ( !v81 )
        goto LABEL_111;
      v130 = *(_QWORD *)(v2 + 296);
      if ( v80 >= v130 )
        goto LABEL_188;
      v131 = (unsigned __int64 *)(v80 + *(_QWORD *)(v2 + 288));
      if ( v130 < v80 + 8 )
      {
        *(_QWORD *)(v2 + 312) = 0;
        v165 = v130 - v80;
        if ( !v165 )
          goto LABEL_188;
        v166 = 0;
        v132 = 0;
        do
        {
          v167 = *((unsigned __int8 *)v131 + v166);
          v168 = 8 * v166++;
          v132 |= v167 << v168;
          *(_QWORD *)(v2 + 312) = v132;
        }
        while ( v165 != v166 );
        v169 = v165 + v80;
        v133 = 8 * v165;
        *(_QWORD *)(v2 + 304) = v169;
        *(_DWORD *)(v2 + 320) = v133;
        if ( v81 > v133 )
          goto LABEL_188;
      }
      else
      {
        v132 = *v131;
        *(_QWORD *)(v2 + 304) = v80 + 8;
        v133 = 64;
      }
      *(_DWORD *)(v2 + 320) = v133 - v81;
      *(_QWORD *)(v2 + 312) = v132 >> v81;
LABEL_111:
      v190 = 0;
      v191 = 0;
      LODWORD(v195) = 0;
      sub_1510D70(m128i_i64, v30, (__int64)&v194, (unsigned __int8 **)&v190);
      v82 = *(const __m128i **)(v2 + 632);
      v83 = v194;
      v84 = *v194;
      v85 = v82;
      if ( v84 > (__int64)(*(_QWORD *)(v2 + 648) - (_QWORD)v82) >> 4 )
      {
        v86 = *(const __m128i **)(v2 + 640);
        v87 = 0;
        v88 = (char *)v86 - (char *)v82;
        if ( *v194 )
        {
          v89 = sub_22077B0(16 * v84);
          v82 = *(const __m128i **)(v2 + 632);
          v86 = *(const __m128i **)(v2 + 640);
          v87 = (__m128i *)v89;
          v85 = v82;
        }
        if ( v82 != v86 )
        {
          v90 = v87;
          do
          {
            if ( v90 )
              *v90 = _mm_loadu_si128(v85);
            ++v85;
            ++v90;
          }
          while ( v85 != v86 );
          v86 = v82;
        }
        if ( v86 )
        {
          v184 = v87;
          j_j___libc_free_0(v86, *(_QWORD *)(v2 + 648) - (_QWORD)v86);
          v87 = v184;
        }
        *(_QWORD *)(v2 + 632) = v87;
        v83 = v194;
        *(_QWORD *)(v2 + 640) = (char *)v87 + v88;
        *(_QWORD *)(v2 + 648) = &v87[v84];
      }
      v188 = v2;
      sub_1515D60(
        &v189,
        v2,
        v83,
        (unsigned int)v195,
        v190,
        v191,
        (void (__fastcall *)(__int64, __int64, _QWORD))sub_1516E80,
        (__int64)&v188);
      v91 = v189 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v189 & 0xFFFFFFFFFFFFFFFELL) == 0 )
        goto LABEL_39;
      a1[8] |= 3u;
      *(_QWORD *)a1 = v91;
      goto LABEL_43;
    case 0x24u:
      *(_DWORD *)(v2 + 320) = 0;
      v74 = (v31 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v2 + 304) = v74;
      v75 = v31 & 0x3F;
      if ( !v75 )
        goto LABEL_104;
      v118 = *(_QWORD *)(v2 + 296);
      if ( v74 >= v118 )
        goto LABEL_188;
      v119 = (unsigned __int64 *)(v74 + *(_QWORD *)(v2 + 288));
      if ( v118 < v74 + 8 )
      {
        *(_QWORD *)(v2 + 312) = 0;
        v145 = v118 - v74;
        if ( !v145 )
          goto LABEL_188;
        v146 = 0;
        v120 = 0;
        do
        {
          v147 = *((unsigned __int8 *)v119 + v146);
          v148 = 8 * v146++;
          v120 |= v147 << v148;
          *(_QWORD *)(v2 + 312) = v120;
        }
        while ( v145 != v146 );
        v149 = v145 + v74;
        v121 = 8 * v145;
        *(_QWORD *)(v2 + 304) = v149;
        *(_DWORD *)(v2 + 320) = v121;
        if ( v75 > v121 )
          goto LABEL_188;
      }
      else
      {
        v120 = *v119;
        *(_QWORD *)(v2 + 304) = v74 + 8;
        v121 = 64;
      }
      *(_DWORD *)(v2 + 320) = v121 - v75;
      *(_QWORD *)(v2 + 312) = v120 >> v75;
LABEL_104:
      LODWORD(v195) = 0;
      sub_1510D70(m128i_i64, v30, (__int64)&v194, 0);
      if ( (v195 & 1) == 0 )
        goto LABEL_210;
      v76 = *(_QWORD **)(v2 + 224);
      if ( (unsigned int)*(_QWORD *)v194 >= -1431655765 * (unsigned int)((__int64)(v76[1] - *v76) >> 3) )
      {
        v193 = 1;
        v37 = "Invalid record";
        goto LABEL_57;
      }
      v77 = *(_QWORD *)(*v76 + 24LL * (unsigned int)*(_QWORD *)v194 + 16);
      v78 = *(_BYTE *)(v77 + 16);
      if ( v78 == 3 || !v78 )
      {
        sub_1518010(&v190, v2, v77, (__int64)(v194 + 2), v195 - 1);
        v79 = v190 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v190 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          a1[8] |= 3u;
          *(_QWORD *)a1 = v79;
          goto LABEL_43;
        }
      }
      goto LABEL_39;
    case 0x26u:
      *(_DWORD *)(v2 + 320) = 0;
      v57 = (v31 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v2 + 304) = v57;
      v58 = v31 & 0x3F;
      if ( !v58 )
        goto LABEL_92;
      v134 = *(_QWORD *)(v2 + 296);
      if ( v57 >= v134 )
        goto LABEL_188;
      v135 = (unsigned __int64 *)(v57 + *(_QWORD *)(v2 + 288));
      if ( v134 < v57 + 8 )
      {
        *(_QWORD *)(v2 + 312) = 0;
        v155 = v134 - v57;
        if ( !v155 )
          goto LABEL_188;
        v156 = 0;
        v136 = 0;
        do
        {
          v157 = *((unsigned __int8 *)v135 + v156);
          v158 = 8 * v156++;
          v136 |= v157 << v158;
          *(_QWORD *)(v2 + 312) = v136;
        }
        while ( v155 != v156 );
        v159 = v155 + v57;
        v137 = 8 * v155;
        *(_QWORD *)(v2 + 304) = v159;
        *(_DWORD *)(v2 + 320) = v137;
        if ( v58 > v137 )
          goto LABEL_188;
      }
      else
      {
        v136 = *v135;
        *(_QWORD *)(v2 + 304) = v57 + 8;
        v137 = 64;
      }
      *(_DWORD *)(v2 + 320) = v137 - v58;
      *(_QWORD *)(v2 + 312) = v136 >> v58;
LABEL_92:
      LODWORD(v195) = 0;
      sub_1510D70(m128i_i64, v30, (__int64)&v194, 0);
      if ( (_DWORD)v195 == 2 )
      {
        v59 = (unsigned __int8 *)(8LL * *(_QWORD *)(v2 + 304) - *(unsigned int *)(v2 + 320));
        v60 = *(_QWORD *)v194 + (*((_QWORD *)v194 + 1) << 32);
        *(_DWORD *)(v2 + 320) = 0;
        v61 = (unsigned __int64)&v59[v60];
        v62 = (v61 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v2 + 304) = v62;
        v63 = v61 & 0x3F;
        if ( v63 )
        {
          v122 = *(_QWORD *)(v2 + 296);
          if ( v62 >= v122 )
            goto LABEL_188;
          v123 = (unsigned __int64 *)(v62 + *(_QWORD *)(v2 + 288));
          if ( v122 < v62 + 8 )
          {
            *(_QWORD *)(v2 + 312) = 0;
            v160 = v122 - v62;
            if ( !v160 )
              goto LABEL_188;
            v161 = 0;
            v124 = 0;
            do
            {
              v162 = *((unsigned __int8 *)v123 + v161);
              v163 = 8 * v161++;
              v124 |= v162 << v163;
              *(_QWORD *)(v2 + 312) = v124;
            }
            while ( v160 != v161 );
            v164 = v160 + v62;
            v125 = 8 * v160;
            *(_QWORD *)(v2 + 304) = v164;
            *(_DWORD *)(v2 + 320) = v125;
            if ( v63 > v125 )
              goto LABEL_188;
          }
          else
          {
            v124 = *v123;
            *(_QWORD *)(v2 + 304) = v62 + 8;
            v125 = 64;
          }
          *(_DWORD *)(v2 + 320) = v125 - v63;
          *(_QWORD *)(v2 + 312) = v124 >> v63;
        }
        v64 = sub_14ED070(m128i_i64, 1);
        LODWORD(v195) = 0;
        sub_1510D70(m128i_i64, SHIDWORD(v64), (__int64)&v194, 0);
        v65 = *(const void **)(v2 + 656);
        v66 = *(_QWORD *)(v2 + 672);
        v190 = (__int64)v59;
        v67 = (unsigned int)v195;
        if ( (unsigned int)v195 > (unsigned __int64)((v66 - (__int64)v65) >> 3) )
        {
          v179 = 8LL * (unsigned int)v195;
          v185 = *(_QWORD *)(v2 + 664) - (_QWORD)v65;
          if ( (_DWORD)v195 )
          {
            v115 = sub_22077B0(8LL * (unsigned int)v195);
            v65 = *(const void **)(v2 + 656);
            v116 = (char *)v115;
            v117 = *(_QWORD *)(v2 + 664) - (_QWORD)v65;
          }
          else
          {
            v117 = *(_QWORD *)(v2 + 664) - (_QWORD)v65;
            v116 = 0;
          }
          if ( v117 > 0 )
          {
            v173 = v65;
            memmove(v116, v65, v117);
            v65 = v173;
            v144 = *(_QWORD *)(v2 + 672) - (_QWORD)v173;
          }
          else
          {
            if ( !v65 )
            {
LABEL_154:
              *(_QWORD *)(v2 + 656) = v116;
              v67 = (unsigned int)v195;
              *(_QWORD *)(v2 + 664) = &v116[v185];
              *(_QWORD *)(v2 + 672) = &v116[v179];
              goto LABEL_95;
            }
            v144 = *(_QWORD *)(v2 + 672) - (_QWORD)v65;
          }
          j_j___libc_free_0(v65, v144);
          goto LABEL_154;
        }
LABEL_95:
        v68 = v194;
        v69 = &v194[2 * v67];
        v70 = &v190;
        if ( v194 != v69 )
        {
          v71 = v69;
          do
          {
            v73 = (unsigned __int8 *)(*(_QWORD *)v68 + v190);
            v190 = (__int64)v73;
            v72 = *(_BYTE **)(v2 + 664);
            if ( v72 == *(_BYTE **)(v2 + 672) )
            {
              v183 = v70;
              sub_9CA200(v2 + 656, v72, v70);
              v70 = v183;
            }
            else
            {
              if ( v72 )
              {
                *(_QWORD *)v72 = v73;
                v72 = *(_BYTE **)(v2 + 664);
              }
              *(_QWORD *)(v2 + 664) = v72 + 8;
            }
            v68 += 2;
          }
          while ( v71 != v68 );
        }
        goto LABEL_39;
      }
LABEL_210:
      v193 = 1;
      v56 = "Invalid record";
LABEL_90:
      v190 = (__int64)v56;
      v192 = 3;
      sub_1514BE0(&v189, (__int64)&v190);
      a1[8] |= 3u;
      *(_QWORD *)a1 = v189 & 0xFFFFFFFFFFFFFFFELL;
LABEL_43:
      v28 = (unsigned __int64)v194;
      if ( v194 != v25 )
LABEL_44:
        _libc_free(v28);
      return a1;
    case 0x27u:
      v193 = 1;
      v56 = "Corrupted Metadata block";
      goto LABEL_90;
    default:
      goto LABEL_39;
  }
}
