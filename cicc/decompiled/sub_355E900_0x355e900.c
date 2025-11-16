// Function: sub_355E900
// Address: 0x355e900
//
__int64 __fastcall sub_355E900(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  int v3; // ecx
  int v4; // ebx
  int v5; // esi
  int v6; // r12d
  unsigned int v8; // esi
  int v9; // ebx
  __int64 v10; // r8
  int v11; // r11d
  unsigned int v12; // edi
  _DWORD *v13; // rdx
  _DWORD *v14; // rax
  int v15; // ecx
  _QWORD *v16; // rbx
  __int64 v17; // r15
  __int64 v18; // r13
  int v19; // r8d
  unsigned int v20; // ecx
  int *v21; // rdi
  _DWORD *v22; // rax
  int v23; // edx
  unsigned __int64 *v24; // r12
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // r11
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r11
  _DWORD *v33; // rcx
  __int64 v34; // rbx
  _DWORD *i; // rax
  __int64 v36; // r12
  __int64 v37; // r13
  int v38; // eax
  int v39; // edx
  int v40; // edx
  __int64 v41; // rdi
  int v42; // r10d
  unsigned int v43; // ecx
  int *v44; // r8
  int *v45; // rbx
  int v46; // esi
  __int64 v47; // r9
  __int64 v48; // r8
  __int64 v49; // rdi
  __int64 v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // r15
  __int64 v55; // r11
  __int64 v56; // r10
  unsigned __int64 *v57; // rdi
  int v58; // esi
  int v59; // edx
  int v60; // esi
  __int64 v61; // r8
  int v62; // edi
  int v63; // r11d
  _DWORD *v64; // r10
  int v65; // ebx
  int v66; // eax
  __int64 v67; // rdi
  int v68; // ecx
  unsigned int v69; // eax
  unsigned __int64 *v70; // rbx
  int v71; // edx
  __int64 *v72; // r12
  __int64 *j; // rbx
  __int64 v74; // rsi
  __int64 result; // rax
  int v76; // ebx
  int *v77; // r13
  __int64 v78; // r10
  __int64 v79; // r9
  __int64 v80; // r8
  __int64 v81; // r12
  __m128i v82; // rdi
  __int64 v83; // rcx
  __int64 v84; // rdx
  unsigned __int64 *v85; // r14
  __int64 v86; // rbx
  __int64 v87; // rax
  unsigned __int64 v88; // rbx
  unsigned __int64 v89; // rdi
  int v90; // eax
  int v91; // eax
  int v92; // ecx
  int v93; // ecx
  __int64 v94; // rdi
  _DWORD *v95; // r9
  int v96; // esi
  int v97; // ebx
  int v98; // r8d
  __int64 v99; // rax
  _DWORD *v100; // rdx
  _DWORD *v101; // rax
  int v102; // ecx
  int v103; // ecx
  __int64 v104; // r8
  unsigned int v105; // esi
  int v106; // edi
  int v107; // r9d
  _DWORD *v108; // r11
  int v109; // ecx
  int v110; // ecx
  __int64 v111; // rdi
  int v112; // r8d
  _DWORD *v113; // r10
  unsigned int v114; // r12d
  int v115; // esi
  int v116; // esi
  int v117; // r9d
  unsigned int v118; // ebx
  _DWORD *v119; // r11
  __int64 v121; // [rsp+8h] [rbp-E8h]
  __int64 v122; // [rsp+10h] [rbp-E0h]
  __int64 v123; // [rsp+18h] [rbp-D8h]
  int v124; // [rsp+24h] [rbp-CCh]
  int v125; // [rsp+30h] [rbp-C0h]
  __int64 v126; // [rsp+38h] [rbp-B8h]
  __int64 v127; // [rsp+38h] [rbp-B8h]
  int k; // [rsp+38h] [rbp-B8h]
  int v129; // [rsp+40h] [rbp-B0h]
  __int64 v130; // [rsp+40h] [rbp-B0h]
  __int64 v131; // [rsp+48h] [rbp-A8h]
  __int64 v132; // [rsp+48h] [rbp-A8h]
  __int64 v133; // [rsp+50h] [rbp-A0h]
  __int64 v134; // [rsp+50h] [rbp-A0h]
  int v135; // [rsp+58h] [rbp-98h]
  int v136; // [rsp+6Ch] [rbp-84h] BYREF
  __m128i v137; // [rsp+70h] [rbp-80h] BYREF
  __m128i v138; // [rsp+80h] [rbp-70h] BYREF
  __m128i v139; // [rsp+90h] [rbp-60h] BYREF
  __m128i v140; // [rsp+A0h] [rbp-50h] BYREF
  __m128i v141; // [rsp+B0h] [rbp-40h] BYREF

  v2 = a1;
  v3 = *(_DWORD *)(a1 + 80);
  v4 = *(_DWORD *)(a1 + 88);
  v5 = *(_DWORD *)(a1 + 84);
  v6 = v3 + v4;
  if ( v3 < v3 + v4 )
  {
    v135 = *(_DWORD *)(a1 + 80);
    while ( 1 )
    {
      v124 = (v5 - v3) / v4;
      if ( v124 > 0 )
        break;
LABEL_60:
      ++v135;
      v6 = v3 + v4;
      if ( v135 >= v3 + v4 )
      {
        v2 = a1;
        goto LABEL_62;
      }
    }
    v125 = 1;
    v129 = 37 * v135;
    while ( 1 )
    {
      v8 = *(_DWORD *)(a1 + 24);
      v9 = v135 + v125 * v4;
      if ( v8 )
      {
        v10 = *(_QWORD *)(a1 + 8);
        v11 = 1;
        v12 = (v8 - 1) & (37 * v9);
        v13 = 0;
        v14 = (_DWORD *)(v10 + 88LL * v12);
        v15 = *v14;
        if ( v9 == *v14 )
        {
LABEL_7:
          v16 = v14 + 2;
          goto LABEL_8;
        }
        while ( v15 != 0x7FFFFFFF )
        {
          if ( v15 == 0x80000000 && !v13 )
            v13 = v14;
          v12 = (v8 - 1) & (v11 + v12);
          v14 = (_DWORD *)(v10 + 88LL * v12);
          v15 = *v14;
          if ( v9 == *v14 )
            goto LABEL_7;
          ++v11;
        }
        if ( !v13 )
          v13 = v14;
        v90 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v91 = v90 + 1;
        if ( 4 * v91 < 3 * v8 )
        {
          if ( v8 - *(_DWORD *)(a1 + 20) - v91 <= v8 >> 3 )
          {
            sub_354BAE0(a1, v8);
            v109 = *(_DWORD *)(a1 + 24);
            if ( !v109 )
            {
LABEL_147:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v110 = v109 - 1;
            v111 = *(_QWORD *)(a1 + 8);
            v112 = 1;
            v113 = 0;
            v114 = v110 & (37 * v9);
            v13 = (_DWORD *)(v111 + 88LL * v114);
            v115 = *v13;
            v91 = *(_DWORD *)(a1 + 16) + 1;
            if ( v9 != *v13 )
            {
              while ( v115 != 0x7FFFFFFF )
              {
                if ( v115 == 0x80000000 && !v113 )
                  v113 = v13;
                v114 = v110 & (v112 + v114);
                v13 = (_DWORD *)(v111 + 88LL * v114);
                v115 = *v13;
                if ( v9 == *v13 )
                  goto LABEL_84;
                ++v112;
              }
              if ( v113 )
                v13 = v113;
            }
          }
          goto LABEL_84;
        }
      }
      else
      {
        ++*(_QWORD *)a1;
      }
      sub_354BAE0(a1, 2 * v8);
      v102 = *(_DWORD *)(a1 + 24);
      if ( !v102 )
        goto LABEL_147;
      v103 = v102 - 1;
      v104 = *(_QWORD *)(a1 + 8);
      v105 = v103 & (37 * v9);
      v13 = (_DWORD *)(v104 + 88LL * v105);
      v106 = *v13;
      v91 = *(_DWORD *)(a1 + 16) + 1;
      if ( v9 != *v13 )
      {
        v107 = 1;
        v108 = 0;
        while ( v106 != 0x7FFFFFFF )
        {
          if ( !v108 && v106 == 0x80000000 )
            v108 = v13;
          v105 = v103 & (v107 + v105);
          v13 = (_DWORD *)(v104 + 88LL * v105);
          v106 = *v13;
          if ( v9 == *v13 )
            goto LABEL_84;
          ++v107;
        }
        if ( v108 )
          v13 = v108;
      }
LABEL_84:
      *(_DWORD *)(a1 + 16) = v91;
      if ( *v13 != 0x7FFFFFFF )
        --*(_DWORD *)(a1 + 20);
      *v13 = v9;
      v16 = v13 + 2;
      *((_QWORD *)v13 + 1) = 0;
      *((_QWORD *)v13 + 2) = 0;
      *((_QWORD *)v13 + 3) = 0;
      *((_QWORD *)v13 + 4) = 0;
      *((_QWORD *)v13 + 5) = 0;
      *((_QWORD *)v13 + 6) = 0;
      *((_QWORD *)v13 + 7) = 0;
      *((_QWORD *)v13 + 8) = 0;
      *((_QWORD *)v13 + 9) = 0;
      *((_QWORD *)v13 + 10) = 0;
      sub_3547BF0((__int64 *)v13 + 1, 0);
LABEL_8:
      v17 = v16[6];
      v18 = v16[7];
      v131 = v16[9];
      v133 = v16[2];
      if ( v133 != v17 )
      {
        while ( 1 )
        {
          v26 = v17;
          if ( v17 == v18 )
            v26 = *(_QWORD *)(v131 - 8) + 512LL;
          v27 = *(unsigned int *)(a1 + 24);
          v28 = *(_QWORD *)(a1 + 8);
          v137.m128i_i64[0] = *(_QWORD *)(v26 - 8);
          if ( !(_DWORD)v27 )
            break;
          v19 = 1;
          v20 = (v27 - 1) & v129;
          v21 = (int *)(v28 + 88LL * v20);
          v22 = 0;
          v23 = *v21;
          if ( v135 == *v21 )
          {
LABEL_11:
            v24 = (unsigned __int64 *)(v21 + 2);
            v25 = *((_QWORD *)v21 + 3);
            if ( v25 == *((_QWORD *)v21 + 4) )
              goto LABEL_56;
LABEL_12:
            *(_QWORD *)(v25 - 8) = v137.m128i_i64[0];
            v24[2] -= 8LL;
            if ( v17 != v18 )
              goto LABEL_13;
LABEL_57:
            v18 = *(_QWORD *)(v131 - 8);
            v131 -= 8;
            v17 = v18 + 504;
            if ( v133 == v18 + 504 )
              goto LABEL_58;
          }
          else
          {
            while ( v23 != 0x7FFFFFFF )
            {
              if ( !v22 && v23 == 0x80000000 )
                v22 = v21;
              v20 = (v27 - 1) & (v19 + v20);
              v21 = (int *)(v28 + 88LL * v20);
              v23 = *v21;
              if ( v135 == *v21 )
                goto LABEL_11;
              ++v19;
            }
            v65 = *(_DWORD *)(a1 + 16);
            if ( !v22 )
              v22 = v21;
            ++*(_QWORD *)a1;
            v59 = v65 + 1;
            if ( 4 * (v65 + 1) >= (unsigned int)(3 * v27) )
              goto LABEL_18;
            if ( (int)v27 - *(_DWORD *)(a1 + 20) - v59 <= (unsigned int)v27 >> 3 )
            {
              sub_354BAE0(a1, v27);
              v92 = *(_DWORD *)(a1 + 24);
              if ( !v92 )
                goto LABEL_149;
              v93 = v92 - 1;
              v94 = *(_QWORD *)(a1 + 8);
              v95 = 0;
              v96 = v93 & v129;
              v59 = *(_DWORD *)(a1 + 16) + 1;
              v97 = 1;
              v22 = (_DWORD *)(v94 + 88LL * (v93 & (unsigned int)v129));
              v98 = *v22;
              if ( v135 != *v22 )
              {
                while ( v98 != 0x7FFFFFFF )
                {
                  if ( v98 != 0x80000000 || v95 )
                    v22 = v95;
                  v117 = v97 + 1;
                  v118 = v93 & (v96 + v97);
                  v96 = v118;
                  v119 = (_DWORD *)(v94 + 88LL * v118);
                  v98 = *v119;
                  if ( v135 == *v119 )
                  {
                    v22 = (_DWORD *)(v94 + 88LL * v118);
                    goto LABEL_53;
                  }
                  v97 = v117;
                  v95 = v22;
                  v22 = v119;
                }
                if ( v95 )
                  v22 = v95;
              }
            }
LABEL_53:
            *(_DWORD *)(a1 + 16) = v59;
            if ( *v22 != 0x7FFFFFFF )
              --*(_DWORD *)(a1 + 20);
            *((_QWORD *)v22 + 1) = 0;
            v24 = (unsigned __int64 *)(v22 + 2);
            *((_QWORD *)v22 + 2) = 0;
            *v22 = v135;
            *((_QWORD *)v22 + 3) = 0;
            *((_QWORD *)v22 + 4) = 0;
            *((_QWORD *)v22 + 5) = 0;
            *((_QWORD *)v22 + 6) = 0;
            *((_QWORD *)v22 + 7) = 0;
            *((_QWORD *)v22 + 8) = 0;
            *((_QWORD *)v22 + 9) = 0;
            *((_QWORD *)v22 + 10) = 0;
            sub_3547BF0((__int64 *)v22 + 1, 0);
            v25 = v24[2];
            if ( v25 != v24[3] )
              goto LABEL_12;
LABEL_56:
            sub_354AFF0(v24, &v137);
            if ( v17 == v18 )
              goto LABEL_57;
LABEL_13:
            v17 -= 8;
            if ( v133 == v17 )
              goto LABEL_58;
          }
        }
        ++*(_QWORD *)a1;
LABEL_18:
        v126 = v28;
        v29 = ((((((((unsigned int)(2 * v27 - 1) | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v27 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 4)
               | (((unsigned int)(2 * v27 - 1) | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v27 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 8)
             | (((((unsigned int)(2 * v27 - 1) | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v27 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 4)
             | (((unsigned int)(2 * v27 - 1) | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v27 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 16;
        v30 = (v29
             | (((((((unsigned int)(2 * v27 - 1) | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v27 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 4)
               | (((unsigned int)(2 * v27 - 1) | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v27 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 8)
             | (((((unsigned int)(2 * v27 - 1) | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v27 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 4)
             | (((unsigned int)(2 * v27 - 1) | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v27 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v27 - 1) >> 1))
            + 1;
        if ( (unsigned int)v30 < 0x40 )
          LODWORD(v30) = 64;
        *(_DWORD *)(a1 + 24) = v30;
        v31 = sub_C7D670(88LL * (unsigned int)v30, 8);
        v32 = v126;
        *(_QWORD *)(a1 + 8) = v31;
        v33 = (_DWORD *)v31;
        if ( v126 )
        {
          *(_QWORD *)(a1 + 16) = 0;
          v127 = 88 * v27;
          v34 = v32 + 88 * v27;
          for ( i = (_DWORD *)(v31 + 88LL * *(unsigned int *)(a1 + 24)); i != v33; v33 += 22 )
          {
            if ( v33 )
              *v33 = 0x7FFFFFFF;
          }
          v36 = v32;
          if ( v32 != v34 )
          {
            v121 = v18;
            v37 = v34;
            v123 = v17;
            v122 = v32;
            do
            {
              while ( 1 )
              {
                v38 = *(_DWORD *)v36;
                if ( (unsigned int)(*(_DWORD *)v36 + 0x7FFFFFFF) <= 0xFFFFFFFD )
                  break;
                v36 += 88;
                if ( v37 == v36 )
                  goto LABEL_34;
              }
              v39 = *(_DWORD *)(a1 + 24);
              if ( !v39 )
              {
                MEMORY[0] = 0;
                BUG();
              }
              v40 = v39 - 1;
              v41 = *(_QWORD *)(a1 + 8);
              v42 = 1;
              v43 = v40 & (37 * v38);
              v44 = 0;
              v45 = (int *)(v41 + 88LL * v43);
              v46 = *v45;
              if ( *v45 != v38 )
              {
                while ( v46 != 0x7FFFFFFF )
                {
                  if ( !v44 && v46 == 0x80000000 )
                    v44 = v45;
                  v43 = v40 & (v42 + v43);
                  v45 = (int *)(v41 + 88LL * v43);
                  v46 = *v45;
                  if ( v38 == *v45 )
                    goto LABEL_31;
                  ++v42;
                }
                if ( v44 )
                  v45 = v44;
              }
LABEL_31:
              *v45 = v38;
              *((_QWORD *)v45 + 1) = 0;
              *((_QWORD *)v45 + 2) = 0;
              *((_QWORD *)v45 + 3) = 0;
              *((_QWORD *)v45 + 4) = 0;
              *((_QWORD *)v45 + 5) = 0;
              *((_QWORD *)v45 + 6) = 0;
              *((_QWORD *)v45 + 7) = 0;
              *((_QWORD *)v45 + 8) = 0;
              *((_QWORD *)v45 + 9) = 0;
              *((_QWORD *)v45 + 10) = 0;
              sub_3547BF0((__int64 *)v45 + 1, 0);
              if ( *(_QWORD *)(v36 + 8) )
              {
                v47 = *((_QWORD *)v45 + 4);
                v48 = *((_QWORD *)v45 + 5);
                *((_QWORD *)v45 + 4) = 0;
                v49 = *((_QWORD *)v45 + 6);
                v50 = *((_QWORD *)v45 + 7);
                *((_QWORD *)v45 + 5) = 0;
                v51 = *((_QWORD *)v45 + 8);
                v52 = *((_QWORD *)v45 + 9);
                *((_QWORD *)v45 + 6) = 0;
                v53 = *((_QWORD *)v45 + 10);
                v54 = *((_QWORD *)v45 + 1);
                *((_QWORD *)v45 + 7) = 0;
                *((_QWORD *)v45 + 1) = 0;
                v55 = *((_QWORD *)v45 + 2);
                *((_QWORD *)v45 + 8) = 0;
                v56 = *((_QWORD *)v45 + 3);
                *((_QWORD *)v45 + 2) = 0;
                *((_QWORD *)v45 + 3) = 0;
                *((_QWORD *)v45 + 9) = 0;
                *((_QWORD *)v45 + 10) = 0;
                *(__m128i *)(v45 + 2) = _mm_loadu_si128((const __m128i *)(v36 + 8));
                *(__m128i *)(v45 + 6) = _mm_loadu_si128((const __m128i *)(v36 + 24));
                *(__m128i *)(v45 + 10) = _mm_loadu_si128((const __m128i *)(v36 + 40));
                *(__m128i *)(v45 + 14) = _mm_loadu_si128((const __m128i *)(v36 + 56));
                *(__m128i *)(v45 + 18) = _mm_loadu_si128((const __m128i *)(v36 + 72));
                *(_QWORD *)(v36 + 8) = v54;
                *(_QWORD *)(v36 + 16) = v55;
                *(_QWORD *)(v36 + 24) = v56;
                *(_QWORD *)(v36 + 32) = v47;
                *(_QWORD *)(v36 + 40) = v48;
                *(_QWORD *)(v36 + 48) = v49;
                *(_QWORD *)(v36 + 56) = v50;
                *(_QWORD *)(v36 + 64) = v51;
                *(_QWORD *)(v36 + 72) = v52;
                *(_QWORD *)(v36 + 80) = v53;
              }
              ++*(_DWORD *)(a1 + 16);
              v57 = (unsigned __int64 *)(v36 + 8);
              v36 += 88;
              sub_3546C50(v57);
            }
            while ( v37 != v36 );
LABEL_34:
            v17 = v123;
            v32 = v122;
            v18 = v121;
          }
          sub_C7D6A0(v32, v127, 8);
          v33 = *(_DWORD **)(a1 + 8);
          v58 = *(_DWORD *)(a1 + 24);
          v59 = *(_DWORD *)(a1 + 16) + 1;
        }
        else
        {
          v99 = *(unsigned int *)(a1 + 24);
          *(_QWORD *)(a1 + 16) = 0;
          v58 = v99;
          v100 = &v33[22 * v99];
          if ( v33 != v100 )
          {
            v101 = v33;
            do
            {
              if ( v101 )
                *v101 = 0x7FFFFFFF;
              v101 += 22;
            }
            while ( v100 != v101 );
          }
          v59 = 1;
        }
        if ( !v58 )
        {
LABEL_149:
          ++*(_DWORD *)(a1 + 16);
          BUG();
        }
        v60 = v58 - 1;
        LODWORD(v61) = v60 & v129;
        v22 = &v33[22 * (v60 & v129)];
        v62 = *v22;
        if ( *v22 != v135 )
        {
          v63 = 1;
          v64 = 0;
          while ( v62 != 0x7FFFFFFF )
          {
            if ( !v64 && v62 == 0x80000000 )
              v64 = v22;
            v61 = v60 & (unsigned int)(v61 + v63);
            v22 = &v33[22 * v61];
            v62 = *v22;
            if ( v135 == *v22 )
              goto LABEL_53;
            ++v63;
          }
          if ( v64 )
            v22 = v64;
        }
        goto LABEL_53;
      }
LABEL_58:
      ++v125;
      v4 = *(_DWORD *)(a1 + 88);
      if ( v124 < v125 )
      {
        v5 = *(_DWORD *)(a1 + 84);
        v3 = *(_DWORD *)(a1 + 80);
        goto LABEL_60;
      }
    }
  }
LABEL_62:
  if ( v6 <= v5 )
  {
    do
    {
      v66 = *(_DWORD *)(v2 + 24);
      v67 = *(_QWORD *)(v2 + 8);
      if ( v66 )
      {
        v68 = v66 - 1;
        v69 = (v66 - 1) & (37 * v6);
        v70 = (unsigned __int64 *)(v67 + 88LL * v69);
        v71 = *(_DWORD *)v70;
        if ( *(_DWORD *)v70 == v6 )
        {
LABEL_65:
          sub_3546C50(v70 + 1);
          *(_DWORD *)v70 = 0x80000000;
          --*(_DWORD *)(v2 + 16);
          ++*(_DWORD *)(v2 + 20);
        }
        else
        {
          v116 = 1;
          while ( v71 != 0x7FFFFFFF )
          {
            v69 = v68 & (v116 + v69);
            v70 = (unsigned __int64 *)(v67 + 88LL * v69);
            v71 = *(_DWORD *)v70;
            if ( *(_DWORD *)v70 == v6 )
              goto LABEL_65;
            ++v116;
          }
        }
      }
      ++v6;
    }
    while ( *(_DWORD *)(v2 + 84) >= v6 );
  }
  v72 = *(__int64 **)(a2 + 56);
  for ( j = *(__int64 **)(a2 + 48); v72 != j; j += 32 )
  {
    v74 = *j;
    sub_354FCF0(a2, v74, v2);
  }
  result = *(unsigned int *)(v2 + 80);
  v76 = result + *(_DWORD *)(v2 + 88);
  v136 = *(_DWORD *)(v2 + 80);
  for ( k = v76; (int)result < k; v136 = result )
  {
    v77 = sub_354BE50(v2, &v136);
    sub_355D5C0(v137.m128i_i64, v2, a2, v77);
    v78 = *((_QWORD *)v77 + 2);
    v79 = *((_QWORD *)v77 + 3);
    *((_QWORD *)v77 + 2) = 0;
    v80 = *((_QWORD *)v77 + 4);
    v81 = *((_QWORD *)v77 + 5);
    *((_QWORD *)v77 + 3) = 0;
    v82 = *(__m128i *)v77;
    *(_QWORD *)v77 = 0;
    v83 = *((_QWORD *)v77 + 6);
    v84 = *((_QWORD *)v77 + 7);
    *((_QWORD *)v77 + 1) = 0;
    v85 = (unsigned __int64 *)(v81 + 8);
    *((_QWORD *)v77 + 4) = 0;
    v86 = *((_QWORD *)v77 + 9);
    *((_QWORD *)v77 + 5) = 0;
    v87 = *((_QWORD *)v77 + 8);
    *((_QWORD *)v77 + 6) = 0;
    *((_QWORD *)v77 + 7) = 0;
    *((_QWORD *)v77 + 8) = 0;
    *((_QWORD *)v77 + 9) = 0;
    v134 = v78;
    *(__m128i *)v77 = _mm_loadu_si128(&v137);
    v132 = v79;
    *((__m128i *)v77 + 1) = _mm_loadu_si128(&v138);
    v130 = v80;
    *((__m128i *)v77 + 2) = _mm_loadu_si128(&v139);
    *((__m128i *)v77 + 3) = _mm_loadu_si128(&v140);
    *((__m128i *)v77 + 4) = _mm_loadu_si128(&v141);
    v141.m128i_i64[1] = v86;
    v88 = v86 + 8;
    v137 = v82;
    v138.m128i_i64[0] = v78;
    v138.m128i_i64[1] = v79;
    v139.m128i_i64[0] = v80;
    v139.m128i_i64[1] = v81;
    v140.m128i_i64[0] = v83;
    v140.m128i_i64[1] = v84;
    v141.m128i_i64[0] = v87;
    if ( v88 > v81 + 8 )
    {
      do
      {
        v89 = *v85++;
        j_j___libc_free_0(v89);
      }
      while ( v88 > (unsigned __int64)v85 );
    }
    v141.m128i_i64[1] = v81;
    v140.m128i_i64[0] = v134;
    v140.m128i_i64[1] = v132;
    v141.m128i_i64[0] = v130;
    sub_3546C50((unsigned __int64 *)&v137);
    sub_35503A0(a2, v77);
    result = (unsigned int)(v136 + 1);
  }
  return result;
}
