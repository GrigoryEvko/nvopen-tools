// Function: sub_355D7E0
// Address: 0x355d7e0
//
__int64 __fastcall sub_355D7E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // ebx
  int v5; // ecx
  int v6; // r15d
  int v7; // ebx
  unsigned int v8; // esi
  __int64 v9; // r8
  int v10; // r11d
  unsigned int v11; // edi
  int *v12; // rdx
  _DWORD *v13; // rax
  int v14; // ecx
  _QWORD *v15; // rbx
  __int64 v16; // r14
  __int64 v17; // r15
  int v18; // r8d
  unsigned int v19; // ecx
  _DWORD *v20; // rdi
  int *v21; // rax
  int v22; // edx
  unsigned __int64 *v23; // r12
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // r13
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  _DWORD *v30; // rax
  _DWORD *v31; // rdx
  __int64 v32; // rbx
  _DWORD *i; // rax
  __int64 v34; // r12
  __int64 v35; // r13
  int v36; // eax
  int v37; // r9d
  unsigned int v38; // edx
  _DWORD *v39; // r8
  _DWORD *v40; // rbx
  int v41; // esi
  __int64 v42; // r9
  __int64 v43; // r8
  __int64 v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // r14
  __int64 v50; // r11
  __int64 v51; // r10
  unsigned __int64 *v52; // rdi
  unsigned int v53; // esi
  int v54; // ecx
  unsigned int v55; // esi
  unsigned int v56; // edi
  int v57; // r8d
  int v58; // r11d
  int *v59; // r9
  int v60; // eax
  __int64 v61; // r15
  __int64 v62; // r12
  int *v63; // r14
  __int64 v64; // rax
  __m128i v65; // rdi
  __int64 v66; // r10
  __int64 v67; // r11
  __int64 v68; // r9
  __int64 v69; // rbx
  __int64 v70; // rcx
  __int64 v71; // rdx
  __int64 v72; // r8
  unsigned __int64 *v73; // r12
  unsigned __int64 v74; // rbx
  unsigned __int64 v75; // rdi
  __int64 **v76; // rbx
  __int64 *v77; // r14
  _BYTE *v78; // rsi
  __int64 v79; // rax
  _QWORD *v80; // rax
  _QWORD *v81; // rsi
  __int64 v82; // rcx
  __int64 v83; // rdx
  int v84; // eax
  unsigned int v85; // esi
  __int64 v86; // rdi
  __int64 v87; // r10
  __int64 *v88; // r8
  int v89; // r14d
  unsigned __int32 v90; // ecx
  _QWORD *v91; // rdx
  __int64 v92; // r9
  int *v93; // rdx
  int v94; // edx
  __int64 v95; // rax
  _DWORD *v96; // rbx
  _DWORD *v97; // r12
  unsigned __int64 *v98; // rdi
  int v100; // eax
  int v101; // ecx
  int v102; // edx
  int v103; // r11d
  unsigned int v104; // edi
  int v105; // esi
  int v106; // esi
  __int64 v107; // r9
  int v108; // esi
  __int64 v109; // r10
  unsigned __int32 v110; // ecx
  int v111; // r14d
  __int64 *v112; // r11
  _DWORD *j; // rcx
  int v114; // ecx
  int v115; // ecx
  __int64 v116; // r8
  unsigned int v117; // esi
  int v118; // edi
  int v119; // r11d
  int *v120; // r9
  int v121; // esi
  int v122; // esi
  __int64 v123; // r10
  int v124; // r14d
  unsigned __int32 v125; // ecx
  int v126; // ecx
  int v127; // ecx
  __int64 v128; // rdi
  int v129; // r10d
  int *v130; // r8
  unsigned int v131; // r12d
  int v132; // esi
  __int64 v136; // [rsp+18h] [rbp-108h]
  __int64 v137; // [rsp+20h] [rbp-100h]
  int v138; // [rsp+28h] [rbp-F8h]
  __int64 v140; // [rsp+38h] [rbp-E8h]
  int v141; // [rsp+40h] [rbp-E0h]
  __int64 v142; // [rsp+40h] [rbp-E0h]
  int v143; // [rsp+40h] [rbp-E0h]
  int v144; // [rsp+40h] [rbp-E0h]
  __int64 v145; // [rsp+48h] [rbp-D8h]
  int v146; // [rsp+50h] [rbp-D0h]
  __int64 v147; // [rsp+50h] [rbp-D0h]
  __int64 ***v148; // [rsp+50h] [rbp-D0h]
  __int64 v149; // [rsp+58h] [rbp-C8h]
  __int64 v150; // [rsp+58h] [rbp-C8h]
  __int64 **v151; // [rsp+58h] [rbp-C8h]
  __int64 v152; // [rsp+60h] [rbp-C0h]
  __int64 v153; // [rsp+60h] [rbp-C0h]
  __int64 v154; // [rsp+60h] [rbp-C0h]
  int v155; // [rsp+68h] [rbp-B8h]
  __int64 v156; // [rsp+68h] [rbp-B8h]
  int v157; // [rsp+7Ch] [rbp-A4h] BYREF
  __int64 v158; // [rsp+80h] [rbp-A0h] BYREF
  _DWORD *v159; // [rsp+88h] [rbp-98h]
  __int64 v160; // [rsp+90h] [rbp-90h]
  unsigned int v161; // [rsp+98h] [rbp-88h]
  __m128i v162; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v163; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v164; // [rsp+C0h] [rbp-60h] BYREF
  __m128i v165; // [rsp+D0h] [rbp-50h] BYREF
  __m128i v166; // [rsp+E0h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a2 + 88);
  v5 = *(_DWORD *)(a2 + 80);
  v158 = 0;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  if ( v4 > 0 )
  {
    v6 = v5;
    while ( 1 )
    {
      v138 = (*(_DWORD *)(a2 + 84) - v5) / v4;
      if ( v138 >= 0 )
        break;
LABEL_62:
      ++v6;
      v60 = v5 + v4;
      if ( v6 >= v5 + v4 )
        goto LABEL_63;
    }
    v155 = v6;
    v141 = 0;
    v146 = 37 * v6;
    while ( 1 )
    {
      v7 = v155 + v141 * v4;
      v8 = *(_DWORD *)(a2 + 24);
      if ( v8 )
      {
        v9 = *(_QWORD *)(a2 + 8);
        v10 = 1;
        v11 = (v8 - 1) & (37 * v7);
        v12 = 0;
        v13 = (_DWORD *)(v9 + 88LL * v11);
        v14 = *v13;
        if ( v7 == *v13 )
        {
LABEL_7:
          v15 = v13 + 2;
          goto LABEL_8;
        }
        while ( v14 != 0x7FFFFFFF )
        {
          if ( !v12 && v14 == 0x80000000 )
            v12 = v13;
          v11 = (v8 - 1) & (v10 + v11);
          v13 = (_DWORD *)(v9 + 88LL * v11);
          v14 = *v13;
          if ( v7 == *v13 )
            goto LABEL_7;
          ++v10;
        }
        if ( !v12 )
          v12 = v13;
        ++*(_QWORD *)a2;
        v100 = *(_DWORD *)(a2 + 16) + 1;
        if ( 4 * v100 < 3 * v8 )
        {
          if ( v8 - *(_DWORD *)(a2 + 20) - v100 <= v8 >> 3 )
          {
            sub_354BAE0(a2, v8);
            v126 = *(_DWORD *)(a2 + 24);
            if ( !v126 )
            {
LABEL_195:
              ++*(_DWORD *)(a2 + 16);
              BUG();
            }
            v127 = v126 - 1;
            v128 = *(_QWORD *)(a2 + 8);
            v129 = 1;
            v130 = 0;
            v131 = v127 & (37 * v7);
            v12 = (int *)(v128 + 88LL * v131);
            v132 = *v12;
            v100 = *(_DWORD *)(a2 + 16) + 1;
            if ( v7 != *v12 )
            {
              while ( v132 != 0x7FFFFFFF )
              {
                if ( v132 == 0x80000000 && !v130 )
                  v130 = v12;
                v131 = v127 & (v129 + v131);
                v12 = (int *)(v128 + 88LL * v131);
                v132 = *v12;
                if ( v7 == *v12 )
                  goto LABEL_105;
                ++v129;
              }
              if ( v130 )
                v12 = v130;
            }
          }
          goto LABEL_105;
        }
      }
      else
      {
        ++*(_QWORD *)a2;
      }
      sub_354BAE0(a2, 2 * v8);
      v114 = *(_DWORD *)(a2 + 24);
      if ( !v114 )
        goto LABEL_195;
      v115 = v114 - 1;
      v116 = *(_QWORD *)(a2 + 8);
      v117 = v115 & (37 * v7);
      v12 = (int *)(v116 + 88LL * v117);
      v118 = *v12;
      v100 = *(_DWORD *)(a2 + 16) + 1;
      if ( v7 != *v12 )
      {
        v119 = 1;
        v120 = 0;
        while ( v118 != 0x7FFFFFFF )
        {
          if ( !v120 && v118 == 0x80000000 )
            v120 = v12;
          v117 = v115 & (v119 + v117);
          v12 = (int *)(v116 + 88LL * v117);
          v118 = *v12;
          if ( v7 == *v12 )
            goto LABEL_105;
          ++v119;
        }
        if ( v120 )
          v12 = v120;
      }
LABEL_105:
      *(_DWORD *)(a2 + 16) = v100;
      if ( *v12 != 0x7FFFFFFF )
        --*(_DWORD *)(a2 + 20);
      *v12 = v7;
      v15 = v12 + 2;
      *((_QWORD *)v12 + 1) = 0;
      *((_QWORD *)v12 + 2) = 0;
      *((_QWORD *)v12 + 3) = 0;
      *((_QWORD *)v12 + 4) = 0;
      *((_QWORD *)v12 + 5) = 0;
      *((_QWORD *)v12 + 6) = 0;
      *((_QWORD *)v12 + 7) = 0;
      *((_QWORD *)v12 + 8) = 0;
      *((_QWORD *)v12 + 9) = 0;
      *((_QWORD *)v12 + 10) = 0;
      sub_3547BF0((__int64 *)v12 + 1, 0);
LABEL_8:
      v16 = v15[6];
      v17 = v15[7];
      v152 = v15[2];
      v149 = v15[9];
      if ( v152 != v16 )
      {
        while ( 1 )
        {
          v25 = v16;
          if ( v16 == v17 )
            v25 = *(_QWORD *)(v149 - 8) + 512LL;
          v26 = v161;
          v27 = (__int64)v159;
          v162.m128i_i64[0] = *(_QWORD *)(v25 - 8);
          if ( !v161 )
            break;
          v18 = 1;
          v19 = (v161 - 1) & v146;
          v20 = &v159[22 * v19];
          v21 = 0;
          v22 = *v20;
          if ( v155 == *v20 )
          {
LABEL_11:
            v23 = (unsigned __int64 *)(v20 + 2);
            v24 = *((_QWORD *)v20 + 3);
            if ( v24 == *((_QWORD *)v20 + 4) )
              goto LABEL_58;
LABEL_12:
            *(_QWORD *)(v24 - 8) = v162.m128i_i64[0];
            v23[2] -= 8LL;
            if ( v16 != v17 )
              goto LABEL_13;
LABEL_59:
            v17 = *(_QWORD *)(v149 - 8);
            v149 -= 8;
            v16 = v17 + 504;
            if ( v152 == v17 + 504 )
              goto LABEL_60;
          }
          else
          {
            while ( v22 != 0x7FFFFFFF )
            {
              if ( v22 == 0x80000000 && !v21 )
                v21 = v20;
              v19 = (v161 - 1) & (v18 + v19);
              v20 = &v159[22 * v19];
              v22 = *v20;
              if ( *v20 == v155 )
                goto LABEL_11;
              ++v18;
            }
            if ( !v21 )
              v21 = v20;
            ++v158;
            v54 = v160 + 1;
            if ( 4 * ((int)v160 + 1) >= 3 * v161 )
              goto LABEL_18;
            if ( v161 - HIDWORD(v160) - v54 <= v161 >> 3 )
            {
              sub_354BAE0((__int64)&v158, v161);
              if ( !v161 )
                goto LABEL_197;
              v59 = 0;
              v103 = 1;
              v104 = (v161 - 1) & v146;
              v54 = v160 + 1;
              v21 = &v159[22 * v104];
              v105 = *v21;
              if ( v155 != *v21 )
              {
                while ( v105 != 0x7FFFFFFF )
                {
                  if ( !v59 && v105 == 0x80000000 )
                    v59 = v21;
                  v104 = (v161 - 1) & (v103 + v104);
                  v21 = &v159[22 * v104];
                  v105 = *v21;
                  if ( v155 == *v21 )
                    goto LABEL_55;
                  ++v103;
                }
                goto LABEL_133;
              }
            }
LABEL_55:
            LODWORD(v160) = v54;
            if ( *v21 != 0x7FFFFFFF )
              --HIDWORD(v160);
            *((_QWORD *)v21 + 1) = 0;
            v23 = (unsigned __int64 *)(v21 + 2);
            *((_QWORD *)v21 + 2) = 0;
            *v21 = v155;
            *((_QWORD *)v21 + 3) = 0;
            *((_QWORD *)v21 + 4) = 0;
            *((_QWORD *)v21 + 5) = 0;
            *((_QWORD *)v21 + 6) = 0;
            *((_QWORD *)v21 + 7) = 0;
            *((_QWORD *)v21 + 8) = 0;
            *((_QWORD *)v21 + 9) = 0;
            *((_QWORD *)v21 + 10) = 0;
            sub_3547BF0((__int64 *)v21 + 1, 0);
            v24 = v23[2];
            if ( v24 != v23[3] )
              goto LABEL_12;
LABEL_58:
            sub_354AFF0(v23, &v162);
            if ( v16 == v17 )
              goto LABEL_59;
LABEL_13:
            v16 -= 8;
            if ( v152 == v16 )
              goto LABEL_60;
          }
        }
        ++v158;
LABEL_18:
        v28 = ((((((((2 * v161 - 1) | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 2)
                 | (2 * v161 - 1)
                 | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 4)
               | (((2 * v161 - 1) | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 2)
               | (2 * v161 - 1)
               | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 8)
             | (((((2 * v161 - 1) | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 2)
               | (2 * v161 - 1)
               | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 4)
             | (((2 * v161 - 1) | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 2)
             | (2 * v161 - 1)
             | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 16;
        v29 = (v28
             | (((((((2 * v161 - 1) | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 2)
                 | (2 * v161 - 1)
                 | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 4)
               | (((2 * v161 - 1) | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 2)
               | (2 * v161 - 1)
               | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 8)
             | (((((2 * v161 - 1) | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 2)
               | (2 * v161 - 1)
               | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 4)
             | (((2 * v161 - 1) | ((unsigned __int64)(2 * v161 - 1) >> 1)) >> 2)
             | (2 * v161 - 1)
             | ((unsigned __int64)(2 * v161 - 1) >> 1))
            + 1;
        if ( (unsigned int)v29 < 0x40 )
          LODWORD(v29) = 64;
        v161 = v29;
        v30 = (_DWORD *)sub_C7D670(88LL * (unsigned int)v29, 8);
        v159 = v30;
        v31 = v30;
        if ( v27 )
        {
          v160 = 0;
          v145 = 88 * v26;
          v32 = v27 + 88 * v26;
          for ( i = &v30[22 * v161]; i != v31; v31 += 22 )
          {
            if ( v31 )
              *v31 = 0x7FFFFFFF;
          }
          v34 = v27;
          if ( v27 != v32 )
          {
            v136 = v27;
            v35 = v32;
            v137 = v16;
            do
            {
              while ( 1 )
              {
                v36 = *(_DWORD *)v34;
                if ( (unsigned int)(*(_DWORD *)v34 + 0x7FFFFFFF) <= 0xFFFFFFFD )
                  break;
                v34 += 88;
                if ( v35 == v34 )
                  goto LABEL_34;
              }
              if ( !v161 )
              {
                MEMORY[0] = 0;
                BUG();
              }
              v37 = 1;
              v38 = (v161 - 1) & (37 * v36);
              v39 = 0;
              v40 = &v159[22 * v38];
              v41 = *v40;
              if ( v36 != *v40 )
              {
                while ( v41 != 0x7FFFFFFF )
                {
                  if ( !v39 && v41 == 0x80000000 )
                    v39 = v40;
                  v38 = (v161 - 1) & (v37 + v38);
                  v40 = &v159[22 * v38];
                  v41 = *v40;
                  if ( v36 == *v40 )
                    goto LABEL_31;
                  ++v37;
                }
                if ( v39 )
                  v40 = v39;
              }
LABEL_31:
              *v40 = v36;
              *((_QWORD *)v40 + 1) = 0;
              *((_QWORD *)v40 + 2) = 0;
              *((_QWORD *)v40 + 3) = 0;
              *((_QWORD *)v40 + 4) = 0;
              *((_QWORD *)v40 + 5) = 0;
              *((_QWORD *)v40 + 6) = 0;
              *((_QWORD *)v40 + 7) = 0;
              *((_QWORD *)v40 + 8) = 0;
              *((_QWORD *)v40 + 9) = 0;
              *((_QWORD *)v40 + 10) = 0;
              sub_3547BF0((__int64 *)v40 + 1, 0);
              if ( *(_QWORD *)(v34 + 8) )
              {
                v42 = *((_QWORD *)v40 + 4);
                v43 = *((_QWORD *)v40 + 5);
                *((_QWORD *)v40 + 4) = 0;
                v44 = *((_QWORD *)v40 + 6);
                v45 = *((_QWORD *)v40 + 7);
                *((_QWORD *)v40 + 5) = 0;
                v46 = *((_QWORD *)v40 + 8);
                v47 = *((_QWORD *)v40 + 9);
                *((_QWORD *)v40 + 6) = 0;
                v48 = *((_QWORD *)v40 + 10);
                v49 = *((_QWORD *)v40 + 1);
                *((_QWORD *)v40 + 7) = 0;
                *((_QWORD *)v40 + 1) = 0;
                v50 = *((_QWORD *)v40 + 2);
                *((_QWORD *)v40 + 8) = 0;
                v51 = *((_QWORD *)v40 + 3);
                *((_QWORD *)v40 + 2) = 0;
                *((_QWORD *)v40 + 3) = 0;
                *((_QWORD *)v40 + 9) = 0;
                *((_QWORD *)v40 + 10) = 0;
                *(__m128i *)(v40 + 2) = _mm_loadu_si128((const __m128i *)(v34 + 8));
                *(__m128i *)(v40 + 6) = _mm_loadu_si128((const __m128i *)(v34 + 24));
                *(__m128i *)(v40 + 10) = _mm_loadu_si128((const __m128i *)(v34 + 40));
                *(__m128i *)(v40 + 14) = _mm_loadu_si128((const __m128i *)(v34 + 56));
                *(__m128i *)(v40 + 18) = _mm_loadu_si128((const __m128i *)(v34 + 72));
                *(_QWORD *)(v34 + 8) = v49;
                *(_QWORD *)(v34 + 16) = v50;
                *(_QWORD *)(v34 + 24) = v51;
                *(_QWORD *)(v34 + 32) = v42;
                *(_QWORD *)(v34 + 40) = v43;
                *(_QWORD *)(v34 + 48) = v44;
                *(_QWORD *)(v34 + 56) = v45;
                *(_QWORD *)(v34 + 64) = v46;
                *(_QWORD *)(v34 + 72) = v47;
                *(_QWORD *)(v34 + 80) = v48;
              }
              v52 = (unsigned __int64 *)(v34 + 8);
              v34 += 88;
              LODWORD(v160) = v160 + 1;
              sub_3546C50(v52);
            }
            while ( v35 != v34 );
LABEL_34:
            v16 = v137;
            v27 = v136;
          }
          sub_C7D6A0(v27, v145, 8);
          v31 = v159;
          v53 = v161;
          v54 = v160 + 1;
        }
        else
        {
          v160 = 0;
          v53 = v161;
          for ( j = &v30[22 * v161]; j != v30; v30 += 22 )
          {
            if ( v30 )
              *v30 = 0x7FFFFFFF;
          }
          v54 = 1;
        }
        if ( !v53 )
        {
LABEL_197:
          LODWORD(v160) = v160 + 1;
          BUG();
        }
        v55 = v53 - 1;
        v56 = v55 & v146;
        v21 = &v31[22 * (v55 & v146)];
        v57 = *v21;
        if ( v155 != *v21 )
        {
          v58 = 1;
          v59 = 0;
          while ( v57 != 0x7FFFFFFF )
          {
            if ( !v59 && v57 == 0x80000000 )
              v59 = v21;
            v56 = v55 & (v58 + v56);
            v21 = &v31[22 * v56];
            v57 = *v21;
            if ( *v21 == v155 )
              goto LABEL_55;
            ++v58;
          }
LABEL_133:
          if ( v59 )
            v21 = v59;
          goto LABEL_55;
        }
        goto LABEL_55;
      }
LABEL_60:
      ++v141;
      v4 = *(_DWORD *)(a2 + 88);
      if ( v138 < v141 )
      {
        v6 = v155;
        v5 = *(_DWORD *)(a2 + 80);
        goto LABEL_62;
      }
    }
  }
  v60 = v5 + v4;
LABEL_63:
  v157 = v5;
  if ( v60 > v5 )
  {
    v156 = a2 + 40;
    v61 = a2;
    v62 = a4;
    while ( 1 )
    {
      v63 = sub_354BE50((__int64)&v158, &v157);
      sub_355D5C0(v162.m128i_i64, v61, a1, v63);
      v64 = *((_QWORD *)v63 + 5);
      v65.m128i_i64[0] = *(_QWORD *)v63;
      *((_QWORD *)v63 + 5) = 0;
      v66 = *((_QWORD *)v63 + 2);
      v67 = *((_QWORD *)v63 + 3);
      *(_QWORD *)v63 = 0;
      v68 = *((_QWORD *)v63 + 4);
      v69 = *((_QWORD *)v63 + 9);
      *((_QWORD *)v63 + 2) = 0;
      v65.m128i_i64[1] = *((_QWORD *)v63 + 1);
      v70 = *((_QWORD *)v63 + 6);
      *((_QWORD *)v63 + 1) = 0;
      *((_QWORD *)v63 + 3) = 0;
      v71 = *((_QWORD *)v63 + 7);
      *((_QWORD *)v63 + 4) = 0;
      v72 = *((_QWORD *)v63 + 8);
      *((_QWORD *)v63 + 6) = 0;
      *((_QWORD *)v63 + 7) = 0;
      *((_QWORD *)v63 + 8) = 0;
      *((_QWORD *)v63 + 9) = 0;
      v153 = v66;
      *(__m128i *)v63 = _mm_loadu_si128(&v162);
      v150 = v67;
      *((__m128i *)v63 + 1) = _mm_loadu_si128(&v163);
      v147 = v68;
      *((__m128i *)v63 + 2) = _mm_loadu_si128(&v164);
      v142 = v64;
      *((__m128i *)v63 + 3) = _mm_loadu_si128(&v165);
      *((__m128i *)v63 + 4) = _mm_loadu_si128(&v166);
      v162 = v65;
      v166.m128i_i64[1] = v69;
      v163.m128i_i64[0] = v66;
      v163.m128i_i64[1] = v67;
      v164.m128i_i64[0] = v68;
      v164.m128i_i64[1] = v64;
      v165.m128i_i64[0] = v70;
      v165.m128i_i64[1] = v71;
      v166.m128i_i64[0] = v72;
      if ( v69 + 8 > (unsigned __int64)(v64 + 8) )
      {
        v140 = v62;
        v73 = (unsigned __int64 *)(v64 + 8);
        v74 = v69 + 8;
        do
        {
          v75 = *v73++;
          j_j___libc_free_0(v75);
        }
        while ( v74 > (unsigned __int64)v73 );
        v62 = v140;
      }
      v165.m128i_i64[0] = v153;
      v165.m128i_i64[1] = v150;
      v166.m128i_i64[0] = v147;
      v166.m128i_i64[1] = v142;
      sub_3546C50((unsigned __int64 *)&v162);
      v76 = (__int64 **)*((_QWORD *)v63 + 2);
      v154 = *((_QWORD *)v63 + 4);
      v148 = (__int64 ***)*((_QWORD *)v63 + 5);
      v151 = (__int64 **)*((_QWORD *)v63 + 6);
LABEL_70:
      if ( v151 != v76 )
        break;
LABEL_87:
      v94 = *(_DWORD *)(v61 + 80) + *(_DWORD *)(v61 + 88);
      if ( ++v157 >= v94 )
        goto LABEL_88;
    }
    while ( 1 )
    {
      v77 = *v76;
      v78 = *(_BYTE **)(a3 + 8);
      v79 = **v76;
      v162.m128i_i64[0] = v79;
      if ( v78 == *(_BYTE **)(a3 + 16) )
      {
        sub_2E997F0(a3, v78, &v162);
        v80 = *(_QWORD **)(v61 + 48);
        if ( !v80 )
          goto LABEL_137;
      }
      else
      {
        if ( v78 )
        {
          *(_QWORD *)v78 = v79;
          v78 = *(_BYTE **)(a3 + 8);
        }
        *(_QWORD *)(a3 + 8) = v78 + 8;
        v80 = *(_QWORD **)(v61 + 48);
        if ( !v80 )
        {
LABEL_137:
          v85 = *(_DWORD *)(v62 + 24);
          v84 = -1;
          if ( !v85 )
            goto LABEL_138;
          goto LABEL_83;
        }
      }
      v81 = (_QWORD *)v156;
      do
      {
        while ( 1 )
        {
          v82 = v80[2];
          v83 = v80[3];
          if ( v80[4] >= (unsigned __int64)v77 )
            break;
          v80 = (_QWORD *)v80[3];
          if ( !v83 )
            goto LABEL_79;
        }
        v81 = v80;
        v80 = (_QWORD *)v80[2];
      }
      while ( v82 );
LABEL_79:
      v84 = -1;
      if ( v81 != (_QWORD *)v156 && v81[4] <= (unsigned __int64)v77 )
        v84 = (*((_DWORD *)v81 + 10) - *(_DWORD *)(v61 + 80)) / *(_DWORD *)(v61 + 88);
      v85 = *(_DWORD *)(v62 + 24);
      if ( !v85 )
      {
LABEL_138:
        ++*(_QWORD *)v62;
        goto LABEL_139;
      }
LABEL_83:
      v86 = v162.m128i_i64[0];
      v87 = *(_QWORD *)(v62 + 8);
      v88 = 0;
      v89 = 1;
      v90 = (v85 - 1) & (((unsigned __int32)v162.m128i_i32[0] >> 9) ^ ((unsigned __int32)v162.m128i_i32[0] >> 4));
      v91 = (_QWORD *)(v87 + 16LL * v90);
      v92 = *v91;
      if ( *v91 != v162.m128i_i64[0] )
      {
        while ( v92 != -4096 )
        {
          if ( !v88 && v92 == -8192 )
            v88 = v91;
          v90 = (v85 - 1) & (v89 + v90);
          v91 = (_QWORD *)(v87 + 16LL * v90);
          v92 = *v91;
          if ( v162.m128i_i64[0] == *v91 )
            goto LABEL_84;
          ++v89;
        }
        v101 = *(_DWORD *)(v62 + 16);
        if ( !v88 )
          v88 = v91;
        ++*(_QWORD *)v62;
        v102 = v101 + 1;
        if ( 4 * (v101 + 1) >= 3 * v85 )
        {
LABEL_139:
          v143 = v84;
          sub_2E261E0(v62, 2 * v85);
          v106 = *(_DWORD *)(v62 + 24);
          if ( !v106 )
            goto LABEL_196;
          v107 = v162.m128i_i64[0];
          v108 = v106 - 1;
          v109 = *(_QWORD *)(v62 + 8);
          v102 = *(_DWORD *)(v62 + 16) + 1;
          v84 = v143;
          v110 = v108 & (((unsigned __int32)v162.m128i_i32[0] >> 9) ^ ((unsigned __int32)v162.m128i_i32[0] >> 4));
          v88 = (__int64 *)(v109 + 16LL * v110);
          v86 = *v88;
          if ( v162.m128i_i64[0] != *v88 )
          {
            v111 = 1;
            v112 = 0;
            while ( v86 != -4096 )
            {
              if ( v86 == -8192 && !v112 )
                v112 = v88;
              v110 = v108 & (v111 + v110);
              v88 = (__int64 *)(v109 + 16LL * v110);
              v86 = *v88;
              if ( v162.m128i_i64[0] == *v88 )
                goto LABEL_127;
              ++v111;
            }
LABEL_143:
            v86 = v107;
            if ( v112 )
              v88 = v112;
          }
        }
        else if ( v85 - *(_DWORD *)(v62 + 20) - v102 <= v85 >> 3 )
        {
          v144 = v84;
          sub_2E261E0(v62, v85);
          v121 = *(_DWORD *)(v62 + 24);
          if ( !v121 )
          {
LABEL_196:
            ++*(_DWORD *)(a4 + 16);
            BUG();
          }
          v107 = v162.m128i_i64[0];
          v122 = v121 - 1;
          v123 = *(_QWORD *)(v62 + 8);
          v112 = 0;
          v124 = 1;
          v102 = *(_DWORD *)(v62 + 16) + 1;
          v84 = v144;
          v125 = v122 & (((unsigned __int32)v162.m128i_i32[0] >> 9) ^ ((unsigned __int32)v162.m128i_i32[0] >> 4));
          v88 = (__int64 *)(v123 + 16LL * v125);
          v86 = *v88;
          if ( v162.m128i_i64[0] != *v88 )
          {
            while ( v86 != -4096 )
            {
              if ( v86 == -8192 && !v112 )
                v112 = v88;
              v125 = v122 & (v124 + v125);
              v88 = (__int64 *)(v123 + 16LL * v125);
              v86 = *v88;
              if ( v162.m128i_i64[0] == *v88 )
                goto LABEL_127;
              ++v124;
            }
            goto LABEL_143;
          }
        }
LABEL_127:
        *(_DWORD *)(v62 + 16) = v102;
        if ( *v88 != -4096 )
          --*(_DWORD *)(v62 + 20);
        *v88 = v86;
        v93 = (int *)(v88 + 1);
        *((_DWORD *)v88 + 2) = 0;
        goto LABEL_85;
      }
LABEL_84:
      v93 = (int *)(v91 + 1);
LABEL_85:
      *v93 = v84;
      if ( (__int64 **)v154 != ++v76 )
        goto LABEL_70;
      v76 = *++v148;
      v154 = (__int64)(*v148 + 64);
      if ( v151 == *v148 )
        goto LABEL_87;
    }
  }
LABEL_88:
  v95 = v161;
  if ( v161 )
  {
    v96 = v159;
    v97 = &v159[22 * v161];
    do
    {
      while ( (unsigned int)(*v96 + 0x7FFFFFFF) > 0xFFFFFFFD )
      {
        v96 += 22;
        if ( v97 == v96 )
          goto LABEL_93;
      }
      v98 = (unsigned __int64 *)(v96 + 2);
      v96 += 22;
      sub_3546C50(v98);
    }
    while ( v97 != v96 );
LABEL_93:
    v95 = v161;
  }
  return sub_C7D6A0((__int64)v159, 88 * v95, 8);
}
