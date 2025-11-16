// Function: sub_1033860
// Address: 0x1033860
//
__int64 __fastcall sub_1033860(
        __int64 a1,
        __int64 a2,
        char a3,
        _QWORD *a4,
        __int16 a5,
        __int64 a6,
        __int64 a7,
        int *a8,
        char a9,
        __int64 *a10)
{
  _QWORD *v13; // rbx
  unsigned __int8 *v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // r14
  _QWORD *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  _QWORD *v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // rdx
  int *v26; // rax
  int v27; // eax
  bool v28; // cc
  int v29; // eax
  int *v30; // rax
  unsigned __int64 v31; // rdx
  char v32; // di
  __int64 v33; // r8
  __int64 v34; // rsi
  unsigned int v35; // eax
  __int64 *v36; // rcx
  __int64 v37; // r9
  __int64 v38; // rcx
  _QWORD *v39; // rax
  __int64 v40; // r14
  char v42; // al
  unsigned __int8 *v43; // rax
  __int64 v44; // rdi
  __m128i v45; // xmm4
  __m128i v46; // xmm2
  _QWORD *v47; // rdi
  __int64 v48; // r12
  unsigned __int64 v49; // rax
  __int64 v50; // rax
  unsigned __int16 v51; // r12
  __int16 v52; // r12
  int v53; // eax
  int v54; // r11d
  __int64 v55; // r13
  int v56; // r12d
  __int64 v57; // r8
  _QWORD *v58; // rax
  unsigned int v59; // edi
  _QWORD *v60; // rdx
  __int64 v61; // rcx
  int *v62; // rax
  __int64 *v63; // rdi
  unsigned int v65; // eax
  __int64 *v66; // rdx
  int v67; // ecx
  unsigned int v68; // r9d
  _QWORD *v69; // rax
  _QWORD *v70; // rdx
  __int64 v71; // rax
  unsigned int v72; // r12d
  __int64 v73; // rax
  __int64 *v74; // r13
  __int64 *v75; // rbx
  __int64 v76; // rax
  unsigned __int16 v77; // r12
  char v78; // al
  __m128i v79; // xmm4
  __int64 v80; // r12
  __m128i v81; // xmm2
  _QWORD *v82; // rdi
  char v83; // al
  __int64 v84; // rax
  unsigned __int64 v85; // rdx
  unsigned __int8 v86; // cl
  __int64 v87; // r11
  unsigned __int8 *v88; // r11
  __m128i *v89; // rdi
  __int32 *v90; // rsi
  __int64 i; // rcx
  _QWORD *v92; // rdi
  _QWORD *v93; // rdi
  char v94; // al
  __int64 *v95; // rdx
  char v96; // al
  int v97; // r10d
  __int64 v98; // rdx
  __int64 v99; // rdi
  int v100; // ecx
  unsigned int v101; // eax
  __int64 v102; // rdi
  int v103; // ecx
  unsigned int v104; // eax
  int v105; // r9d
  __int64 *v106; // r8
  int v107; // ecx
  int v108; // ecx
  int v109; // ecx
  int v110; // ecx
  int v111; // edi
  int v112; // edi
  __int64 v113; // r8
  _QWORD *v114; // r10
  unsigned int v115; // ebx
  int v116; // edx
  int v117; // r10d
  int v118; // r10d
  __int64 v119; // r9
  unsigned int v120; // edx
  __int64 v121; // r8
  int v122; // edi
  int v123; // r9d
  __int64 v124; // [rsp+8h] [rbp-188h]
  __int64 v125; // [rsp+8h] [rbp-188h]
  int v126; // [rsp+1Ch] [rbp-174h]
  _QWORD *v127; // [rsp+30h] [rbp-160h]
  unsigned int v130; // [rsp+40h] [rbp-150h]
  unsigned __int8 v131; // [rsp+51h] [rbp-13Fh]
  char v133; // [rsp+53h] [rbp-13Dh]
  char v134; // [rsp+54h] [rbp-13Ch]
  _QWORD *v136; // [rsp+58h] [rbp-138h]
  int v137; // [rsp+58h] [rbp-138h]
  int v138; // [rsp+58h] [rbp-138h]
  int v139; // [rsp+6Ch] [rbp-124h] BYREF
  __m128i v140; // [rsp+70h] [rbp-120h] BYREF
  __int64 v141; // [rsp+80h] [rbp-110h]
  __int64 v142; // [rsp+88h] [rbp-108h]
  __int64 v143; // [rsp+90h] [rbp-100h]
  __int64 v144; // [rsp+98h] [rbp-F8h]
  __m128i v145; // [rsp+A0h] [rbp-F0h] BYREF
  __m128i v146; // [rsp+B0h] [rbp-E0h]
  __m128i v147; // [rsp+C0h] [rbp-D0h]
  char v148; // [rsp+D0h] [rbp-C0h]
  __int64 v149; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v150; // [rsp+E8h] [rbp-A8h]
  _QWORD *v151; // [rsp+F0h] [rbp-A0h]
  __int16 v152; // [rsp+F8h] [rbp-98h]
  __int64 v153; // [rsp+100h] [rbp-90h]
  __int64 *v154; // [rsp+108h] [rbp-88h]
  __int64 v155; // [rsp+110h] [rbp-80h]
  _BYTE v156[120]; // [rsp+118h] [rbp-78h] BYREF
  int *v157; // [rsp+1A8h] [rbp+18h]

  v13 = a4;
  v14 = *(unsigned __int8 **)a2;
  v15 = sub_AA4E30(a6);
  v131 = sub_BD5420(v14, v15);
  v127 = sub_C52410();
  v16 = sub_C959E0();
  v17 = (_QWORD *)v127[2];
  v18 = v127 + 1;
  if ( v17 )
  {
    v19 = v127 + 1;
    do
    {
      while ( 1 )
      {
        v20 = v17[2];
        v21 = v17[3];
        if ( v16 <= v17[4] )
          break;
        v17 = (_QWORD *)v17[3];
        if ( !v21 )
          goto LABEL_6;
      }
      v19 = v17;
      v17 = (_QWORD *)v17[2];
    }
    while ( v20 );
LABEL_6:
    if ( v18 != v19 && v16 >= v19[4] )
      v18 = v19;
  }
  if ( v18 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_17;
  v22 = v18[7];
  if ( !v22 )
    goto LABEL_17;
  v23 = v18 + 6;
  do
  {
    while ( 1 )
    {
      v24 = *(_QWORD *)(v22 + 16);
      v25 = *(_QWORD *)(v22 + 24);
      if ( *(_DWORD *)(v22 + 32) >= dword_4F8F208 )
        break;
      v22 = *(_QWORD *)(v22 + 24);
      if ( !v25 )
        goto LABEL_15;
    }
    v23 = (_QWORD *)v22;
    v22 = *(_QWORD *)(v22 + 16);
  }
  while ( v24 );
LABEL_15:
  if ( v18 + 6 == v23 || dword_4F8F208 < *((_DWORD *)v23 + 8) || (v29 = qword_4F8F288, !*((_DWORD *)v23 + 9)) )
  {
LABEL_17:
    v26 = (int *)sub_C94E20((__int64)qword_4F86370);
    if ( v26 )
      v27 = *v26;
    else
      v27 = qword_4F86370[2];
    v28 = v27 < 3;
    v29 = 500;
    if ( !v28 )
      v29 = 3200;
  }
  v139 = v29;
  v30 = &v139;
  if ( a8 )
    v30 = a8;
  v157 = v30;
  v133 = a3 & (a7 != 0);
  if ( v133 )
  {
    v133 = 0;
    if ( *(_BYTE *)a7 == 61 )
    {
      if ( (*(_BYTE *)(a7 + 7) & 0x20) != 0 )
        v133 = sub_B91C10(a7, 6) != 0;
      _BitScanReverse64(&v31, 1LL << (*(_WORD *)(a7 + 2) >> 1));
      v131 = 63 - (v31 ^ 0x3F);
    }
  }
  v134 = qword_4F8EFE8 & a9;
  v32 = *(_BYTE *)(a1 + 512) & 1;
  if ( v32 )
  {
    v33 = a1 + 520;
    v34 = 3;
  }
  else
  {
    v34 = *(unsigned int *)(a1 + 528);
    v33 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v34 )
    {
      v65 = *(_DWORD *)(a1 + 512);
      ++*(_QWORD *)(a1 + 504);
      v66 = 0;
      v67 = (v65 >> 1) + 1;
      goto LABEL_94;
    }
    v34 = (unsigned int)(v34 - 1);
  }
  v35 = v34 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
  v36 = (__int64 *)(v33 + 88LL * v35);
  v37 = *v36;
  if ( a6 != *v36 )
  {
    v97 = 1;
    v66 = 0;
    while ( v37 != -4096 )
    {
      if ( !v66 && v37 == -8192 )
        v66 = v36;
      v35 = v34 & (v97 + v35);
      v36 = (__int64 *)(v33 + 88LL * v35);
      v37 = *v36;
      if ( a6 == *v36 )
        goto LABEL_33;
      ++v97;
    }
    v65 = *(_DWORD *)(a1 + 512);
    v68 = 12;
    v34 = 4;
    if ( !v66 )
      v66 = v36;
    ++*(_QWORD *)(a1 + 504);
    v67 = (v65 >> 1) + 1;
    if ( v32 )
    {
LABEL_95:
      if ( 4 * v67 < v68 )
      {
        if ( (int)v34 - *(_DWORD *)(a1 + 516) - v67 > (unsigned int)v34 >> 3 )
        {
LABEL_97:
          *(_DWORD *)(a1 + 512) = (2 * (v65 >> 1) + 2) | v65 & 1;
          if ( *v66 != -4096 )
            --*(_DWORD *)(a1 + 516);
          *v66 = a6;
          v38 = (__int64)(v66 + 1);
          v69 = v66 + 3;
          v70 = v66 + 11;
          *(v70 - 10) = 0;
          *(v70 - 9) = 1;
          do
          {
            if ( v69 )
              *v69 = -4096;
            v69 += 2;
          }
          while ( v69 != v70 );
          goto LABEL_34;
        }
        sub_1033480(a1 + 504, v34);
        if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
        {
          v102 = a1 + 520;
          v103 = 3;
          goto LABEL_193;
        }
        v108 = *(_DWORD *)(a1 + 528);
        v102 = *(_QWORD *)(a1 + 520);
        if ( v108 )
        {
          v103 = v108 - 1;
LABEL_193:
          v104 = v103 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
          v66 = (__int64 *)(v102 + 88LL * v104);
          v34 = *v66;
          if ( a6 != *v66 )
          {
            v105 = 1;
            v106 = 0;
            while ( v34 != -4096 )
            {
              if ( v34 == -8192 && !v106 )
                v106 = v66;
              v104 = v103 & (v105 + v104);
              v66 = (__int64 *)(v102 + 88LL * v104);
              v34 = *v66;
              if ( a6 == *v66 )
                goto LABEL_189;
              ++v105;
            }
LABEL_196:
            if ( v106 )
              v66 = v106;
            goto LABEL_189;
          }
          goto LABEL_189;
        }
LABEL_260:
        *(_DWORD *)(a1 + 512) = (2 * (*(_DWORD *)(a1 + 512) >> 1) + 2) | *(_DWORD *)(a1 + 512) & 1;
        BUG();
      }
      sub_1033480(a1 + 504, 2 * v34);
      if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
      {
        v99 = a1 + 520;
        v100 = 3;
      }
      else
      {
        v107 = *(_DWORD *)(a1 + 528);
        v99 = *(_QWORD *)(a1 + 520);
        if ( !v107 )
          goto LABEL_260;
        v100 = v107 - 1;
      }
      v101 = v100 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
      v66 = (__int64 *)(v99 + 88LL * v101);
      v34 = *v66;
      if ( a6 != *v66 )
      {
        v123 = 1;
        v106 = 0;
        while ( v34 != -4096 )
        {
          if ( !v106 && v34 == -8192 )
            v106 = v66;
          v101 = v100 & (v123 + v101);
          v66 = (__int64 *)(v99 + 88LL * v101);
          v34 = *v66;
          if ( a6 == *v66 )
            goto LABEL_189;
          ++v123;
        }
        goto LABEL_196;
      }
LABEL_189:
      v65 = *(_DWORD *)(a1 + 512);
      goto LABEL_97;
    }
    v34 = *(unsigned int *)(a1 + 528);
LABEL_94:
    v68 = 3 * v34;
    goto LABEL_95;
  }
LABEL_33:
  v38 = (__int64)(v36 + 1);
LABEL_34:
  v149 = v38;
  v150 = a6;
  v152 = a5;
  v154 = (__int64 *)v156;
  v155 = 0x800000000LL;
  v151 = a4;
  v153 = 0;
  v136 = a4;
  do
  {
    while ( 1 )
    {
      do
      {
        while ( 1 )
        {
          while ( 1 )
          {
            while ( 1 )
            {
              while ( 1 )
              {
                v39 = *(_QWORD **)(a6 + 56);
                if ( !v134 )
                  break;
                if ( v151 == v39 )
                  goto LABEL_179;
                v40 = *(_QWORD *)(sub_102E740((__int64)&v149) + 16);
                if ( v40 )
                  v40 -= 24;
                if ( (_BYTE)qword_4F8EF08 )
                {
                  do
                    v13 = (_QWORD *)(*v13 & 0xFFFFFFFFFFFFFFF8LL);
                  while ( v151 != v13 && v13 != *(_QWORD **)(a6 + 56) );
                }
                if ( *(_BYTE *)v40 != 85 )
                  goto LABEL_41;
LABEL_57:
                v50 = *(_QWORD *)(v40 - 32);
                if ( !v50
                  || *(_BYTE *)v50
                  || *(_QWORD *)(v50 + 24) != *(_QWORD *)(v40 + 80)
                  || (*(_BYTE *)(v50 + 33) & 0x20) == 0
                  || (unsigned int)(*(_DWORD *)(v50 + 36) - 68) > 3 )
                {
                  goto LABEL_41;
                }
              }
              if ( v136 == v39 )
              {
                v39 = v151;
LABEL_179:
                v98 = *(_QWORD *)(*(_QWORD *)(a6 + 72) + 80LL);
                if ( !v98 || (v48 = 0x4000000000000003LL, a6 != v98 - 24) )
                  v48 = 0x2000000000000003LL;
                goto LABEL_89;
              }
              v49 = *v136 & 0xFFFFFFFFFFFFFFF8LL;
              v136 = (_QWORD *)v49;
              if ( !v49 )
                BUG();
              v40 = v49 - 24;
              if ( *(_BYTE *)(v49 - 24) == 85 )
                goto LABEL_57;
LABEL_41:
              if ( (*v157)-- == 1 )
              {
                v39 = v151;
                v48 = 0x6000000000000003LL;
                goto LABEL_89;
              }
              v42 = *(_BYTE *)v40;
              if ( *(_BYTE *)v40 > 0x1Cu )
                break;
LABEL_43:
              if ( v42 == 60 )
                goto LABEL_45;
LABEL_44:
              if ( (unsigned __int8)sub_CF6FD0((unsigned __int8 *)v40) )
              {
LABEL_45:
                v34 = 6;
                v43 = sub_98ACB0(*(unsigned __int8 **)a2, 6u);
                if ( v43 == (unsigned __int8 *)v40 )
                  goto LABEL_236;
                v145.m128i_i64[0] = (__int64)v43;
                v34 = (__int64)&v140;
                v145.m128i_i64[1] = 1;
                v44 = *a10;
                v146 = 0u;
                v147 = 0u;
                v140.m128i_i64[0] = v40;
                v140.m128i_i64[1] = 1;
                v141 = 0;
                v142 = 0;
                v143 = 0;
                v144 = 0;
                if ( (unsigned __int8)sub_CF4D50(v44, (__int64)&v140, (__int64)&v145, (__int64)(a10 + 1), 0) == 3 )
                {
LABEL_236:
                  v39 = v151;
                  v48 = v40 | 2;
                  goto LABEL_89;
                }
              }
              if ( *(_BYTE *)v40 == 86 )
              {
                if ( *(_QWORD *)a2 == v40 )
                  goto LABEL_87;
                if ( !v133 )
                  goto LABEL_162;
              }
              else if ( !v133 )
              {
                if ( *(_BYTE *)v40 == 64 && a3 )
                {
                  if ( (*(_WORD *)(v40 + 2) & 7) != 5 )
                  {
                    v34 = v40;
                    v148 = 1;
                    v45 = _mm_loadu_si128((const __m128i *)(a2 + 32));
                    v46 = _mm_loadu_si128((const __m128i *)(a2 + 16));
                    v145 = _mm_loadu_si128((const __m128i *)a2);
                    v47 = (_QWORD *)*a10;
                    v146 = v46;
                    v147 = v45;
                    if ( (unsigned __int8)sub_CF63E0(v47, (unsigned __int8 *)v40, &v145, (__int64)(a10 + 1)) > 1u )
                      goto LABEL_53;
                  }
                }
                else
                {
LABEL_162:
                  v34 = v40;
                  v148 = 1;
                  v145 = _mm_loadu_si128((const __m128i *)a2);
                  v146 = _mm_loadu_si128((const __m128i *)(a2 + 16));
                  v93 = (_QWORD *)*a10;
                  v147 = _mm_loadu_si128((const __m128i *)(a2 + 32));
                  v94 = sub_CF63E0(v93, (unsigned __int8 *)v40, &v145, (__int64)(a10 + 1));
                  if ( v94 == 1 )
                  {
                    if ( !a3 )
                      goto LABEL_53;
                  }
                  else if ( v94 )
                  {
                    goto LABEL_53;
                  }
                }
              }
            }
            if ( v42 != 85 )
              break;
            v71 = *(_QWORD *)(v40 - 32);
            if ( !v71
              || *(_BYTE *)v71
              || *(_QWORD *)(v71 + 24) != *(_QWORD *)(v40 + 80)
              || (*(_BYTE *)(v71 + 33) & 0x20) == 0 )
            {
              goto LABEL_44;
            }
            v72 = *(_DWORD *)(v71 + 36);
            if ( v72 == 211 )
            {
              v34 = (__int64)&v145;
              v73 = *(_QWORD *)(v40 + 32 * (1LL - (*(_DWORD *)(v40 + 4) & 0x7FFFFFF)));
              v146 = 0u;
              v145.m128i_i64[0] = v73;
              v145.m128i_i64[1] = 0xBFFFFFFFFFFFFFFELL;
              v147 = 0u;
              if ( (unsigned __int8)sub_CF4D50(*a10, (__int64)&v145, a2, (__int64)(a10 + 1), 0) == 3 )
                goto LABEL_87;
            }
            else
            {
              if ( v72 <= 0xD2 || (v72 & 0xFFFFFFFD) != 0xE4 )
                goto LABEL_44;
              v145.m128i_i64[0] = 0;
              v145.m128i_i64[1] = -1;
              v95 = *(__int64 **)(a1 + 272);
              v146 = 0u;
              v147 = 0u;
              sub_102A4D0((unsigned __int8 *)v40, &v145, v95);
              v34 = (__int64)&v145;
              v96 = sub_CF4D50(*a10, (__int64)&v145, a2, (__int64)(a10 + 1), 0);
              if ( v96 )
              {
                if ( v96 == 3 )
                {
LABEL_87:
                  v48 = v40 | 2;
                  goto LABEL_88;
                }
                if ( v72 != 228 )
                {
                  v48 = v40 | 1;
                  goto LABEL_88;
                }
              }
            }
          }
          if ( v42 == 61 )
            break;
          if ( v42 != 62 )
            goto LABEL_43;
          v77 = *(_WORD *)(v40 + 2);
          if ( ((v77 >> 7) & 6) != 0 )
          {
            if ( sub_B46500((unsigned __int8 *)v40) )
              goto LABEL_156;
            if ( (v77 & 1) == 0 )
              goto LABEL_126;
          }
          else
          {
            if ( (v77 & 1) == 0 )
              goto LABEL_126;
            if ( sub_B46500((unsigned __int8 *)v40) )
            {
LABEL_156:
              if ( !a7 || sub_B46560((unsigned __int8 *)a7) )
                goto LABEL_53;
              if ( *(_BYTE *)a7 != 61 && *(_BYTE *)a7 != 62 )
              {
                if ( (unsigned __int8)sub_B46420(a7) || (unsigned __int8)sub_B46490(a7) )
                  goto LABEL_53;
                goto LABEL_126;
              }
              v78 = byte_3F70480[8 * ((*(_WORD *)(a7 + 2) >> 7) & 7) + 1];
              goto LABEL_125;
            }
          }
          if ( !a7 )
            goto LABEL_53;
          v78 = sub_B46560((unsigned __int8 *)a7);
LABEL_125:
          if ( v78 )
            goto LABEL_53;
LABEL_126:
          v34 = v40;
          v148 = 1;
          v79 = _mm_loadu_si128((const __m128i *)(a2 + 32));
          v80 = (__int64)(a10 + 1);
          v81 = _mm_loadu_si128((const __m128i *)(a2 + 16));
          v82 = (_QWORD *)*a10;
          v145 = _mm_loadu_si128((const __m128i *)a2);
          v146 = v81;
          v147 = v79;
          if ( (unsigned __int8)sub_CF63E0(v82, (unsigned __int8 *)v40, &v145, (__int64)(a10 + 1)) )
          {
            sub_D66630(&v140, v40);
            v34 = (__int64)&v140;
            v83 = sub_CF4D50(*a10, (__int64)&v140, a2, v80, 0);
            if ( v83 )
            {
              if ( v83 == 3 )
                goto LABEL_87;
              if ( !v133 )
              {
                v34 = 0xBFFFFFFFFFFFFFFELL;
                v130 = *v157;
                v84 = *(_QWORD *)(a2 + 8);
                if ( v84 == 0xBFFFFFFFFFFFFFFELL )
                  goto LABEL_53;
                if ( v84 == -1 )
                  goto LABEL_53;
                v34 = v40;
                sub_D66630(&v145, v40);
                if ( v145.m128i_i64[1] != *(_QWORD *)(a2 + 8) )
                  goto LABEL_53;
                v34 = 0x4000000000000000LL;
                if ( (v145.m128i_i64[1] & 0x4000000000000000LL) != 0 )
                  goto LABEL_53;
                _BitScanReverse64(&v85, 1LL << (*(_WORD *)(v40 + 2) >> 1));
                v86 = 63 - (v85 ^ 0x3F);
                if ( v131 <= v86 )
                  v86 = v131;
                v34 = 1LL << v86;
                if ( 1LL << v86 < (v145.m128i_i64[1] & 0x3FFFFFFFFFFFFFFFuLL)
                  || (v87 = *(_QWORD *)(v40 - 64), *(_BYTE *)v87 != 61)
                  || *(_QWORD *)(v87 + 40) != *(_QWORD *)(v40 + 40)
                  || (v124 = *(_QWORD *)(v40 - 64),
                      sub_D665A0(&v145, v124),
                      v34 = (__int64)&v145,
                      (unsigned __int8)sub_CF4D50(*a10, (__int64)&v145, a2, v80, 0) != 3) )
                {
LABEL_53:
                  v48 = v40 | 1;
                  goto LABEL_88;
                }
                v88 = (unsigned __int8 *)v124;
                v126 = 0;
                while ( v88 != (unsigned __int8 *)v40 )
                {
                  if ( v130 < ++v126 )
                    goto LABEL_53;
                  v89 = &v145;
                  v90 = (__int32 *)a2;
                  for ( i = 12; i; --i )
                  {
                    v89->m128i_i32[0] = *v90++;
                    v89 = (__m128i *)((char *)v89 + 4);
                  }
                  v34 = (__int64)v88;
                  v92 = (_QWORD *)*a10;
                  v148 = 1;
                  v125 = (__int64)v88;
                  if ( (sub_CF63E0(v92, v88, &v145, v80) & 2) != 0 )
                    goto LABEL_53;
                  v34 = 0;
                  v88 = (unsigned __int8 *)sub_B46B10(v125, 0);
                }
              }
            }
          }
        }
        v51 = *(_WORD *)(v40 + 2);
        if ( (v51 & 1) != 0 && (!a7 || sub_B46560((unsigned __int8 *)a7)) )
          goto LABEL_53;
        if ( sub_B46500((unsigned __int8 *)v40) )
        {
          v52 = v51 >> 7;
          if ( byte_3F70480[8 * (v52 & 7) + 1] )
          {
            if ( !a7 || sub_B46560((unsigned __int8 *)a7) )
              goto LABEL_53;
            if ( *(_BYTE *)a7 == 61 || *(_BYTE *)a7 == 62 )
            {
              if ( byte_3F70480[8 * ((*(_WORD *)(a7 + 2) >> 7) & 7)] )
                goto LABEL_53;
            }
            else if ( (unsigned __int8)sub_B46420(a7) || (unsigned __int8)sub_B46490(a7) )
            {
              goto LABEL_53;
            }
            if ( (v52 & 7) != 2 )
              goto LABEL_53;
          }
        }
        sub_D665A0(&v145, v40);
        v34 = (__int64)&v145;
        v53 = sub_CF4D50(*a10, (__int64)&v145, a2, (__int64)(a10 + 1), 0);
        v54 = v53 >> 9;
      }
      while ( !(_BYTE)v53 );
      if ( a3 )
        break;
      v34 = (__int64)&v145;
      if ( (sub_CF4FA0(*a10, (__int64)&v145, (__int64)(a10 + 1), 0) & 2) != 0 )
        goto LABEL_87;
    }
    if ( (_BYTE)v53 == 3 )
      goto LABEL_87;
  }
  while ( (_BYTE)v53 != 2 || (v53 & 0x100) == 0 );
  v34 = *(unsigned int *)(a1 + 1008);
  v55 = a1 + 984;
  if ( !(_DWORD)v34 )
  {
    ++*(_QWORD *)(a1 + 984);
    goto LABEL_229;
  }
  v56 = 1;
  v57 = *(_QWORD *)(a1 + 992);
  v58 = 0;
  v59 = (v34 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
  v60 = (_QWORD *)(v57 + 16LL * v59);
  v61 = *v60;
  if ( *v60 == v40 )
  {
LABEL_82:
    v62 = (int *)(v60 + 1);
    goto LABEL_83;
  }
  while ( v61 != -4096 )
  {
    if ( !v58 && v61 == -8192 )
      v58 = v60;
    v59 = (v34 - 1) & (v56 + v59);
    v60 = (_QWORD *)(v57 + 16LL * v59);
    v61 = *v60;
    if ( *v60 == v40 )
      goto LABEL_82;
    ++v56;
  }
  v109 = *(_DWORD *)(a1 + 1000);
  if ( !v58 )
    v58 = v60;
  ++*(_QWORD *)(a1 + 984);
  v110 = v109 + 1;
  if ( 4 * v110 >= (unsigned int)(3 * v34) )
  {
LABEL_229:
    v34 = (unsigned int)(2 * v34);
    v138 = v54;
    sub_102FB10(v55, v34);
    v117 = *(_DWORD *)(a1 + 1008);
    if ( v117 )
    {
      v118 = v117 - 1;
      v119 = *(_QWORD *)(a1 + 992);
      v54 = v138;
      v120 = v118 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v110 = *(_DWORD *)(a1 + 1000) + 1;
      v58 = (_QWORD *)(v119 + 16LL * v120);
      v121 = *v58;
      if ( *v58 != v40 )
      {
        v122 = 1;
        v34 = 0;
        while ( v121 != -4096 )
        {
          if ( !v34 && v121 == -8192 )
            v34 = (__int64)v58;
          v120 = v118 & (v122 + v120);
          v58 = (_QWORD *)(v119 + 16LL * v120);
          v121 = *v58;
          if ( *v58 == v40 )
            goto LABEL_219;
          ++v122;
        }
        if ( v34 )
          v58 = (_QWORD *)v34;
      }
      goto LABEL_219;
    }
    goto LABEL_261;
  }
  if ( (int)v34 - *(_DWORD *)(a1 + 1004) - v110 <= (unsigned int)v34 >> 3 )
  {
    v137 = v54;
    sub_102FB10(v55, v34);
    v111 = *(_DWORD *)(a1 + 1008);
    if ( v111 )
    {
      v112 = v111 - 1;
      v113 = *(_QWORD *)(a1 + 992);
      v114 = 0;
      v115 = v112 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v54 = v137;
      v116 = 1;
      v110 = *(_DWORD *)(a1 + 1000) + 1;
      v58 = (_QWORD *)(v113 + 16LL * v115);
      v34 = *v58;
      if ( *v58 != v40 )
      {
        while ( v34 != -4096 )
        {
          if ( !v114 && v34 == -8192 )
            v114 = v58;
          v115 = v112 & (v116 + v115);
          v58 = (_QWORD *)(v113 + 16LL * v115);
          v34 = *v58;
          if ( *v58 == v40 )
            goto LABEL_219;
          ++v116;
        }
        if ( v114 )
          v58 = v114;
      }
      goto LABEL_219;
    }
LABEL_261:
    ++*(_DWORD *)(a1 + 1000);
    BUG();
  }
LABEL_219:
  *(_DWORD *)(a1 + 1000) = v110;
  if ( *v58 != -4096 )
    --*(_DWORD *)(a1 + 1004);
  *v58 = v40;
  v62 = (int *)(v58 + 1);
  *v62 = 0;
LABEL_83:
  *v62 = v54;
  v48 = v40 | 1;
LABEL_88:
  v39 = v151;
LABEL_89:
  v63 = v154;
  if ( *(_QWORD **)(v150 + 56) == v39 )
  {
    v74 = &v154[(unsigned int)v155];
    if ( v154 != v74 )
    {
      v75 = v154;
      do
      {
        v76 = *v75;
        v34 = (__int64)&v145;
        ++v75;
        v145.m128i_i64[0] = v76;
        *sub_102E450(v149, v145.m128i_i64) = 0;
      }
      while ( v74 != v75 );
      v63 = v154;
    }
  }
  if ( v63 != (__int64 *)v156 )
    _libc_free(v63, v34);
  return v48;
}
