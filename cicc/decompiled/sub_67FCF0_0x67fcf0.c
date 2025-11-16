// Function: sub_67FCF0
// Address: 0x67fcf0
//
__int64 __fastcall sub_67FCF0(_QWORD *a1, char a2, char *a3, int a4)
{
  const __m128i *v5; // r15
  __int64 v6; // rax
  char v7; // al
  __int64 result; // rax
  char v9; // al
  _QWORD *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  int v13; // r8d
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  const char *v17; // r13
  size_t v18; // rax
  _QWORD *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // r14
  char v24; // si
  char v25; // al
  __int64 v26; // rbx
  int v27; // eax
  __int64 v28; // rsi
  __int64 *v29; // rax
  int v30; // edx
  int v31; // ecx
  int v32; // r8d
  int v33; // r9d
  char v34; // r12
  __int64 v35; // rbx
  __int64 v36; // r13
  __int64 v37; // r14
  __int64 v38; // rdx
  char *v39; // r12
  size_t v40; // rax
  size_t v41; // rdx
  const char *v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned __int8 v46; // al
  __int64 v47; // rcx
  __int64 v48; // r10
  __int64 v49; // r13
  __int64 v50; // rdx
  __int64 i; // r11
  char v52; // cl
  int v53; // r9d
  _QWORD *v54; // rdi
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdx
  char v58; // cl
  __int64 v59; // r11
  __int64 v60; // r10
  char v61; // al
  char v62; // al
  char v63; // al
  __int64 v64; // r10
  __int64 v65; // r11
  char v66; // al
  _QWORD *v67; // rbx
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rax
  char *v74; // rax
  char *v75; // rax
  __int64 v76; // rbx
  char *v77; // rax
  const char *v78; // r12
  size_t v79; // rax
  int v80; // eax
  char *v81; // rax
  __int64 v82; // rax
  __int64 v83; // rdx
  char v84; // al
  __int64 v85; // rax
  __int64 *v86; // rax
  const char *v87; // r13
  size_t v88; // rax
  const char *v89; // rsi
  char v90; // al
  char *v91; // rbx
  char *v92; // rax
  __int64 v93; // r14
  __int64 v94; // r15
  char v95; // al
  __int64 v96; // rax
  char v97; // al
  __int64 v98; // rax
  __int64 **v99; // rbx
  size_t v100; // rax
  __int64 v101; // rdi
  __int64 v102; // r11
  __int64 v103; // r10
  char *v104; // rax
  __int64 v105; // rax
  __int64 v106; // rax
  char *v107; // rax
  char v108; // al
  char v109; // al
  char v110; // al
  __int64 **v111; // [rsp+8h] [rbp-128h]
  const char *v112; // [rsp+8h] [rbp-128h]
  __int64 v113; // [rsp+10h] [rbp-120h]
  __int64 v114; // [rsp+10h] [rbp-120h]
  __int64 v115; // [rsp+18h] [rbp-118h]
  __int64 v116; // [rsp+18h] [rbp-118h]
  __int64 v117; // [rsp+18h] [rbp-118h]
  __int64 v118; // [rsp+18h] [rbp-118h]
  __int64 v119; // [rsp+18h] [rbp-118h]
  __int64 v120; // [rsp+18h] [rbp-118h]
  __int64 v121; // [rsp+18h] [rbp-118h]
  __int64 v122; // [rsp+20h] [rbp-110h]
  _BOOL4 v123; // [rsp+20h] [rbp-110h]
  __int64 v124; // [rsp+20h] [rbp-110h]
  __int64 v125; // [rsp+20h] [rbp-110h]
  __int64 v126; // [rsp+20h] [rbp-110h]
  __int64 v127; // [rsp+28h] [rbp-108h]
  _BOOL4 v128; // [rsp+28h] [rbp-108h]
  __int64 v129; // [rsp+28h] [rbp-108h]
  char v130; // [rsp+28h] [rbp-108h]
  char v131; // [rsp+28h] [rbp-108h]
  __int64 v132; // [rsp+28h] [rbp-108h]
  __int64 v133; // [rsp+28h] [rbp-108h]
  char v134; // [rsp+30h] [rbp-100h]
  char v135; // [rsp+30h] [rbp-100h]
  char v136; // [rsp+30h] [rbp-100h]
  __int64 v137; // [rsp+30h] [rbp-100h]
  __int64 v138; // [rsp+38h] [rbp-F8h]
  int v139; // [rsp+40h] [rbp-F0h]
  __int64 v140; // [rsp+40h] [rbp-F0h]
  __int64 v141; // [rsp+40h] [rbp-F0h]
  __int64 v142; // [rsp+40h] [rbp-F0h]
  size_t n; // [rsp+48h] [rbp-E8h]
  unsigned __int8 v144; // [rsp+56h] [rbp-DAh]
  _BYTE v145[9]; // [rsp+57h] [rbp-D9h] BYREF
  _BYTE v146[4]; // [rsp+60h] [rbp-D0h] BYREF
  int v147; // [rsp+64h] [rbp-CCh]
  __int16 v148; // [rsp+9Ah] [rbp-96h]
  __m128i si128; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v150; // [rsp+B0h] [rbp-80h]
  char *s; // [rsp+E0h] [rbp-50h]
  __int64 v152; // [rsp+E8h] [rbp-48h]
  __int64 v153; // [rsp+F0h] [rbp-40h]

  switch ( a2 )
  {
    case 'T':
      v5 = (const __m128i *)a1[23];
      v6 = 6;
      if ( v5 )
        goto LABEL_5;
      if ( !*a3 )
      {
        sub_8238B0(qword_4D039E8, "\"<", 2);
        BUG();
      }
      goto LABEL_23;
    case 'd':
      v5 = (const __m128i *)a1[23];
      v6 = 0;
      if ( !v5 )
        goto LABEL_30;
      goto LABEL_5;
    case 'n':
      v5 = (const __m128i *)a1[23];
      if ( !v5 )
        goto LABEL_18;
      v6 = 4;
      goto LABEL_5;
    case 'p':
      v5 = (const __m128i *)a1[23];
      v6 = 2;
      if ( v5 )
        goto LABEL_5;
      if ( *a3 )
        goto LABEL_23;
      return (__int64)sub_67C870((unsigned int *)&v5[1], a1, byte_3F871B3, byte_3F871B3, byte_3F871B3);
    case 'r':
      v5 = (const __m128i *)a1[23];
      v6 = 7;
      if ( v5 )
        goto LABEL_5;
      if ( !*a3 )
      {
        si128 = _mm_load_si128((const __m128i *)0x10);
        BUG();
      }
      goto LABEL_23;
    case 's':
      v5 = (const __m128i *)a1[23];
      if ( !v5 )
        goto LABEL_41;
      v6 = 3;
      goto LABEL_5;
    case 't':
      v5 = (const __m128i *)a1[23];
      v6 = 5;
      if ( v5 )
        goto LABEL_5;
      if ( !*a3 )
      {
        byte_4CFFE51 = 1;
        sub_8238B0(qword_4D039E8, "\"", 1);
        BUG();
      }
      goto LABEL_23;
    case 'u':
      v5 = (const __m128i *)a1[23];
      v6 = 1;
      if ( !v5 )
      {
LABEL_30:
        if ( !*a3 )
          BUG();
LABEL_23:
        sub_721090(a1);
      }
      do
      {
LABEL_5:
        if ( v5->m128i_i32[0] == (_DWORD)v6 && !--a4 )
          break;
        v5 = (const __m128i *)v5->m128i_i64[1];
      }
      while ( v5 );
      if ( (_DWORD)v6 != 3 )
      {
        if ( (_DWORD)v6 != 4 )
        {
          if ( *a3 )
            goto LABEL_23;
          switch ( v6 )
          {
            case 0LL:
              v43 = v5[1].m128i_i64[0];
              v148 = 0;
              v147 = 0;
              v152 = 0;
              v153 = 0;
              si128.m128i_i32[1] = 1;
              s = &si128.m128i_i8[8];
              sub_67F6D0((__int64)&si128, (__int64)v146, v43);
              goto LABEL_81;
            case 1LL:
              v38 = v5[1].m128i_i64[0];
              v148 = 0;
              v147 = 0;
              v152 = 0;
              v153 = 0;
              si128.m128i_i32[1] = 1;
              s = &si128.m128i_i8[8];
              sub_67F9E0((__int64)&si128, (__int64)v146, v38);
LABEL_81:
              v39 = s;
              v40 = strlen(s);
              result = sub_8238B0(qword_4D039E8, v39, v40);
              if ( s != (char *)&si128.m128i_u64[1] )
                return sub_823A00(s, v152);
              return result;
            case 2LL:
              return (__int64)sub_67C870((unsigned int *)&v5[1], a1, byte_3F871B3, byte_3F871B3, byte_3F871B3);
            case 5LL:
              v34 = byte_4CFFE51;
              byte_4CFFE51 = 1;
              sub_8238B0(qword_4D039E8, "\"", 1);
              v35 = qword_4D039E8;
              v36 = *(_QWORD *)(qword_4D039E8 + 16);
              sub_74B930(v5[1].m128i_i64[0], &qword_4CFFDC0);
              v37 = *(_QWORD *)(v35 + 16);
              sub_8238B0(qword_4D039E8, "\"", 1);
              result = sub_8DBE50(v5[1].m128i_i64[0]);
              if ( (_DWORD)result )
              {
                v145[0] = byte_4CFFE59;
                n = v37 - v36;
                v141 = *(_QWORD *)(qword_4D039E8 + 16);
                byte_4CFFE59 = 1;
                byte_4CFFE51 = 0;
                sub_8238B0(qword_4D039E8, " (aka \"", 7);
                v93 = qword_4D039E8;
                *(_QWORD *)&v145[1] = *(_QWORD *)(qword_4D039E8 + 16);
                sub_74B930(v5[1].m128i_i64[0], &qword_4CFFDC0);
                v94 = *(_QWORD *)(v93 + 16) - *(_QWORD *)&v145[1];
                sub_8238B0(qword_4D039E8, "\")", 2);
                if ( n == v94
                  && !strncmp(
                        (const char *)(v36 + *(_QWORD *)(v35 + 32)),
                        (const char *)(*(_QWORD *)(v93 + 32) + *(_QWORD *)&v145[1]),
                        n) )
                {
                  sub_823940(qword_4D039E8, v141);
                }
                result = v145[0];
                byte_4CFFE59 = v145[0];
              }
              byte_4CFFE51 = v34;
              return result;
            case 6LL:
              sub_8238B0(qword_4D039E8, "\"<", 2);
              v26 = v5[1].m128i_i64[0];
              v27 = 1;
              if ( !v26 )
                goto LABEL_86;
              break;
            case 7LL:
              si128 = _mm_loadu_si128(v5 + 1);
              v150 = v5[2].m128i_i64[0];
              v28 = v5[1].m128i_u8[0];
              v29 = (__int64 *)sub_72A270(v5[1].m128i_i64[1], v28);
              if ( v29 && *v29 )
                return sub_67C3A0(*v29);
              else
                return sub_74C5E0(
                         (unsigned int)&qword_4CFFDC0,
                         v28,
                         v30,
                         v31,
                         v32,
                         v33,
                         si128.m128i_i8[0],
                         si128.m128i_i64[1]);
          }
          while ( 1 )
          {
            if ( *(_BYTE *)(v26 + 8) != 3 )
            {
              if ( !v27 )
                goto LABEL_74;
              while ( 1 )
              {
                sub_747370(v26, &qword_4CFFDC0);
                v26 = *(_QWORD *)v26;
                if ( !v26 )
                {
LABEL_86:
                  v41 = 2;
                  v42 = ">\"";
                  return sub_8238B0(qword_4D039E8, v42, v41);
                }
                if ( *(_BYTE *)(v26 + 8) == 3 )
                  break;
LABEL_74:
                sub_8238B0(qword_4D039E8, ", ", 2);
              }
              v27 = 0;
            }
            v26 = *(_QWORD *)v26;
            if ( !v26 )
              goto LABEL_86;
          }
        }
LABEL_18:
        while ( 1 )
        {
          v7 = *a3;
          if ( !*a3 )
            break;
          switch ( v7 )
          {
            case 'f':
              v5[1].m128i_i8[12] = 1;
              break;
            case 'o':
              v5[1].m128i_i8[13] = 1;
              break;
            case 'p':
              v5[1].m128i_i8[14] = 1;
              break;
            case 't':
              v5[1].m128i_i8[15] = 1;
              v5[1].m128i_i8[12] = 1;
              break;
            case 'a':
              v5[1].m128i_i8[13] = 1;
              v5[2].m128i_i8[1] = 1;
              break;
            case 'd':
              v5[2].m128i_i8[0] = 1;
              break;
            case 'T':
              v5[2].m128i_i8[2] = 1;
              break;
          }
          ++a3;
        }
        v23 = v5[1].m128i_i64[0];
        v24 = *(_BYTE *)(v23 + 80);
        v144 = byte_4CFFE55;
        v138 = v23;
        v25 = v24;
        if ( v24 == 16 )
        {
          v23 = **(_QWORD **)(v23 + 88);
          v25 = *(_BYTE *)(v23 + 80);
        }
        if ( v25 == 24 )
        {
          v23 = *(_QWORD *)(v23 + 88);
          v25 = *(_BYTE *)(v23 + 80);
        }
        switch ( v25 )
        {
          case 0:
            if ( !v5[1].m128i_i8[13] )
            {
              v107 = sub_67C860(1461);
              sub_823910(qword_4D039E8, v107);
              sub_8238B0(qword_4D039E8, " ", 1);
            }
            sub_8238B0(qword_4D039E8, "\"", 1);
            v87 = *(const char **)(*(_QWORD *)v138 + 8LL);
            v88 = strlen(v87);
            v89 = v87;
            v49 = 0;
            sub_8238B0(qword_4D039E8, v89, v88);
            goto LABEL_151;
          case 1:
            v52 = 0;
            v48 = 0;
            v49 = 0;
            memset(v145, 0, sizeof(v145));
            v53 = 1462;
            v139 = 0;
            goto LABEL_111;
          case 2:
            v53 = 1477;
            v139 = 0;
            v49 = 0;
            i = *(_QWORD *)(*(_QWORD *)(v23 + 88) + 128LL);
            if ( *(_BYTE *)(i + 140) == 14 )
              v53 = (*(_BYTE *)(i + 160) != 2) + 1476;
            goto LABEL_210;
          case 3:
            v52 = 0;
            memset(v145, 0, sizeof(v145));
            v48 = 0;
            v139 = 0;
            v49 = 0;
            if ( *(_BYTE *)(*(_QWORD *)(v23 + 88) + 140LL) == 14 )
              v53 = 1464;
            else
              v53 = 1465;
            goto LABEL_111;
          case 4:
          case 5:
            if ( dword_4F077C4 == 2 && *(char *)(*(_QWORD *)(v23 + 88) + 177LL) < 0 )
              goto LABEL_176;
            v53 = 1466;
            if ( v25 != 5 )
              v53 = (dword_4F077C4 != 2) + 1467;
            v139 = unk_4F0697C;
            if ( unk_4F0697C )
            {
              v52 = v5[1].m128i_i8[15];
              if ( v52 )
              {
                v139 = 0;
                v48 = sub_67C320(*(_QWORD *)(v23 + 88));
                v145[8] = 0;
                *(_QWORD *)v145 = v48 != 0;
                v49 = 0;
                v52 = v48 != 0;
              }
              else
              {
                memset(v145, 0, sizeof(v145));
                v48 = 0;
                v49 = 0;
                v139 = 0;
              }
            }
            else
            {
              memset(v145, 0, sizeof(v145));
              v52 = 0;
              v48 = 0;
              v49 = 0;
            }
            goto LABEL_111;
          case 6:
            v52 = 0;
            v48 = 0;
            v49 = 0;
            memset(v145, 0, sizeof(v145));
            v53 = 1472;
            v139 = 0;
            goto LABEL_111;
          case 7:
            v82 = *(_QWORD *)(v23 + 88);
            i = *(_QWORD *)(v82 + 120);
            if ( *(char *)(v82 + 169) < 0 )
              goto LABEL_214;
            if ( *(char *)(v82 + 171) < 0 )
            {
              v52 = 0;
              v48 = 0;
              v49 = 0;
              v145[0] = 0;
              v80 = 1;
              v53 = 1474;
            }
            else
            {
              v52 = *(_BYTE *)(v82 + 170) & 1;
              if ( v52 )
              {
                v52 = 0;
                v48 = 0;
                v49 = 0;
                memset(v145, 0, sizeof(v145));
                v53 = 2937;
                v139 = 0;
                goto LABEL_111;
              }
              v49 = *(_QWORD *)(v23 + 96);
              if ( v49 )
              {
                if ( unk_4F0697C && (v52 = v5[1].m128i_i8[15]) != 0 )
                {
                  v48 = **(_QWORD **)(*(_QWORD *)(v82 + 216) + 16LL);
                  v83 = *(_QWORD *)(v48 + 88);
                  if ( *(_QWORD *)(v83 + 88) && (*(_BYTE *)(v83 + 160) & 1) == 0 )
                    v48 = *(_QWORD *)(v83 + 88);
                  v84 = *(_BYTE *)(v48 + 80);
                  if ( v84 == 9 || v84 == 7 )
                  {
                    v85 = *(_QWORD *)(v48 + 88);
                  }
                  else
                  {
                    if ( v84 != 21 )
                      BUG();
                    v85 = *(_QWORD *)(*(_QWORD *)(v48 + 88) + 192LL);
                  }
                  i = *(_QWORD *)(v85 + 120);
                  v145[0] = 1;
                  v52 = 1;
                  v49 = 0;
                  v80 = 1;
                  v53 = 1475;
                }
                else
                {
                  v145[0] = 0;
                  v48 = 0;
                  v80 = 1;
                  v49 = 0;
                  v53 = 1475;
                }
              }
              else
              {
                v145[0] = 0;
                v48 = 0;
                v80 = 1;
                v53 = 1475;
              }
            }
LABEL_182:
            if ( i )
              goto LABEL_183;
            *(_QWORD *)&v145[1] = 0;
            v139 = 0;
            goto LABEL_111;
          case 8:
            v52 = 0;
            v145[0] = 0;
            v48 = 0;
            i = *(_QWORD *)(*(_QWORD *)(v23 + 88) + 120LL);
            if ( dword_4F077C4 == 2 )
            {
              v80 = 1;
              v53 = 1480;
            }
            else
            {
              v80 = 0;
              v53 = 1481;
            }
            v49 = 0;
            goto LABEL_182;
          case 9:
            v49 = *(_QWORD *)(v23 + 96);
            i = *(_QWORD *)(*(_QWORD *)(v23 + 88) + 120LL);
            if ( v49 )
            {
              if ( unk_4F0697C )
              {
                v48 = *(_QWORD *)(v49 + 32);
                v53 = 1480;
                v145[0] = v48 != 0;
                v49 = 0;
                v52 = v48 != 0;
                v80 = 1;
              }
              else
              {
                v145[0] = 0;
                v52 = 0;
                v48 = 0;
                v49 = 0;
                v80 = 1;
                v53 = 1480;
              }
            }
            else
            {
              v48 = 0;
              v52 = 0;
              v80 = 1;
              v53 = 1480;
              v145[0] = 0;
            }
            goto LABEL_182;
          case 10:
          case 11:
            v44 = *(_QWORD *)(v23 + 96);
            if ( v44 && unk_4F0697C )
            {
              v45 = *(_QWORD *)(v44 + 32);
              v46 = *(_BYTE *)(v45 + 80);
              if ( (unsigned __int8)(v46 - 19) > 3u )
              {
                v48 = v45;
                goto LABEL_291;
              }
              v47 = *(_QWORD *)(v45 + 88);
              v48 = *(_QWORD *)(v47 + 88);
              if ( v48 && (*(_BYTE *)(v47 + 160) & 1) == 0 )
              {
                v46 = *(_BYTE *)(v48 + 80);
                if ( v46 <= 0xAu )
                  goto LABEL_291;
              }
              else
              {
                v48 = v45;
              }
              if ( (unsigned __int8)(v46 - 19) <= 3u )
              {
                v49 = *(_QWORD *)(v23 + 88);
                v50 = v49;
                if ( v46 == 20 )
                {
                  for ( i = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v48 + 88) + 176LL) + 152LL);
                        *(_BYTE *)(i + 140) == 12;
                        i = *(_QWORD *)(i + 160) )
                  {
                    ;
                  }
LABEL_103:
                  v145[0] = 1;
                  v52 = 1;
                  goto LABEL_284;
                }
LABEL_292:
                v49 = v50;
                for ( i = *(_QWORD *)(*(_QWORD *)(v48 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
                  ;
                goto LABEL_103;
              }
LABEL_291:
              v50 = *(_QWORD *)(v23 + 88);
              goto LABEL_292;
            }
            v49 = *(_QWORD *)(v23 + 88);
            for ( i = *(_QWORD *)(v49 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
              ;
            v145[0] = 0;
            v52 = 0;
            v48 = 0;
LABEL_284:
            v53 = 2892;
            if ( *(_BYTE *)(v49 + 174) != 7 )
              v53 = 1478;
            if ( (*(_BYTE *)(v49 + 206) & 2) != 0 )
            {
              v49 = 0;
              v139 = 0;
              *(_QWORD *)&v145[1] = **(_QWORD **)(v23 + 64);
              goto LABEL_111;
            }
            v80 = 1;
LABEL_183:
            if ( v5[1].m128i_i8[13] )
            {
              v139 = 0;
              *(_QWORD *)&v145[1] = 0;
            }
            else if ( !v5[1].m128i_i8[12] || (v139 = 0, *(_QWORD *)&v145[1] = 0, !v80) )
            {
              v139 = 0;
              *(_QWORD *)&v145[1] = 0;
              goto LABEL_187;
            }
            goto LABEL_112;
          case 12:
            v52 = 0;
            v48 = 0;
            v49 = 0;
            memset(v145, 0, sizeof(v145));
            v53 = 1463;
            v139 = 0;
            goto LABEL_111;
          case 13:
            v52 = 0;
            v48 = 0;
            v49 = 0;
            memset(v145, 0, sizeof(v145));
            i = 0;
            v139 = 0;
            goto LABEL_112;
          case 14:
            v52 = 0;
            v48 = 0;
            v145[0] = 0;
            v53 = 1475;
            v49 = 0;
            i = **(_QWORD **)(v23 + 88);
            v80 = 1;
            goto LABEL_182;
          case 15:
            v86 = *(__int64 **)(v23 + 88);
            v52 = 0;
            v48 = 0;
            v145[0] = 0;
            v53 = 1478;
            i = *v86;
            v49 = v86[1];
            v80 = 1;
            goto LABEL_182;
          case 17:
            v52 = 0;
            v48 = 0;
            v49 = 0;
            memset(v145, 0, sizeof(v145));
            v53 = 1479;
            v139 = 0;
            goto LABEL_111;
          case 18:
            i = *(_QWORD *)(*(_QWORD *)(v23 + 88) + 16LL);
LABEL_214:
            v52 = 0;
            v48 = 0;
            v49 = 0;
            v145[0] = 0;
            v80 = 1;
            v53 = 1473;
            goto LABEL_182;
          case 19:
LABEL_176:
            v52 = *(_BYTE *)(v138 + 81) & 0x40;
            if ( v52 )
            {
              v52 = 0;
              v48 = 0;
              v49 = 0;
              memset(v145, 0, sizeof(v145));
              v53 = 1469;
              v139 = 0;
            }
            else
            {
              if ( v24 != 19 )
              {
LABEL_178:
                v48 = 0;
                memset(v145, 0, sizeof(v145));
                v49 = 0;
                v139 = 0;
                v53 = 1471;
                goto LABEL_111;
              }
              v106 = *(_QWORD *)(v138 + 88);
              if ( (*(_BYTE *)(v106 + 160) & 2) != 0 )
              {
                v48 = 0;
                memset(v145, 0, sizeof(v145));
                v49 = 0;
                v139 = 0;
                v53 = 1470;
              }
              else
              {
                v52 = *(_BYTE *)(v106 + 265) & 1;
                if ( !v52 )
                  goto LABEL_178;
                v52 = 0;
                v48 = 0;
                v49 = 0;
                memset(v145, 0, sizeof(v145));
                v53 = 1889;
                v139 = 0;
              }
            }
LABEL_111:
            i = 0;
            if ( !v5[1].m128i_i8[13] )
            {
LABEL_187:
              v124 = v48;
              v129 = i;
              v136 = v52;
              v81 = sub_67C860(v53);
              sub_823910(qword_4D039E8, v81);
              sub_8238B0(qword_4D039E8, " ", 1);
              v52 = v136;
              i = v129;
              v48 = v124;
            }
LABEL_112:
            v54 = (_QWORD *)qword_4D039E8;
            if ( dword_4F073CC[0] )
            {
              v55 = *(_QWORD *)(qword_4D039E8 + 16);
              if ( (unsigned __int64)(v55 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
              {
                v119 = v48;
                v126 = i;
                v131 = v52;
                v137 = qword_4D039E8;
                sub_823810(qword_4D039E8);
                v54 = (_QWORD *)v137;
                v48 = v119;
                i = v126;
                v52 = v131;
                v55 = *(_QWORD *)(v137 + 16);
              }
              *(_BYTE *)(v54[4] + v55) = 27;
              v56 = v54[2];
              v57 = v56 + 1;
              v54[2] = v56 + 1;
              if ( (unsigned __int64)(v56 + 2) > v54[1] )
              {
                v118 = v48;
                v125 = i;
                v130 = v52;
                sub_823810(v54);
                v48 = v118;
                i = v125;
                v52 = v130;
                v57 = v54[2];
              }
              *(_BYTE *)(v54[4] + v57) = 6;
              ++v54[2];
              v54 = (_QWORD *)qword_4D039E8;
            }
            v122 = v48;
            v127 = i;
            v134 = v52;
            sub_8238B0(v54, "\"", 1);
            v58 = v134;
            v59 = v127;
            v60 = v122;
            if ( v49 )
            {
              v61 = sub_877F80(v23);
              v58 = v134;
              v59 = v127;
              v60 = v122;
              v123 = v61 != 1
                  && (v62 = sub_877F80(v23), v58 = v134, v59 = v127, v60 = v122, v62 != 2)
                  && *(_BYTE *)(v49 + 174) != 3;
              if ( *(_QWORD *)&v145[1] )
              {
                v63 = byte_4CFFE51;
                byte_4CFFE51 = v58;
                v135 = v63;
                goto LABEL_124;
              }
              if ( v60 )
              {
                v97 = byte_4CFFE51;
                *(_QWORD *)&v145[1] = v60;
                byte_4CFFE51 = v58;
                v135 = v97;
                goto LABEL_124;
              }
              if ( (*(_DWORD *)(v138 + 80) & 0x41000) == 0 )
              {
                v110 = byte_4CFFE51;
                *(_QWORD *)&v145[1] = v23;
                byte_4CFFE51 = v58;
                v135 = v110;
LABEL_124:
                if ( !v5[1].m128i_i8[14] )
                {
                  if ( *(_BYTE *)(v23 + 80) != 20 )
                  {
                    v128 = 1;
                    if ( *(_QWORD *)(v49 + 240) )
                      goto LABEL_127;
                  }
                  if ( (*(_BYTE *)(*(_QWORD *)&v145[1] + 83LL) & 0x20) == 0 )
                  {
                    v128 = 0;
                    if ( v60 == *(_QWORD *)&v145[1] )
                    {
                      v128 = (*(_BYTE *)(v23 + 83) & 0x20) != 0;
                      if ( !v59 )
                        goto LABEL_133;
                      goto LABEL_128;
                    }
LABEL_127:
                    if ( !v59 )
                      goto LABEL_132;
LABEL_128:
                    if ( (v5[1].m128i_i8[12] || v139) && (!v49 || v123) )
                    {
                      v114 = v60;
                      v117 = v59;
                      sub_74A390(v59, 0, 1, 0, 0, &qword_4CFFDC0);
                      v60 = v114;
                      v59 = v117;
                    }
LABEL_132:
                    if ( v60 )
                    {
LABEL_133:
                      v113 = v60;
                      v115 = v59;
                      sub_87D380(*(_QWORD *)&v145[1], &qword_4CFFDC0);
                      v64 = v113;
                      v65 = v115;
                      goto LABEL_134;
                    }
                    v116 = v59;
                    sub_67C3A0(*(__int64 *)&v145[1]);
                    v65 = v116;
                    v64 = 0;
LABEL_134:
                    if ( v65 && !v5[1].m128i_i8[13] && (v5[1].m128i_i8[12] || v128) )
                    {
                      if ( *(_BYTE *)(*(_QWORD *)&v145[1] + 80LL) == 20 )
                      {
                        if ( unk_4F0697C )
                        {
                          v98 = *(_QWORD *)(*(_QWORD *)&v145[1] + 88LL);
                          if ( (*(_BYTE *)(v98 + 424) & 1) != 0 )
                          {
                            v111 = **(__int64 ****)(v98 + 328);
                            if ( v111 )
                            {
                              v120 = v64;
                              v132 = v65;
                              sub_8238B0(qword_4D039E8, "<", 1);
                              v99 = v111;
                              do
                              {
                                v112 = *(const char **)(*v99[1] + 8);
                                v100 = strlen(v112);
                                sub_8238B0(qword_4D039E8, v112, v100);
                                if ( ((_BYTE)v99[7] & 0x10) != 0 )
                                  sub_8238B0(qword_4D039E8, "...", 3);
                                v101 = qword_4D039E8;
                                if ( !*v99 )
                                {
                                  v102 = v132;
                                  v103 = v120;
                                  goto LABEL_270;
                                }
                                sub_8238B0(qword_4D039E8, ",", 1);
                                v99 = (__int64 **)*v99;
                              }
                              while ( v99 );
                              v102 = v132;
                              v103 = v120;
                              v101 = qword_4D039E8;
LABEL_270:
                              v121 = v103;
                              v133 = v102;
                              sub_8238B0(v101, ">", 1);
                              v65 = v133;
                              v64 = v121;
                            }
                          }
                        }
                      }
                      if ( (!v49 || v123) && (v5[1].m128i_i8[12] || v139) )
                      {
                        v142 = v64;
                        sub_74D110(v65, 0, 0, &qword_4CFFDC0);
                        v64 = v142;
                      }
                      else
                      {
                        v140 = v64;
                        sub_74BA50(v65, &qword_4CFFDC0);
                        v64 = v140;
                      }
                    }
                    byte_4CFFE51 = v135;
                    if ( !unk_4F0697C )
                      goto LABEL_151;
                    if ( v145[0] )
                    {
                      v66 = *(_BYTE *)(v23 + 80);
                      if ( v66 == 16 )
                      {
                        v23 = **(_QWORD **)(v23 + 88);
                        v66 = *(_BYTE *)(v23 + 80);
                      }
                      if ( v66 == 24 )
                      {
                        v23 = *(_QWORD *)(v23 + 88);
                        v66 = *(_BYTE *)(v23 + 80);
                      }
                      if ( (unsigned __int8)(v66 - 10) > 1u )
                      {
                        if ( v66 != 20 )
                          goto LABEL_150;
                        v105 = *(_QWORD *)(*(_QWORD *)(v23 + 88) + 176LL);
                        if ( (*(_BYTE *)(v105 + 194) & 0x40) == 0 )
                          goto LABEL_150;
                        do
                          v105 = *(_QWORD *)(v105 + 232);
                        while ( (*(_BYTE *)(v105 + 194) & 0x40) != 0 );
                        v96 = *(_QWORD *)(v105 + 248);
                        goto LABEL_256;
                      }
                      v96 = *(_QWORD *)(v23 + 88);
                      if ( (*(_BYTE *)(v96 + 194) & 0x40) != 0 )
                      {
                        do
                          v96 = *(_QWORD *)(v96 + 232);
                        while ( (*(_BYTE *)(v96 + 194) & 0x40) != 0 );
LABEL_256:
                        v23 = *(_QWORD *)v96;
                      }
                    }
                    else
                    {
                      v23 = *(_QWORD *)&v145[1];
                    }
LABEL_150:
                    sub_67CC10(v23, v64, 0);
LABEL_151:
                    sub_8238B0(qword_4D039E8, "\"", 1);
                    if ( dword_4F073CC[0] )
                    {
                      v67 = (_QWORD *)qword_4D039E8;
                      v68 = *(_QWORD *)(qword_4D039E8 + 16);
                      if ( (unsigned __int64)(v68 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
                      {
                        sub_823810(qword_4D039E8);
                        v68 = v67[2];
                      }
                      *(_BYTE *)(v67[4] + v68) = 27;
                      v69 = v67[2];
                      v70 = v69 + 1;
                      v67[2] = v69 + 1;
                      if ( (unsigned __int64)(v69 + 2) > v67[1] )
                      {
                        sub_823810(v67);
                        v70 = v67[2];
                      }
                      *(_BYTE *)(v67[4] + v70) = 1;
                      ++v67[2];
                    }
                    if ( v5[2].m128i_i8[1] )
                    {
                      v71 = qword_4F04C68[0] + 776LL * v5[1].m128i_i32[2];
                      v72 = *(_QWORD *)(v71 + 376);
                      si128.m128i_i64[0] = v72;
                      if ( v72 )
                      {
                        if ( *(_BYTE *)(v72 + 8) != 3 || (sub_72F220(&si128), si128.m128i_i64[0]) )
                        {
                          sub_8238B0(qword_4D039E8, " ", 1);
                          v73 = *(_QWORD *)si128.m128i_i64[0];
                          si128.m128i_i64[0] = v73;
                          if ( v73 && (*(_BYTE *)(v73 + 8) != 3 || (sub_72F220(&si128), si128.m128i_i64[0])) )
                          {
                            v74 = sub_67C860(1487);
                            sub_823910(qword_4D039E8, v74);
                          }
                          else
                          {
                            v104 = sub_67C860(1486);
                            sub_823910(qword_4D039E8, v104);
                          }
                          sub_8238B0(qword_4D039E8, " ", 1);
                          sub_7477E0(*(_QWORD *)(v71 + 376), 0, &qword_4CFFDC0);
                        }
                      }
                    }
                    if ( v5[2].m128i_i8[0] )
                    {
                      if ( !v49
                        || (*(_BYTE *)(v49 + 193) & 0x10) == 0
                        || (*(_BYTE *)(v49 + 206) & 2) != 0 && *(_DWORD *)(v138 + 48) )
                      {
                        v91 = sub_67C860(1489);
                        v92 = sub_67C860(1488);
                        sub_67C870((unsigned int *)(v138 + 48), a1, v92, ")", v91);
                      }
                      else
                      {
                        v75 = sub_67C860(2322);
                        sub_823910(qword_4D039E8, v75);
                      }
                    }
                    if ( v5[2].m128i_i8[2] )
                    {
                      if ( *(_DWORD *)(v138 + 40) != -1 )
                      {
                        v76 = sub_880F80(v138);
                        if ( !unk_4D03FE8 || v76 != unk_4D03FF0 )
                        {
                          sub_8238B0(qword_4D039E8, " (", 2);
                          v77 = sub_67C860(1167);
                          sub_823910(qword_4D039E8, v77);
                          v78 = (const char *)sub_723640(*(_QWORD *)(v76 + 176), 0, 1);
                          v79 = strlen(v78);
                          sub_8238B0(qword_4D039E8, v78, v79);
                          sub_8238B0(qword_4D039E8, ")", 1);
                        }
                      }
                    }
                    byte_4CFFE55 = v144;
                    return v144;
                  }
                }
                v128 = 1;
                goto LABEL_127;
              }
              *(_QWORD *)&v145[1] = v138;
            }
            else
            {
              if ( *(_QWORD *)&v145[1] )
              {
                v90 = byte_4CFFE51;
                byte_4CFFE51 = v134;
                v123 = 1;
                v135 = v90;
                goto LABEL_232;
              }
              if ( v122 )
              {
                v109 = byte_4CFFE51;
                byte_4CFFE51 = v134;
                *(_QWORD *)&v145[1] = v122;
                v135 = v109;
                v123 = 1;
                v128 = 0;
                goto LABEL_127;
              }
              if ( (*(_DWORD *)(v138 + 80) & 0x41000) != 0 )
              {
                v108 = byte_4CFFE51;
                byte_4CFFE51 = v134;
                v123 = 1;
                v135 = v108;
                *(_QWORD *)&v145[1] = v138;
                goto LABEL_232;
              }
              *(_QWORD *)&v145[1] = v23;
              v123 = 1;
            }
            v95 = byte_4CFFE51;
            byte_4CFFE51 = v58;
            v135 = v95;
            if ( v49 )
              goto LABEL_124;
LABEL_232:
            v128 = 0;
            goto LABEL_127;
          case 20:
            v49 = *(_QWORD *)(*(_QWORD *)(v23 + 88) + 176LL);
            for ( i = *(_QWORD *)(v49 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
              ;
            v139 = v5[1].m128i_u8[14];
            if ( (*(_BYTE *)(v49 + 206) & 2) != 0 )
            {
              v145[0] = 0;
              v52 = 0;
              v48 = 0;
              v49 = 0;
              v53 = 1485;
              *(_QWORD *)&v145[1] = **(_QWORD **)(v23 + 64);
              goto LABEL_111;
            }
            v53 = 1485;
LABEL_210:
            v52 = v5[1].m128i_i8[13];
            if ( v52 )
            {
              memset(v145, 0, sizeof(v145));
              v52 = 0;
              v48 = 0;
              goto LABEL_112;
            }
            memset(v145, 0, sizeof(v145));
            v48 = 0;
            goto LABEL_187;
          case 21:
            v52 = 0;
            v48 = 0;
            v49 = 0;
            memset(v145, 0, sizeof(v145));
            v53 = 2750;
            v139 = 0;
            goto LABEL_111;
          case 22:
            v52 = 0;
            v48 = 0;
            v49 = 0;
            memset(v145, 0, sizeof(v145));
            v53 = 3050;
            v139 = 0;
            goto LABEL_111;
          case 23:
            v52 = 0;
            v48 = 0;
            v49 = 0;
            memset(v145, 0, sizeof(v145));
            v53 = 1482;
            v139 = 0;
            goto LABEL_111;
          default:
            goto LABEL_23;
        }
      }
LABEL_41:
      v9 = *a3;
      if ( *a3 )
      {
        do
        {
          if ( v9 != 113 )
            goto LABEL_23;
          v9 = *++a3;
        }
        while ( v9 );
        v10 = (_QWORD *)qword_4D039E8;
        v11 = *(_QWORD *)(qword_4D039E8 + 16);
        if ( (unsigned __int64)(v11 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
        {
          sub_823810(qword_4D039E8);
          v10 = (_QWORD *)qword_4D039E8;
          v11 = *(_QWORD *)(qword_4D039E8 + 16);
        }
        *(_BYTE *)(v10[4] + v11) = 34;
        v12 = v10[2];
        v13 = dword_4F073CC[0];
        v14 = v12 + 1;
        v10[2] = v12 + 1;
        if ( v13 )
        {
          if ( (unsigned __int64)(v12 + 2) > v10[1] )
          {
            sub_823810(v10);
            v14 = v10[2];
          }
          *(_BYTE *)(v10[4] + v14) = 27;
          v15 = v10[2];
          v16 = v15 + 1;
          v10[2] = v15 + 1;
          if ( (unsigned __int64)(v15 + 2) > v10[1] )
          {
            sub_823810(v10);
            v16 = v10[2];
          }
          *(_BYTE *)(v10[4] + v16) = 6;
          ++v10[2];
          v10 = (_QWORD *)qword_4D039E8;
        }
        v17 = (const char *)v5[1].m128i_i64[0];
        v18 = strlen(v17);
        sub_8238B0(v10, v17, v18);
        v19 = (_QWORD *)qword_4D039E8;
        if ( dword_4F073CC[0] )
        {
          v20 = *(_QWORD *)(qword_4D039E8 + 16);
          if ( (unsigned __int64)(v20 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
          {
            sub_823810(qword_4D039E8);
            v20 = v19[2];
          }
          *(_BYTE *)(v19[4] + v20) = 27;
          v21 = v19[2];
          v22 = v21 + 1;
          v19[2] = v21 + 1;
          if ( (unsigned __int64)(v21 + 2) > v19[1] )
          {
            sub_823810(v19);
            v22 = v19[2];
          }
          *(_BYTE *)(v19[4] + v22) = 1;
          ++v19[2];
          v19 = (_QWORD *)qword_4D039E8;
        }
        result = v19[2];
        if ( (unsigned __int64)(result + 1) > v19[1] )
        {
          sub_823810(v19);
          v19 = (_QWORD *)qword_4D039E8;
          result = *(_QWORD *)(qword_4D039E8 + 16);
        }
        *(_BYTE *)(v19[4] + result) = 34;
        ++v19[2];
      }
      else
      {
        v42 = (const char *)v5[1].m128i_i64[0];
        v41 = strlen(v42);
        return sub_8238B0(qword_4D039E8, v42, v41);
      }
      return result;
    default:
      goto LABEL_23;
  }
}
