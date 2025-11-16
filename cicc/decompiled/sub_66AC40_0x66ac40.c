// Function: sub_66AC40
// Address: 0x66ac40
//
__int64 __fastcall sub_66AC40(
        __int64 a1,
        unsigned __int64 a2,
        _BOOL4 a3,
        unsigned int a4,
        int a5,
        __int64 *a6,
        _DWORD *a7,
        _DWORD *a8,
        __int64 a9)
{
  __int64 v10; // r13
  unsigned __int8 v11; // di
  _QWORD *v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  char v16; // al
  __int64 v17; // rcx
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  int v21; // eax
  __int64 v22; // rdi
  __int64 v23; // rsi
  unsigned __int64 v24; // rcx
  unsigned int v25; // r13d
  __int64 v26; // rdx
  __int64 v27; // rax
  char v28; // dl
  __int64 v29; // rdi
  __int64 v30; // rax
  _BOOL4 v31; // r9d
  __int64 v32; // rax
  _BOOL8 v33; // r9
  __int64 v34; // rbx
  _QWORD *v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // rax
  unsigned int v39; // eax
  _BOOL4 v40; // eax
  int v41; // r9d
  unsigned __int64 v42; // rdi
  int v43; // r14d
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned int v46; // eax
  _BOOL4 v47; // ebx
  char v48; // bl
  int v49; // edx
  __int64 v50; // rdi
  __int64 *v51; // rax
  unsigned __int16 v52; // dx
  unsigned __int16 *v53; // rax
  unsigned __int64 v54; // rdx
  __m128i *v55; // r10
  int v56; // r9d
  int v57; // eax
  unsigned int v58; // r13d
  __m128i *v59; // rax
  __int64 v60; // rax
  __int64 v62; // r8
  bool v63; // zf
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // r14
  char v67; // al
  __int64 v68; // rax
  __int64 v69; // rdi
  char v70; // al
  __int64 v71; // rdx
  __int64 *v72; // rax
  __int64 v73; // r14
  __int64 *v74; // rax
  int v75; // eax
  __m128i v76; // xmm7
  __m128i v77; // xmm6
  __m128i v78; // xmm7
  int v79; // r9d
  __int64 v80; // rdx
  __m128i *v81; // rax
  int v82; // r9d
  char v83; // al
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // rcx
  _QWORD *v89; // rdx
  __m128i v90; // xmm6
  __m128i v91; // xmm7
  __m128i v92; // xmm6
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // rdx
  char v96; // al
  __m128i *v97; // rax
  __int64 v98; // rax
  __int64 v99; // rcx
  __int64 i; // rax
  __int64 v101; // r14
  __m128i *v102; // rax
  unsigned int v103; // r10d
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // r14
  char v107; // dl
  unsigned __int8 v108; // al
  __int64 v109; // rax
  __int64 v110; // rdx
  _QWORD *v111; // rdx
  _QWORD *v112; // rax
  int v113; // eax
  __int64 v114; // rdx
  __int64 v115; // rax
  __int64 v116; // rdx
  __int64 v117; // rax
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  _QWORD *v124; // rdx
  int v125; // eax
  _BOOL4 v126; // [rsp+8h] [rbp-188h]
  int v127; // [rsp+Ch] [rbp-184h]
  __int64 v128; // [rsp+10h] [rbp-180h]
  unsigned int v129; // [rsp+10h] [rbp-180h]
  __int64 v130; // [rsp+18h] [rbp-178h]
  _BOOL4 v131; // [rsp+18h] [rbp-178h]
  int v132; // [rsp+18h] [rbp-178h]
  _QWORD *v133; // [rsp+20h] [rbp-170h]
  __int64 v134; // [rsp+20h] [rbp-170h]
  int v136; // [rsp+2Ch] [rbp-164h]
  __int64 v137; // [rsp+38h] [rbp-158h]
  _BOOL4 v139; // [rsp+48h] [rbp-148h]
  unsigned __int8 v140; // [rsp+50h] [rbp-140h]
  __int64 v141; // [rsp+50h] [rbp-140h]
  __int64 v142; // [rsp+50h] [rbp-140h]
  __int64 v143; // [rsp+50h] [rbp-140h]
  unsigned int v144; // [rsp+58h] [rbp-138h]
  int v145; // [rsp+58h] [rbp-138h]
  unsigned int v146; // [rsp+60h] [rbp-130h]
  _BOOL4 v147; // [rsp+64h] [rbp-12Ch]
  int v148; // [rsp+64h] [rbp-12Ch]
  unsigned __int8 v149; // [rsp+70h] [rbp-120h]
  int v150; // [rsp+70h] [rbp-120h]
  __int64 v151; // [rsp+70h] [rbp-120h]
  __int64 v152; // [rsp+70h] [rbp-120h]
  __int64 v153; // [rsp+70h] [rbp-120h]
  __int64 v154; // [rsp+70h] [rbp-120h]
  __m128i *v155; // [rsp+70h] [rbp-120h]
  __int64 v156; // [rsp+70h] [rbp-120h]
  __int64 v157; // [rsp+70h] [rbp-120h]
  __int64 v158; // [rsp+70h] [rbp-120h]
  int v159; // [rsp+70h] [rbp-120h]
  int v160; // [rsp+70h] [rbp-120h]
  int v161; // [rsp+70h] [rbp-120h]
  __int64 v162; // [rsp+70h] [rbp-120h]
  __m128i *v163; // [rsp+70h] [rbp-120h]
  unsigned int v164; // [rsp+80h] [rbp-110h]
  unsigned int v165; // [rsp+88h] [rbp-108h] BYREF
  _BOOL4 v166[2]; // [rsp+8Ch] [rbp-104h] BYREF
  __int64 v167; // [rsp+94h] [rbp-FCh] BYREF
  int v168; // [rsp+9Ch] [rbp-F4h] BYREF
  int v169; // [rsp+A0h] [rbp-F0h] BYREF
  unsigned int v170; // [rsp+A4h] [rbp-ECh] BYREF
  int v171; // [rsp+A8h] [rbp-E8h] BYREF
  int v172; // [rsp+ACh] [rbp-E4h] BYREF
  __int64 v173; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v174; // [rsp+B8h] [rbp-D8h] BYREF
  __m128i v175; // [rsp+C0h] [rbp-D0h] BYREF
  __m128i v176; // [rsp+D0h] [rbp-C0h]
  __m128i v177; // [rsp+E0h] [rbp-B0h]
  __m128i v178; // [rsp+F0h] [rbp-A0h]
  _BYTE v179[88]; // [rsp+100h] [rbp-90h] BYREF
  __int64 v180; // [rsp+158h] [rbp-38h] BYREF

  v166[0] = a3;
  v165 = a4;
  v170 = dword_4F04C5C;
  v167 = 0;
  v128 = a2 & 0x40;
  v10 = (a2 >> 24) & 1;
  v147 = (a2 & 0x40) != 0;
  v168 = 0;
  v137 = a2 & 0x10000;
  v169 = 0;
  v126 = (a2 & 0x10000) != 0;
  v171 = 0;
  v130 = a2 & 0x20000;
  v139 = (a2 & 0x20000) != 0;
  *a7 = 0;
  *a8 = 0;
  v173 = *(_QWORD *)&dword_4F063F8;
  memset(v179, 0, sizeof(v179));
  *(_QWORD *)&v179[32] = *(_QWORD *)&dword_4F063F8;
  switch ( word_4F06418[0] )
  {
    case 0x68u:
      v149 = 5;
      v140 = 11;
      break;
    case 0x97u:
      v149 = 4;
      v140 = 9;
      break;
    case 0x65u:
      v149 = 4;
      v140 = 10;
      break;
    default:
      v149 = 4;
      v11 = 4;
      v140 = 9;
LABEL_5:
      v174 = *(_QWORD *)&dword_4F063F8;
      *(_QWORD *)&v179[16] = *(_QWORD *)&dword_4F063F8;
      *a7 = 1;
      if ( !dword_4F077BC
        || unk_4F04C50 | v137
        || dword_4F04C40 == -1
        || (v27 = qword_4F04C68[0] + 776LL * (int)dword_4F04C40, v28 = *(_BYTE *)(v27 + 7), (v28 & 8) != 0) )
      {
        v12 = sub_668EE0(v11, &v175, &v165, v139, v166, v147, v10, (int *)&v170, &v169, &v171, (__int64)v179);
      }
      else
      {
        if ( dword_4F077C4 == 2 )
          *(_BYTE *)(v27 + 7) = v28 | 8;
        v12 = sub_668EE0(v11, &v175, &v165, v139, v166, v147, v10, (int *)&v170, &v169, &v171, (__int64)v179);
        if ( dword_4F077C4 == 2 )
        {
          v29 = (int)dword_4F04C40;
          v30 = 776LL * (int)dword_4F04C40;
          *(_BYTE *)(qword_4F04C68[0] + v30 + 7) &= ~8u;
          if ( *(_QWORD *)(qword_4F04C68[0] + v30 + 456) )
            sub_8845B0(v29);
        }
      }
      v133 = 0;
      if ( !v12 )
        goto LABEL_40;
      if ( v165 )
      {
        if ( dword_4F077BC )
        {
          if ( (*((_BYTE *)v12 + 84) & 2) != 0 )
            nullsub_2();
        }
        else if ( (v176.m128i_i8[0] & 1) != 0 && *(_BYTE *)(v176.m128i_i64[1] + 80) == 24 )
        {
          v85 = *(_QWORD *)(v175.m128i_i64[0] + 8);
          if ( (v176.m128i_i8[2] & 2) != 0 || !v177.m128i_i64[0] )
            sub_6851A0(470, &v175.m128i_u64[1], v85);
          else
            sub_686A10(742, &v175.m128i_u64[1], v85, *(_QWORD *)v177.m128i_i64[0]);
          v176.m128i_i8[1] |= 0x20u;
          v12 = 0;
          v176.m128i_i64[1] = 0;
          v133 = 0;
          goto LABEL_40;
        }
      }
      else
      {
        if ( (a2 & 0x30000) == 0 )
          goto LABEL_28;
        if ( word_4F06418[0] == 1 )
        {
          if ( dword_4F077C4 != 2 || (unk_4D04A11 & 2) == 0 && !(unsigned int)sub_7C0F00(0, 0) || (unk_4D04A12 & 1) == 0 )
            goto LABEL_28;
        }
        else if ( word_4F06418[0] == 34
               || word_4F06418[0] == 27
               || dword_4F077C4 == 2
               && (word_4F06418[0] == 33
                || dword_4D04474 && word_4F06418[0] == 52
                || dword_4D0485C && word_4F06418[0] == 25
                || word_4F06418[0] == 156) )
        {
          goto LABEL_28;
        }
        if ( (v176.m128i_i32[0] & 0x20001) == 0x20001 )
        {
          if ( v176.m128i_i64[1] )
          {
            if ( !dword_4F077BC && (unsigned __int8)(*(_BYTE *)(v176.m128i_i64[1] + 80) - 4) <= 1u )
            {
              v13 = *(_QWORD *)(v176.m128i_i64[1] + 88);
              if ( v13 )
              {
                if ( (*(_BYTE *)(v13 + 177) & 0x30) == 0x10 )
                {
                  v14 = *(_QWORD *)(v176.m128i_i64[1] + 64);
                  if ( (v176.m128i_i8[2] & 2) != 0 )
                  {
                    if ( v14 == v177.m128i_i64[0] )
                      goto LABEL_28;
                    if ( v177.m128i_i64[0] )
                    {
                      if ( v14 )
                      {
                        if ( dword_4F07588 )
                        {
                          v15 = *(_QWORD *)(v14 + 32);
                          if ( *(_QWORD *)(v177.m128i_i64[0] + 32) == v15 )
                          {
                            if ( v15 )
                              goto LABEL_28;
                          }
                        }
                      }
                    }
                  }
                  else if ( !v14 )
                  {
                    goto LABEL_28;
                  }
                  sub_685360(1431, &v175.m128i_u64[1]);
                }
              }
            }
          }
        }
      }
LABEL_28:
      *(_QWORD *)a1 = v12;
      v16 = *((_BYTE *)v12 + 80);
      if ( v16 == 3 )
      {
        v83 = *((_BYTE *)v12 + 81);
        if ( (v83 & 0x40) != 0 || (v133 = 0, *(_BYTE *)(v12[11] + 140LL) == 14) )
        {
          if ( v165 || (v133 = 0, (v83 & 0x10) != 0) )
          {
            v133 = 0;
            if ( v83 >= 0 )
            {
              v84 = sub_7CFE40(v12[11]);
              v12 = *(_QWORD **)v84;
              *(_BYTE *)(v84 + 140) = v140;
              *((_BYTE *)v12 + 80) = v149;
            }
          }
        }
        goto LABEL_40;
      }
      v133 = 0;
      if ( v16 != v149 )
      {
        v17 = v12[11];
        if ( (unsigned __int8)(v16 - 4) > 1u )
          goto LABEL_31;
        if ( (*(_BYTE *)(v17 + 177) & 0x30) != 0x30 )
        {
          v89 = *(_QWORD **)(v12[12] + 72LL);
          if ( v89 )
          {
LABEL_349:
            sub_6854C0(467, &v173, v89);
            v133 = v12;
            v12 = 0;
            v176.m128i_i8[1] |= 0x20u;
            v176.m128i_i64[1] = 0;
            goto LABEL_40;
          }
LABEL_31:
          v18 = qword_4F06C80;
          while ( 1 )
          {
            v19 = *v18;
            if ( *v18 == v17 )
              break;
            if ( v17 )
            {
              if ( v19 )
              {
                if ( dword_4F07588 )
                {
                  v20 = *(_QWORD *)(v17 + 32);
                  if ( *(_QWORD *)(v19 + 32) == v20 )
                  {
                    if ( v20 )
                      break;
                  }
                }
              }
            }
            if ( ++v18 == &qword_4F06C80[11] )
            {
              v133 = 0;
              goto LABEL_40;
            }
          }
          v89 = v12;
          goto LABEL_349;
        }
      }
LABEL_40:
      v136 = 1;
      v164 = (v176.m128i_i8[1] & 0x20) != 0;
      v21 = dword_4F077C4;
      goto LABEL_41;
  }
  sub_7B8B50(&v180, a2, *(_QWORD *)&dword_4F063F8, 0);
  *(_QWORD *)(a1 + 216) = sub_5CC190(2);
  v52 = word_4F06418[0];
  if ( word_4F06418[0] == 76 )
  {
    sub_6851C0(819, &dword_4F063F8);
    sub_7B8B50(819, &dword_4F063F8, v87, v88);
    v52 = word_4F06418[0];
  }
  if ( v52 != 185 )
  {
LABEL_152:
    if ( v52 == 1 )
      goto LABEL_155;
    if ( v52 == 146 )
    {
      if ( (unsigned __int16)sub_7BE840(0, 0) == 1 )
        goto LABEL_155;
      v52 = word_4F06418[0];
    }
    if ( v52 == 156 )
    {
LABEL_155:
      v11 = v149;
      goto LABEL_5;
    }
    v174 = v173;
    v175.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v176 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v177 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v176.m128i_i8[1] |= 0x20u;
    v175.m128i_i64[1] = *(_QWORD *)dword_4F07508;
    v178 = _mm_loadu_si128(&xmmword_4F06660[3]);
    if ( (a2 & 0x1000040) != 0 )
      goto LABEL_290;
    v21 = dword_4F077C4;
    if ( v52 == 73 )
    {
      v164 = dword_4F077BC;
      if ( dword_4F077BC )
      {
        if ( qword_4F077A8 <= 0x765Bu && v140 == 9 )
        {
          v136 = 0;
          v12 = 0;
          v164 = 0;
          v133 = 0;
          v140 = 10;
        }
        else
        {
          v136 = 0;
          v12 = 0;
          v164 = 0;
          v133 = 0;
        }
      }
      else
      {
        v136 = 0;
        v12 = 0;
        v133 = 0;
      }
      goto LABEL_41;
    }
    goto LABEL_267;
  }
  if ( dword_4F077C4 == 2 )
  {
    sub_7C0F00(0, 0);
    v52 = word_4F06418[0];
    goto LABEL_152;
  }
  v174 = v173;
  v175.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v177 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v176 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v178 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v176.m128i_i8[1] |= 0x20u;
  v175.m128i_i64[1] = *(_QWORD *)dword_4F07508;
  if ( (a2 & 0x1000040) != 0 )
    goto LABEL_290;
  v21 = dword_4F077C4;
LABEL_267:
  if ( v21 != 2 )
  {
    if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) != 8 )
    {
      ++*(_BYTE *)(qword_4F061C8 + 81LL);
LABEL_270:
      sub_6851D0(110);
      v80 = qword_4F061C8;
      v21 = dword_4F077C4;
      if ( dword_4F077C4 == 2 )
        --*(_BYTE *)(qword_4F061C8 + 63LL);
      --*(_BYTE *)(v80 + 81);
      v12 = 0;
      v136 = 0;
      v164 = 1;
      v133 = 0;
LABEL_41:
      if ( v21 != 2 )
        goto LABEL_42;
      goto LABEL_143;
    }
    goto LABEL_290;
  }
  if ( v52 != 55 )
  {
    if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) != 8 )
    {
      v93 = qword_4F061C8;
      ++*(_BYTE *)(qword_4F061C8 + 81LL);
      ++*(_BYTE *)(v93 + 63);
      goto LABEL_270;
    }
LABEL_290:
    v12 = 0;
    sub_6851D0(40);
    v136 = 0;
    v164 = 1;
    v133 = 0;
    v21 = dword_4F077C4;
    goto LABEL_41;
  }
  v164 = dword_4F077BC;
  if ( dword_4F077BC )
  {
    v12 = 0;
    if ( qword_4F077A8 <= 0x765Bu && v140 == 9 )
    {
      v136 = 0;
      v164 = 0;
      v133 = 0;
      v140 = 10;
    }
    else
    {
      v136 = 0;
      v164 = 0;
      v133 = 0;
    }
  }
  else
  {
    v136 = 0;
    v12 = 0;
    v133 = 0;
  }
LABEL_143:
  if ( unk_4F07778 > 201102 || dword_4F07774 || HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x9EFBu )
    sub_66A730(v140, (_DWORD *)&v167 + 1, &v167, &v168);
LABEL_42:
  v22 = word_4F06418[0];
  if ( word_4F06418[0] != 17 )
  {
    v23 = v149;
    v25 = sub_667FD0(word_4F06418[0], v149, v147, v10);
    if ( !v25 )
    {
      v146 = 0;
      if ( !v12 )
      {
        v144 = 0;
        v34 = 0;
LABEL_100:
        v31 = v166[0];
        v127 = 0;
        v148 = 0;
        if ( v166[0] )
          v31 = word_4F06418[0] == 75;
        goto LABEL_102;
      }
      v148 = 0;
      goto LABEL_98;
    }
    if ( v12 )
    {
      v26 = v12[11];
      if ( (unsigned __int8)(*(_BYTE *)(v26 + 140) - 9) <= 2u )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v26 + 168) + 256LL) )
        {
          if ( *(_BYTE *)(v177.m128i_i64[0] + 140) == 14 && *(_QWORD *)(*(_QWORD *)(v177.m128i_i64[0] + 168) + 8LL) )
          {
            v23 = (__int64)&v174;
            sub_6854C0(551, &v174, v12);
          }
          else
          {
            v23 = (__int64)&v174;
            sub_686A10(1018, &v174, *(_QWORD *)(v175.m128i_i64[0] + 8), *(_QWORD *)v177.m128i_i64[0]);
          }
          v176.m128i_i8[1] |= 0x20u;
          v12 = 0;
          v176.m128i_i64[1] = 0;
        }
      }
    }
    v22 = a1;
    sub_643D30(a1);
    v146 = word_4D04898;
    if ( !word_4D04898 )
    {
      v148 = v25;
      goto LABEL_62;
    }
    v146 = dword_4D0488C | dword_4F077BC;
    if ( dword_4D0488C | dword_4F077BC )
    {
      v148 = v25;
      v146 = 0;
    }
    else
    {
      v148 = v25;
      v68 = unk_4F04C50;
      if ( unk_4F04C50 )
        goto LABEL_233;
    }
LABEL_62:
    if ( !v165 )
      goto LABEL_240;
LABEL_63:
    if ( !dword_4F077BC || qword_4F077A8 > 0x76BFu || (v176.m128i_i32[0] & 0x10001) != 0 )
    {
      v176.m128i_i8[1] |= 0x20u;
      v176.m128i_i64[1] = 0;
      v144 = 0;
LABEL_67:
      v127 = 0;
      v31 = 1;
      goto LABEL_68;
    }
    v23 = (__int64)&v173;
    v22 = 1377;
    sub_684B30(1377, &v173);
    v165 = 0;
LABEL_240:
    if ( !v12 )
    {
      v31 = 1;
      v144 = 0;
      v127 = 0;
      goto LABEL_68;
    }
    goto LABEL_98;
  }
  v23 = 0;
  if ( (unsigned __int16)sub_7BE840(0, 0) == 75 )
    word_4F06418[0] = 75;
  v22 = a1;
  sub_643D30(a1);
  if ( word_4D04898 && (v25 = dword_4D0488C | dword_4F077BC) == 0 )
  {
    v148 = 1;
    v146 = 1;
    v68 = unk_4F04C50;
    if ( unk_4F04C50 )
    {
LABEL_233:
      if ( (*(_BYTE *)(*(_QWORD *)(v68 + 32) + 193LL) & 2) != 0 )
      {
        if ( !(_DWORD)qword_4F077B4 || qword_4F077A0 <= 0x765Bu || (v22 = 5, !(unsigned int)sub_729F80(dword_4F063F8)) )
          v22 = 8;
        v23 = 2407;
        sub_684AA0(v22, 2407, &v173);
      }
      if ( v25 )
      {
        if ( !v165 )
          goto LABEL_240;
        goto LABEL_63;
      }
    }
  }
  else
  {
    v146 = 1;
    v148 = 1;
  }
  if ( !v12 )
  {
    v144 = 0;
    v34 = 0;
    v25 = 0;
    goto LABEL_338;
  }
  v25 = 0;
LABEL_98:
  v34 = 0;
  v144 = 0;
  if ( dword_4F077C4 != 2 )
    goto LABEL_99;
  v34 = v12[11];
  if ( *((_BYTE *)v12 + 80) == 3 )
  {
    if ( v25 )
      goto LABEL_67;
    goto LABEL_247;
  }
  v66 = v12[12];
  if ( v25 )
  {
    v67 = *((_BYTE *)v12 + 81);
    if ( (v67 & 2) != 0 )
    {
      if ( (v67 & 0x20) == 0 )
      {
        if ( v130 && (*(_BYTE *)(v34 + 178) & 1) == 0 )
          sub_686A30(8, 1449, &v174, v66 + 88);
        else
          sub_685920(&v174, v12, 8);
      }
      v176.m128i_i8[1] |= 0x20u;
      v176.m128i_i64[1] = 0;
      v133 = v12;
      v144 = 0;
      v164 = 1;
      goto LABEL_67;
    }
  }
  v22 = v12[11];
  v144 = sub_8D23B0(v22);
  if ( (*(_BYTE *)(v34 + 177) & 0x10) == 0 )
  {
    v144 = 0;
    goto LABEL_358;
  }
  if ( v130 )
  {
    if ( !v25 && word_4F06418[0] != 75 )
    {
      v139 = 1;
LABEL_247:
      v70 = *((_BYTE *)v12 + 81);
      goto LABEL_248;
    }
    if ( (*(_BYTE *)(v34 + 178) & 1) != 0 )
    {
      v139 = 1;
      v144 = 1;
      *a7 = 0;
      goto LABEL_358;
    }
    v23 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
    if ( *((_DWORD *)v12 + 10) == *(_DWORD *)v23 )
      goto LABEL_556;
    if ( (*((_BYTE *)v12 + 81) & 0x10) == 0 && !v12[8]
      || (v22 = (__int64)v12, !(unsigned int)sub_85ED80(v12, v23))
      && (v22 = (__int64)v12, !(unsigned int)sub_880920(v12, v23, v114)) )
    {
      v23 = (__int64)&v174;
      v22 = 503;
      sub_6854C0(503, &v174, v12);
      v176.m128i_i8[1] |= 0x20u;
      v176.m128i_i64[1] = 0;
      if ( (*(_BYTE *)(v34 + 177) & 0x20) == 0 )
      {
        *(_BYTE *)(v34 + 178) |= 1u;
        if ( unk_4D04734 == 3 )
        {
          v22 = v34;
          v12 = 0;
          sub_66A6A0(v34);
          v139 = 0;
        }
        else
        {
          v139 = 0;
          v12 = 0;
        }
        v144 = 1;
        v164 = 1;
        goto LABEL_99;
      }
      v139 = 0;
      v12 = 0;
      v164 = 1;
      goto LABEL_491;
    }
    if ( (*(_BYTE *)(v34 + 178) & 1) == 0 )
    {
LABEL_556:
      v23 = (__int64)&v174;
      v22 = (__int64)v12;
      sub_899850(v12, &v174);
    }
    if ( (*(_BYTE *)(v34 + 177) & 0x20) == 0 )
    {
      if ( !(v164 | v144) )
      {
        v23 = 1449;
        v22 = 8;
        sub_686A30(8, 1449, &v174, v66 + 88);
        v164 = 0;
        v139 = 1;
        v144 = 1;
        goto LABEL_358;
      }
      *(_BYTE *)(v34 + 178) |= 1u;
      v22 = (__int64)v12;
      v139 = 1;
      *(_QWORD *)(v66 + 120) = sub_8807C0(v12);
      v144 = 1;
      if ( unk_4D04734 != 3 )
        goto LABEL_358;
      v22 = v34;
      sub_66A6A0(v34);
LABEL_494:
      v144 = 1;
      goto LABEL_358;
    }
    v139 = 1;
LABEL_491:
    *(_BYTE *)(v34 + 178) |= 1u;
    if ( dword_4F04C44 != -1
      || (v23 = (__int64)qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0) )
    {
      *(_BYTE *)(v34 + 177) |= 0x80u;
    }
    if ( !v12 )
    {
      v144 = 1;
      goto LABEL_99;
    }
    goto LABEL_494;
  }
  v70 = *((_BYTE *)v12 + 81);
  v22 = v70 & 0x10;
  if ( !v25 )
  {
    v24 = (unsigned __int64)word_4F06418;
    if ( word_4F06418[0] == 17 || (v139 = 0, word_4F06418[0] == 75) )
    {
      if ( !v128 && (*(_BYTE *)(a1 + 122) & 0x20) == 0 )
      {
        v23 = v165 | v126;
        if ( !(v165 | v126) )
          goto LABEL_506;
      }
      v139 = 0;
    }
LABEL_248:
    v144 = 0;
    if ( (v70 & 0x10) == 0 )
      goto LABEL_99;
LABEL_249:
    v24 = (unsigned __int64)&dword_4F04C64;
    v23 = qword_4F04C68[0];
    v71 = qword_4F04C68[0] + 776LL * dword_4F04C64;
LABEL_250:
    if ( *(_BYTE *)(v71 + 4) != 6
      || (v104 = v12[8], v22 = *(_QWORD *)(v71 + 208), v104 != v22)
      && (!v104 || !v22 || !dword_4F07588 || (v105 = *(_QWORD *)(v104 + 32), *(_QWORD *)(v22 + 32) != v105) || !v105) )
    {
      if ( v25 )
      {
        v72 = (__int64 *)v12[8];
        v73 = *v72;
        if ( (*(_BYTE *)(*v72 + 81) & 0x10) != 0 )
        {
          do
          {
            v74 = *(__int64 **)(v73 + 64);
            v73 = *v74;
          }
          while ( (*(_BYTE *)(*v74 + 81) & 0x10) != 0 );
        }
        if ( *(_DWORD *)(v73 + 40) != *(_DWORD *)v71 )
        {
          if ( *(_QWORD *)(v73 + 64) )
          {
            v22 = v73;
            v23 += 776LL * (int)dword_4F04C5C;
            v75 = sub_85ED80(v73, v23);
            v24 = (unsigned __int64)&dword_4F04C64;
            if ( v75 )
            {
              v22 = *(_QWORD *)(v73 + 64);
              v23 = 0;
              sub_864230(v22, 0);
              v24 = (unsigned __int64)&dword_4F04C64;
              v127 = 1;
              v129 = 1;
              v170 = dword_4F04C64;
              goto LABEL_104;
            }
          }
          if ( !dword_4F077BC
            || unk_4F04C50
            || (v23 = (__int64)qword_4F04C68, v109 = qword_4F04C68[0] + 776LL * dword_4F04C64, *(_BYTE *)(v109 + 4) != 6)
            || (v110 = v12[11], v23 = *(unsigned __int8 *)(v110 + 140), (unsigned __int8)(v23 - 9) <= 2u)
            && (v24 = *(_QWORD *)(v110 + 168), *(_QWORD *)(v24 + 256))
            || (v24 = (unsigned int)*((unsigned __int8 *)v12 + 80) - 4, (unsigned __int8)(*((_BYTE *)v12 + 80) - 4) <= 1u)
            && (*(_BYTE *)(v110 + 177) & 0x10) != 0 )
          {
LABEL_259:
            v127 = 0;
            v129 = 0;
            if ( !v139 )
            {
              sub_6854C0(551, &v174, v12);
              v76 = _mm_loadu_si128(&xmmword_4F06660[1]);
              v175 = _mm_loadu_si128(xmmword_4F06660);
              v77 = _mm_loadu_si128(&xmmword_4F06660[2]);
              v176 = v76;
              v78 = _mm_loadu_si128(&xmmword_4F06660[3]);
              v177 = v77;
              v178 = v78;
              v176.m128i_i8[1] |= 0x20u;
              v175.m128i_i64[1] = *(_QWORD *)dword_4F07508;
              goto LABEL_67;
            }
LABEL_104:
            v44 = v12[11];
            if ( *((_BYTE *)v12 + 80) == 3 )
            {
              if ( *(_BYTE *)(v44 + 140) == 14 )
              {
                v132 = 0;
                v43 = 0;
              }
              else
              {
                v23 = (__int64)v12;
                v22 = 4;
                v43 = 0;
                sub_8767A0(4, v12, &v175.m128i_u64[1], 1);
                v132 = 0;
                *a7 = 0;
              }
              goto LABEL_159;
            }
            v45 = *(_QWORD *)(v44 + 168);
            v132 = *(_BYTE *)(v44 + 89) & 1;
            if ( !v144 || (v43 = 0, !*a7) )
            {
              *(_BYTE *)(a1 + 127) |= 0x40u;
              v43 = 1;
            }
            v46 = v165;
            v47 = 0;
            if ( !(unk_4D047D0 | v165) )
            {
              v48 = *((_BYTE *)v12 + 83);
              *((_BYTE *)v12 + 83) = v48 & 0xBF;
              v47 = (v48 & 0x40) != 0;
            }
            if ( !v25 )
            {
              if ( v46 )
              {
                if ( v171 )
                {
LABEL_212:
                  v50 = 257;
                  goto LABEL_213;
                }
LABEL_299:
                if ( word_4F06418[0] == 75 || word_4F06418[0] == 17 )
                {
                  if ( v46 | v166[0] | v144 )
                    goto LABEL_302;
                  if ( *(_QWORD *)(a1 + 216) )
                    goto LABEL_125;
                }
                if ( !v47 )
                {
                  v23 = (__int64)v12;
                  v22 = 4;
                  v158 = v44;
                  sub_8767A0(4, v12, &v175.m128i_u64[1], 1);
                  *a7 = 0;
                  v34 = v158;
                  goto LABEL_160;
                }
LABEL_302:
                if ( v46 )
                  goto LABEL_212;
LABEL_125:
                if ( (v176.m128i_i8[0] & 1) != 0 )
                {
                  v50 = 1;
                  if ( v46 | (v139 || v126) || word_4F06418[0] != 75 )
                  {
LABEL_213:
                    v23 = (__int64)v12;
                    v151 = v44;
                    sub_8756F0(v50, v12, &v175.m128i_u64[1], 0);
                    v22 = (__int64)v12;
                    sub_86F690(v12);
                    v62 = v151;
                    v63 = (v171 | v47) == 0;
                    v34 = v151;
                    if ( !v63 )
                    {
                      v64 = 0;
                      if ( dword_4F04C64 != -1
                        && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0
                        && dword_4F077C4 == 2
                        && (*(_BYTE *)(v151 - 8) & 1) != 0
                        && (v176.m128i_i8[2] & 0x40) == 0 )
                      {
                        v65 = sub_7CAFF0(&v175, v151, 0);
                        v62 = v151;
                        v64 = v65;
                      }
                      v22 = v62;
                      v152 = v62;
                      v23 = 0;
                      sub_86A320(v62, 0, v64, a5 == 0 ? 16 : 80);
                      v34 = v152;
                    }
                    goto LABEL_160;
                  }
                  v51 = *(__int64 **)(a1 + 216);
                  if ( !v51 )
                  {
LABEL_131:
                    v141 = v44;
                    sub_684AA0(((unsigned int)qword_4F077B4 | dword_4D04964) == 0 ? 5 : 8, 283, &v175.m128i_u64[1]);
                    v50 = 1;
                    v44 = v141;
                    goto LABEL_213;
                  }
                  while ( *((_BYTE *)v51 + 9) != 3 )
                  {
                    v51 = (__int64 *)*v51;
                    if ( !v51 )
                      goto LABEL_131;
                  }
                }
                v50 = 1;
                goto LABEL_213;
              }
              goto LABEL_119;
            }
            *(_BYTE *)(v44 + 140) = v140;
            if ( unk_4F04C48 != -1 && !v139 && (*((_BYTE *)v12 + 81) & 0x20) == 0 )
            {
              v23 = v140;
              v134 = v45;
              v156 = v44;
              sub_896D00(v12, v140);
              v45 = v134;
              v44 = v156;
            }
            v22 = (unsigned int)dword_4D04964;
            if ( !dword_4D04964 || (v176.m128i_i8[0] & 1) == 0 || v139 || v126 )
              goto LABEL_117;
            v95 = v12[8];
            v96 = v176.m128i_i8[2] & 2;
            if ( (*((_BYTE *)v12 + 81) & 0x10) != 0 )
            {
              if ( v96 )
              {
                if ( v95 != v177.m128i_i64[0] )
                {
                  if ( !v177.m128i_i64[0] )
                    goto LABEL_383;
                  if ( !v95 )
                    goto LABEL_383;
                  v23 = dword_4F07588;
                  if ( !dword_4F07588 )
                    goto LABEL_383;
                  v121 = *(_QWORD *)(v177.m128i_i64[0] + 32);
                  if ( *(_QWORD *)(v95 + 32) != v121 || !v121 )
                    goto LABEL_383;
                }
              }
              else if ( v95 )
              {
LABEL_383:
                v23 = 1431;
                v142 = v45;
                v22 = unk_4F07471;
                v157 = v44;
                sub_685260(unk_4F07471, 1431, &v175.m128i_u64[1], v12[11]);
                v46 = v165;
                v44 = v157;
                v45 = v142;
                goto LABEL_118;
              }
            }
            else
            {
              if ( v96 )
              {
                if ( !v95 )
                  goto LABEL_117;
              }
              else
              {
                if ( v95 == v177.m128i_i64[0] )
                  goto LABEL_117;
                if ( v95 )
                {
                  if ( v177.m128i_i64[0] )
                  {
                    if ( dword_4F07588 )
                    {
                      v122 = *(_QWORD *)(v177.m128i_i64[0] + 32);
                      if ( *(_QWORD *)(v95 + 32) == v122 )
                      {
                        if ( v122 )
                          goto LABEL_117;
                      }
                    }
                  }
                }
              }
              v22 = (__int64)v12;
              v143 = v45;
              v162 = v44;
              v113 = sub_880920(v12, v23, v95);
              v44 = v162;
              v45 = v143;
              if ( !v113 )
              {
                v23 = 1432;
                v22 = unk_4F07471;
                sub_685260(unk_4F07471, 1432, &v175.m128i_u64[1], v12[11]);
                v46 = v165;
                v44 = v162;
                v45 = v143;
LABEL_118:
                if ( v46 )
                {
                  v69 = 259;
                  goto LABEL_244;
                }
LABEL_119:
                if ( (v176.m128i_i8[2] & 1) == 0 )
                {
                  v49 = *(unsigned __int8 *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4);
                  v24 = (unsigned int)(v49 - 3);
                  if ( ((unsigned __int8)(v49 - 3) <= 1u || !(_BYTE)v49) && *(char *)(v44 + 177) < 0 )
                  {
                    if ( *(_QWORD *)(v45 + 168) )
                    {
                      v94 = *(_QWORD *)(v45 + 152);
                      if ( !v94 || (*(_BYTE *)(v94 + 29) & 0x20) != 0 || *(_DWORD *)(v94 + 240) == -1 )
                      {
                        v34 = v44;
                        goto LABEL_159;
                      }
                    }
                  }
                }
                if ( !v25 )
                {
                  if ( v171 )
                    goto LABEL_125;
                  goto LABEL_299;
                }
                v69 = 3;
LABEL_244:
                v23 = (__int64)v12;
                v153 = v44;
                sub_8756F0(v69, v12, &v175.m128i_u64[1], 0);
                v22 = (__int64)v12;
                sub_86F690(v12);
                LODWORD(v53) = 1;
                v34 = v153;
                goto LABEL_161;
              }
            }
LABEL_117:
            v46 = v165;
            goto LABEL_118;
          }
          if ( qword_4F077A8 > 0x765Bu )
          {
            v111 = **(_QWORD ***)(v109 + 208);
            v112 = v12;
            while ( 1 )
            {
              v112 = *(_QWORD **)v112[8];
              if ( v111 == v112 )
                break;
              if ( (*((_BYTE *)v112 + 81) & 0x10) == 0 )
                goto LABEL_259;
            }
          }
        }
        v127 = 0;
        v129 = 1;
        goto LABEL_104;
      }
      goto LABEL_99;
    }
    v106 = v12[11];
    v129 = v165;
    if ( (v176.m128i_i8[0] & 1) != 0 )
    {
      if ( v25 )
      {
        v22 = 5;
        if ( dword_4D04964 )
          v22 = byte_4F07472[0];
        sub_684AA0(v22, 427, &v174);
        v24 = (unsigned __int64)&dword_4F04C64;
LABEL_434:
        v23 = v129;
        if ( !v129 )
        {
          v23 = (__int64)qword_4F04C68;
          v107 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 5) & 3;
          v24 = *(_BYTE *)(v106 + 88) & 3;
          if ( v107 != (_BYTE)v24 )
          {
            v23 = 720;
            if ( v25 )
            {
              v23 = 936;
              *(_BYTE *)(v106 + 88) = v107 | *(_BYTE *)(v106 + 88) & 0xFC;
            }
            v108 = 5;
            if ( dword_4D04964 )
              v108 = unk_4F07471;
            v22 = v108;
            sub_6853B0(v108, v23, &v174, v12);
          }
          if ( (*((_BYTE *)v12 + 81) & 2) == 0 && v25 )
          {
            v127 = 0;
            goto LABEL_104;
          }
          if ( (*(_DWORD *)(v106 + 176) & 0x19000) != 0x1000 )
          {
            v22 = 5;
            if ( dword_4D04964 )
              v22 = byte_4F07472[0];
            v23 = 1028;
            sub_684AA0(v22, 1028, &v174);
          }
        }
        goto LABEL_99;
      }
    }
    else if ( v25 )
    {
      goto LABEL_434;
    }
    if ( word_4F06418[0] != 75 || (*(_BYTE *)(a1 + 122) & 0x20) != 0 )
    {
      *a7 = 0;
      goto LABEL_99;
    }
    goto LABEL_434;
  }
  if ( (_BYTE)v22 || (unsigned __int8)(*((_BYTE *)v12 + 80) - 4) <= 1u && (*(_BYTE *)(v12[11] + 177LL) & 0x30) == 0x30 )
  {
    if ( !v137 && *(_QWORD *)(v66 + 72) )
    {
      if ( dword_4F04C34 == dword_4F04C64 )
        sub_6854C0(795, &v174, v12);
      else
        sub_6854C0(503, &v174, v12);
      v176.m128i_i8[1] |= 0x20u;
      v176.m128i_i64[1] = 0;
      v139 = 0;
      v144 = 0;
      v164 = 1;
      goto LABEL_67;
    }
LABEL_506:
    v24 = (unsigned __int64)&dword_4F04C64;
    v23 = qword_4F04C68[0];
    v71 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( *((_DWORD *)v12 + 10) == *(_DWORD *)v71 )
    {
      if ( (_BYTE)v22 )
        goto LABEL_566;
LABEL_570:
      v139 = 0;
      if ( !v144 )
      {
LABEL_359:
        if ( v25 )
          goto LABEL_360;
LABEL_99:
        if ( !v148 )
          goto LABEL_100;
LABEL_338:
        v127 = 0;
        v31 = 1;
        goto LABEL_102;
      }
      goto LABEL_335;
    }
    if ( (_BYTE)v22 )
    {
LABEL_331:
      v22 = (__int64)v12;
      v23 += 776LL * (int)dword_4F04C5C;
      if ( (unsigned int)sub_85ED80(v12, v23) )
      {
        v24 = (unsigned __int64)&dword_4F04C64;
        if ( (*((_BYTE *)v12 + 81) & 0x10) != 0 )
        {
          v23 = qword_4F04C68[0];
          v71 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          if ( *((_DWORD *)v12 + 10) != *(_DWORD *)v71 )
          {
            v139 = 0;
            if ( !v144 )
              goto LABEL_250;
            goto LABEL_335;
          }
LABEL_566:
          v139 = 0;
          v144 = 0;
          goto LABEL_250;
        }
        goto LABEL_570;
      }
LABEL_545:
      v124 = v12;
      v23 = (__int64)&v174;
      v22 = 503;
      v12 = 0;
      sub_6854C0(503, &v174, v124);
      v176.m128i_i8[1] |= 0x20u;
      v176.m128i_i64[1] = 0;
      v139 = 0;
      v144 = 0;
      v164 = 1;
      goto LABEL_99;
    }
LABEL_330:
    if ( !v12[8] )
      goto LABEL_545;
    goto LABEL_331;
  }
  v24 = (unsigned __int64)&dword_4F04C64;
  v23 = qword_4F04C68[0];
  if ( *((_DWORD *)v12 + 10) != *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64) )
    goto LABEL_330;
  v139 = 0;
  v22 = v144;
  if ( v144 )
  {
LABEL_335:
    v23 = (__int64)&v174;
    sub_648C10((__int64)v12, (__int64)&v174);
    *(_BYTE *)(v34 + 178) |= 3u;
    v22 = (__int64)v12;
    v139 = 0;
    *(_QWORD *)(v66 + 120) = sub_8807C0(v12);
    v144 = 1;
    if ( unk_4D04734 == 3 )
    {
      v22 = v34;
      sub_66A6A0(v34);
    }
LABEL_358:
    if ( (*((_BYTE *)v12 + 81) & 0x10) == 0 )
      goto LABEL_359;
    goto LABEL_249;
  }
LABEL_360:
  if ( !v12[8] )
  {
    v23 = (__int64)qword_4F04C68;
    if ( *((_DWORD *)v12 + 10) != *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C) )
    {
      sub_6854C0(551, &v174, v12);
      v175 = _mm_loadu_si128(xmmword_4F06660);
      v176 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v177 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v178 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v176.m128i_i8[1] |= 0x20u;
      v175.m128i_i64[1] = *(_QWORD *)dword_4F07508;
      goto LABEL_67;
    }
    v127 = 0;
    v129 = 0;
    goto LABEL_104;
  }
  v23 = (__int64)&v174;
  v22 = (__int64)v12;
  v172 = 0;
  if ( sub_668160((__int64)v12, (__int64)&v174, &v172, 1) )
  {
    v22 = v12[8];
    v23 = 0;
    sub_864230(v22, 0);
    v127 = 1;
    v170 = dword_4F04C64;
  }
  else
  {
    v127 = v172;
    if ( v172 )
    {
      v12 = 0;
      v127 = 0;
      v90 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v175 = _mm_loadu_si128(xmmword_4F06660);
      v91 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v176 = v90;
      v92 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v177 = v91;
      v176.m128i_i8[1] |= 0x20u;
      v178 = v92;
      v175.m128i_i64[1] = *(_QWORD *)dword_4F07508;
    }
  }
  v31 = 1;
LABEL_102:
  if ( v12 )
  {
    v129 = 0;
    goto LABEL_104;
  }
LABEL_68:
  v131 = v31;
  v32 = sub_7259C0(v140);
  v33 = v131;
  v34 = v32;
  if ( dword_4F04C58 == -1 )
  {
    v132 = unk_4F04C38 != 0;
  }
  else
  {
    *(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 202LL) |= 0x20u;
    v132 = 1;
  }
  v35 = qword_4F04C68;
  v36 = (int)v170;
  if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)v170 + 4) == 1 )
    *(_BYTE *)(v32 + 141) |= 0x80u;
  if ( dword_4F077C4 == 2 && v133 )
  {
    v35 = *(_QWORD **)(*(_QWORD *)(v133[11] + 168LL) + 168LL);
    *(_QWORD *)(*(_QWORD *)(v32 + 168) + 168LL) = v35;
  }
  v37 = v149;
  v150 = v33;
  if ( v136 )
  {
    v38 = sub_647630(v37, (__int64)&v175, v36, 0);
    *(_QWORD *)(v38 + 88) = v34;
    v12 = (_QWORD *)v38;
    sub_877D80(v34, v38);
    if ( v165 && !unk_4D047D0 )
      *((_BYTE *)v12 + 83) |= 0x40u;
    v39 = v164;
    if ( (v176.m128i_i8[1] & 0x20) != 0 )
      v39 = v136;
    v164 = v39;
    v40 = 0;
    if ( (v176.m128i_i8[1] & 0x20) == 0 )
      v40 = v139;
    v139 = v40;
    sub_85E280(v12, v170);
    v41 = v150;
    if ( !v165 || !v25 )
      goto LABEL_85;
    goto LABEL_386;
  }
  v86 = sub_87F680(v37, &dword_4F063F8, v36, v35, 0, v33);
  *(_QWORD *)(v86 + 88) = v34;
  v12 = (_QWORD *)v86;
  sub_877D80(v34, v86);
  sub_877D70(v34);
  *(_BYTE *)(v34 + 177) |= 4u;
  sub_85E280(v12, v170);
  v41 = v150;
  if ( v165 && v25 )
  {
LABEL_386:
    v159 = v41;
    sub_6854C0(551, &v174, v12);
    v41 = v159;
    v164 = 1;
LABEL_85:
    sub_66A7C0((__int64)v12, v41, 0, v170);
    if ( v165 && v136 && *qword_4D03FD0 )
      sub_8CFCB0(v34);
    goto LABEL_89;
  }
  sub_66A7C0((__int64)v12, v150, 0, v170);
LABEL_89:
  if ( unk_4F04C48 != -1 && (*((_BYTE *)v12 + 81) & 0x10) != 0 )
    sub_8968E0(v12, v140);
  v42 = (-(__int64)(v25 == 0) & 0xFFFFFFFFFFFFFFFELL) + 3;
  if ( v165 )
    v42 |= 0x100u;
  if ( dword_4F077C4 == 2
    && (unk_4F07778 > 201102 || dword_4F07774)
    && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 8) & 2) != 0 )
  {
    *(_BYTE *)(v34 + 88) = *(_BYTE *)(v34 + 88) & 0x8F | 0x10;
  }
  sub_8756F0(v42, v12, &v175.m128i_u64[1], 0);
  sub_86F690(v12);
  v23 = (__int64)&v175;
  v22 = v34;
  v43 = 0;
  sub_667260(v34, (__int64)&v175, v25, a5);
  v129 = 0;
LABEL_159:
  LODWORD(v53) = 1;
  if ( !v25 )
  {
LABEL_160:
    v53 = word_4F06418;
    LOBYTE(v53) = word_4F06418[0] == 75;
    v25 = 0;
  }
LABEL_161:
  v54 = (unsigned int)((_DWORD)v53 << 7);
  *(_BYTE *)(a1 + 127) = ((_BYTE)v53 << 7) | *(_BYTE *)(a1 + 127) & 0x7F;
  if ( *((_BYTE *)v12 + 80) != 3 && !(v43 | v144) )
  {
    v23 = v170;
    if ( v165 && dword_4F07590 && *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)v170 + 4) == 9 )
    {
      if ( !(unsigned int)sub_8DBE70(v34) )
      {
        v170 = dword_4F04C34;
        v12 = (_QWORD *)sub_886210(v12, (unsigned int)dword_4F04C34, 0);
        if ( !unk_4D047D0 )
          *((_BYTE *)v12 + 83) |= 0x40u;
      }
      v23 = v170;
    }
    v22 = v34;
    if ( (unsigned int)sub_736420(v34, v23) )
    {
      v23 = v170;
      v22 = v34;
      sub_7365B0(v34, v170);
    }
    if ( v132 )
    {
      v54 = (unsigned __int64)qword_4F04C68;
      if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * (int)v170 + 6) & 2) != 0 )
        *(_BYTE *)(v34 + 177) |= 0x20u;
    }
  }
  if ( v164
    || dword_4F04C44 != -1
    && (v54 = (unsigned __int64)qword_4F04C68, *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) == 6)
    || (unsigned __int8)(*(_BYTE *)(v34 + 140) - 9) <= 2u
    && (*(_BYTE *)(v34 + 177) & 0x20) != 0
    && (v22 = dword_4F07590) == 0
    && dword_4F04C58 == -1 )
  {
    sub_854B40();
  }
  else if ( v25 || word_4F06418[0] == 75 )
  {
    v23 = 0;
    v22 = (__int64)v12;
    sub_854980(v12, 0);
  }
  v55 = *(__m128i **)(a1 + 216);
  if ( v55 )
  {
    if ( !HIDWORD(qword_4F077B4) || (v56 = qword_4F077B4 | v25) != 0 )
    {
      v79 = 0;
      if ( !v137 || (v55 = (__m128i *)sub_5CF190(*(const __m128i **)(a1 + 216)), (v79 = dword_4D043F8) == 0) )
      {
LABEL_262:
        v23 = v34;
        v22 = (__int64)v55;
        sub_66A990(v55, v34, a1, v148 != 0, word_4F06418[0] == 75, v79);
        if ( v25 )
        {
          v22 = v34;
          sub_75BF90(v34);
        }
        goto LABEL_186;
      }
      v82 = 0;
LABEL_296:
      v145 = v82;
      v155 = v55;
      sub_66ABD0(v55->m128i_i64);
      v79 = v145;
      v55 = v155;
      goto LABEL_262;
    }
    if ( qword_4F077A8 > 0x9D07u )
    {
      if ( v137 )
      {
        v161 = qword_4F077B4 | v25;
        v98 = sub_5CF190(*(const __m128i **)(a1 + 216));
        v82 = v161;
        v55 = (__m128i *)v98;
        if ( dword_4D043F8 )
          goto LABEL_296;
      }
LABEL_185:
      v22 = (__int64)v55;
      v23 = v34;
      sub_66A990(v55, v34, a1, v148 != 0, word_4F06418[0] == 75, 0);
      goto LABEL_186;
    }
    if ( v12 && (*((_BYTE *)v12 + 81) & 0x10) != 0 )
    {
      if ( v137 )
        goto LABEL_396;
      v163 = *(__m128i **)(a1 + 216);
      v123 = sub_8788F0(v12);
      v55 = v163;
      if ( v123 && (*(_BYTE *)(v123 + 81) & 2) != 0 )
        goto LABEL_185;
    }
    else if ( v137 )
    {
LABEL_396:
      if ( word_4F06418[0] == 1 )
      {
        v56 = 1;
        if ( dword_4F077C4 == 2 )
          v56 = (unk_4D04A11 & 2) == 0 && (v125 = sub_7C0F00(0, 0), v55 = *(__m128i **)(a1 + 216), !v125)
             || !(unk_4D04A12 & 1);
      }
      else
      {
        if ( word_4F06418[0] == 34 || word_4F06418[0] == 27 )
          goto LABEL_501;
        if ( dword_4F077C4 == 2 )
        {
          v56 = 1;
          if ( word_4F06418[0] != 33 && (!dword_4D04474 || word_4F06418[0] != 52) )
          {
            if ( !dword_4D0485C || word_4F06418[0] != 25 )
            {
              v56 = word_4F06418[0] == 156;
              goto LABEL_400;
            }
LABEL_501:
            v56 = 1;
          }
        }
      }
LABEL_400:
      v160 = v56;
      v97 = (__m128i *)sub_5CF190(v55);
      v82 = v160;
      v55 = v97;
      if ( !dword_4D043F8 )
      {
        v23 = v34;
        v22 = (__int64)v97;
        sub_66A990(v97, v34, a1, v148 != 0, word_4F06418[0] == 75, v160);
        goto LABEL_186;
      }
      goto LABEL_296;
    }
    v23 = v34;
    v22 = (__int64)v55;
    sub_66A990(v55, v34, a1, v148 != 0, word_4F06418[0] == 75, 1);
  }
LABEL_186:
  if ( *(_QWORD *)(v34 + 104) )
  {
    v57 = sub_8D23B0(v34);
    v23 = 6;
    v22 = a1;
    sub_656C00(a1, 6, v34, v57 == 0, v25);
  }
  if ( v146 )
    sub_7B8B50(v22, v23, v54, v24);
  if ( v167 )
  {
    v22 = v34;
    sub_66A7B0(v34, SHIDWORD(v167));
  }
  if ( HIDWORD(qword_4F077B4) )
  {
    if ( dword_4F077C4 == 2 && (unsigned __int8)(*(_BYTE *)(v34 + 140) - 9) <= 2u )
    {
      v22 = (__int64)&v172;
      v154 = *(_QWORD *)(v34 + 168);
      LOBYTE(v172) = *(_BYTE *)(v154 + 109) & 7;
      sub_5D0D60(&v172, (*(_BYTE *)(v34 + 89) & 4) != 0);
      *(_BYTE *)(v154 + 109) = v172 & 7 | *(_BYTE *)(v154 + 109) & 0xF8;
      if ( (*(_BYTE *)(v34 + 177) & 0x10) == 0 )
        *(_BYTE *)(v34 + 143) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) & 0x80
                              | *(_BYTE *)(v34 + 143) & 0x7F;
    }
  }
  if ( !v25 )
  {
    v58 = v164 ^ 1;
    if ( a9 )
      *(_QWORD *)(a9 + 40) = *(_QWORD *)&v179[40];
    if ( !v43 )
    {
      if ( *((_BYTE *)v12 + 80) == 3 )
      {
        if ( !*a7 )
          goto LABEL_206;
LABEL_274:
        v22 = v164;
        if ( !v164 )
          goto LABEL_289;
        goto LABEL_275;
      }
      v59 = *(__m128i **)(v34 + 72);
      if ( !v59 )
      {
        if ( *a7 )
        {
LABEL_204:
          v22 = v34;
          v60 = sub_86A2A0(v34);
          if ( v60 )
          {
            if ( *(_BYTE *)(v60 + 16) == 53 )
            {
              v101 = *(_QWORD *)(v60 + 24);
              if ( !*(_QWORD *)(v101 + 8) )
              {
                v22 = *(_BYTE *)(v101 - 8) & 1;
                v102 = (__m128i *)sub_7274B0(v22);
                v103 = v165;
                v102[1] = _mm_loadu_si128((const __m128i *)&v179[32]);
                if ( v103 )
                  v102[1].m128i_i64[0] = *(_QWORD *)(a9 + 32);
                if ( v136 )
                  *v102 = _mm_loadu_si128((const __m128i *)&v179[16]);
                *(_QWORD *)(v101 + 8) = v102;
              }
            }
          }
          goto LABEL_206;
        }
        if ( !v164 )
          goto LABEL_208;
LABEL_275:
        *a6 = sub_72C930(v22);
        goto LABEL_209;
      }
      v59[1] = _mm_loadu_si128((const __m128i *)&v179[32]);
      if ( v136 )
        *v59 = _mm_loadu_si128((const __m128i *)&v179[16]);
    }
    if ( !*a7 )
      goto LABEL_206;
    if ( *((_BYTE *)v12 + 80) != 3 )
      goto LABEL_204;
    goto LABEL_274;
  }
  v22 = v34;
  v58 = sub_607B60((__int64 *)v34, a1, v170, 0, v132, v129, 0, v139, 0, (__int64)v179);
  if ( !v58 )
  {
    v164 = 1;
    if ( !v127 )
      goto LABEL_278;
LABEL_298:
    sub_8642D0();
    goto LABEL_278;
  }
  *a8 = 1;
  v58 = v164 ^ 1;
  if ( v127 )
    goto LABEL_298;
LABEL_278:
  if ( dword_4F04C44 == -1 || dword_4F04C44 < dword_4F04C58 )
  {
    v22 = 0;
    sub_5F94C0(0);
  }
  if ( !(unsigned int)sub_86D9F0() )
    goto LABEL_283;
  if ( dword_4F077C4 != 2 )
    goto LABEL_283;
  v22 = v34;
  if ( (unsigned int)sub_8D3E20(v34) )
    goto LABEL_283;
  if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
  {
    if ( *(_BYTE *)(v34 + 140) == 12 )
    {
      v115 = v34;
      do
        v115 = *(_QWORD *)(v115 + 160);
      while ( *(_BYTE *)(v115 + 140) == 12 );
      v116 = *(_QWORD *)(*(_QWORD *)v115 + 96LL);
      v117 = v34;
      if ( !*(_QWORD *)(v116 + 8) )
        goto LABEL_520;
      do
        v117 = *(_QWORD *)(v117 + 160);
      while ( *(_BYTE *)(v117 + 140) == 12 );
    }
    else
    {
      v116 = *(_QWORD *)v34;
      v117 = v34;
      v99 = *(_QWORD *)(*(_QWORD *)v34 + 96LL);
      if ( !*(_QWORD *)(v99 + 8) )
        goto LABEL_588;
    }
    v22 = *(_QWORD *)(*(_QWORD *)v117 + 96LL);
    if ( !(unsigned int)sub_879360(v22, a1, v116, v99) )
    {
      if ( *(_BYTE *)(v34 + 140) == 12 )
      {
LABEL_520:
        v118 = v34;
        do
          v118 = *(_QWORD *)(v118 + 160);
        while ( *(_BYTE *)(v118 + 140) == 12 );
        v119 = *(_QWORD *)(*(_QWORD *)v118 + 96LL);
        v120 = v34;
        if ( !*(_QWORD *)(v119 + 24) )
          goto LABEL_283;
        do
          v120 = *(_QWORD *)(v120 + 160);
        while ( *(_BYTE *)(v120 + 140) == 12 );
        goto LABEL_524;
      }
      v116 = *(_QWORD *)v34;
LABEL_588:
      v120 = v34;
      if ( !*(_QWORD *)(*(_QWORD *)(v116 + 96) + 24LL) )
        goto LABEL_283;
LABEL_524:
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v120 + 96LL) + 177LL) & 2) != 0 )
        goto LABEL_283;
    }
  }
  else
  {
    for ( i = v34; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( *(char *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 178LL) < 0 )
      goto LABEL_283;
  }
  v22 = 1230;
  sub_6851C0(1230, &v173);
LABEL_283:
  if ( a9 )
    *(_QWORD *)(a9 + 40) = *(_QWORD *)&v179[40];
  v81 = *(__m128i **)(v34 + 72);
  if ( v81 )
  {
    v81[1] = _mm_loadu_si128((const __m128i *)&v179[32]);
    if ( v136 )
      *v81 = _mm_loadu_si128((const __m128i *)&v179[16]);
  }
LABEL_206:
  if ( v164 )
    goto LABEL_275;
  if ( *((_BYTE *)v12 + 80) == 3 )
  {
LABEL_289:
    *a6 = v12[11];
    goto LABEL_209;
  }
LABEL_208:
  *a6 = v34;
LABEL_209:
  *(_QWORD *)a1 = v12;
  return v58;
}
