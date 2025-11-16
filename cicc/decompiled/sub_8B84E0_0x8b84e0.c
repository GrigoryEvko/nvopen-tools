// Function: sub_8B84E0
// Address: 0x8b84e0
//
__int64 __fastcall sub_8B84E0(__int64 a1)
{
  __int64 v2; // r12
  _QWORD *v3; // rax
  __int64 v4; // rcx
  unsigned __int64 v5; // r8
  unsigned __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // rax
  char v12; // dl
  __int64 result; // rax
  __int64 v14; // rdx
  char v15; // al
  unsigned __int8 v16; // r14
  int v17; // r11d
  unsigned __int64 v18; // rdi
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rdx
  unsigned int *v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // r14
  int v32; // r8d
  bool v33; // zf
  unsigned __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // ebx
  char v39; // al
  __int64 v40; // r9
  char v41; // bl
  unsigned __int16 v42; // dx
  bool v43; // al
  bool v44; // al
  unsigned __int64 v45; // rbx
  __int64 v46; // rax
  __int64 v47; // r9
  char v48; // al
  __m128i v49; // xmm6
  __m128i v50; // xmm7
  __m128i v51; // xmm5
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rsi
  bool v56; // dl
  int v57; // ebx
  unsigned __int8 v58; // dl
  __int64 v59; // rax
  _QWORD *v60; // rcx
  char v61; // al
  __int16 v62; // r8
  unsigned __int64 v63; // r9
  unsigned __int64 v64; // r9
  __m128i *v65; // r10
  __int64 v66; // rsi
  unsigned __int64 v67; // rdi
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  _DWORD *v71; // rdx
  int v72; // r15d
  __int16 v73; // r13
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  char v77; // al
  __int64 v78; // r10
  __int64 v79; // rax
  char v80; // al
  char v81; // al
  bool v82; // sf
  char v83; // al
  char v84; // al
  __int64 v85; // rdi
  char v86; // r15
  char v87; // r15
  char v88; // al
  const char *v89; // rdi
  size_t v90; // rax
  char *v91; // rax
  int v92; // eax
  __int16 v93; // r15
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 i; // rdx
  __int64 j; // rax
  int v98; // eax
  __int64 v99; // rcx
  __int64 v100; // r8
  int v101; // eax
  unsigned int v102; // edi
  __int16 v103; // r15
  char v104; // dl
  FILE *v105; // rsi
  __int64 v106; // rdx
  int v107; // ecx
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rax
  int v112; // eax
  __int64 v113; // r13
  unsigned __int64 v114; // rsi
  __m128i *v115; // rax
  char v116; // dl
  __int64 v117; // rax
  __int64 v118; // rbx
  __m128i *v119; // rax
  char v120; // al
  __int64 v121; // rsi
  __m128i *v122; // rax
  char v123; // dl
  __int64 v124; // rax
  __int64 v125; // rdi
  int v126; // eax
  char v127; // al
  char v128; // al
  char v129; // al
  __int64 v130; // [rsp+8h] [rbp-1A8h]
  __int64 v131; // [rsp+10h] [rbp-1A0h]
  int v132; // [rsp+20h] [rbp-190h]
  __int16 v133; // [rsp+20h] [rbp-190h]
  unsigned __int64 v134; // [rsp+20h] [rbp-190h]
  int v135; // [rsp+28h] [rbp-188h]
  __int64 v136; // [rsp+28h] [rbp-188h]
  int v137; // [rsp+30h] [rbp-180h]
  unsigned int v138; // [rsp+34h] [rbp-17Ch]
  __int64 v139; // [rsp+40h] [rbp-170h]
  __int64 v140; // [rsp+48h] [rbp-168h]
  __int64 v141; // [rsp+48h] [rbp-168h]
  unsigned __int64 v142; // [rsp+48h] [rbp-168h]
  int v143; // [rsp+48h] [rbp-168h]
  int v144; // [rsp+48h] [rbp-168h]
  unsigned __int64 v145; // [rsp+48h] [rbp-168h]
  __int64 v146; // [rsp+48h] [rbp-168h]
  unsigned __int8 v147; // [rsp+50h] [rbp-160h]
  __int64 v148; // [rsp+58h] [rbp-158h]
  __int64 v149; // [rsp+58h] [rbp-158h]
  int v150; // [rsp+64h] [rbp-14Ch] BYREF
  __int64 v151; // [rsp+68h] [rbp-148h] BYREF
  __m128i v152; // [rsp+70h] [rbp-140h] BYREF
  __m128i v153; // [rsp+80h] [rbp-130h]
  __m128i v154; // [rsp+90h] [rbp-120h]
  __m128i v155; // [rsp+A0h] [rbp-110h]
  __m128i v156[5]; // [rsp+B0h] [rbp-100h] BYREF
  _QWORD *v157; // [rsp+100h] [rbp-B0h]
  __m128i v158; // [rsp+110h] [rbp-A0h] BYREF
  unsigned int v159[10]; // [rsp+128h] [rbp-88h] BYREF
  char v160; // [rsp+150h] [rbp-60h]
  char v161; // [rsp+151h] [rbp-5Fh]

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD **)(*(_QWORD *)a1 + 448LL);
  *(_BYTE *)(*(_QWORD *)a1 + 130LL) |= 4u;
  memset(v156, 0, sizeof(v156));
  v157 = v3;
  if ( dword_4F0774C )
    *(_BYTE *)(v2 + 124) |= 0x40u;
  if ( dword_4D04870 && dword_4F077C4 == 2 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) |= 8u;
  sub_854C10(*(const __m128i **)(a1 + 328));
  *(_QWORD *)(v2 + 184) = sub_5CC190(1);
  v6 = (-(__int64)(dword_4F077BC == 0) & 0xFFFFFFFFFFC00000LL) + 4327474;
  if ( *(_DWORD *)(a1 + 20) )
  {
    *(_BYTE *)(v2 + 121) |= 0x40u;
    v6 |= 4u;
  }
  sub_672A20(v6 | 1, v2, (__int64)v156, v4, v5);
  v7 = *(_QWORD *)(v2 + 8);
  sub_854AB0();
  v10 = *(unsigned int *)(a1 + 52);
  if ( !(_DWORD)v10 )
  {
    if ( (v7 & 8) != 0 )
    {
      sub_6851C0(0xCD1u, (_DWORD *)(v2 + 32));
    }
    else if ( (v7 & 0x1000) != 0 )
    {
      sub_6851C0(0x2CFu, (_DWORD *)(v2 + 260));
    }
  }
  v11 = *(_QWORD *)(v2 + 288);
  if ( !v11 )
    goto LABEL_21;
  while ( 1 )
  {
    v12 = *(_BYTE *)(v11 + 140);
    if ( v12 != 12 )
      break;
    v11 = *(_QWORD *)(v11 + 160);
  }
  if ( !v12 )
  {
    if ( word_4F06418[0] != 1 )
    {
      if ( word_4F06418[0] == 27
        || word_4F06418[0] == 34
        || dword_4F077C4 == 2
        && (word_4F06418[0] == 33
         || dword_4D04474 && word_4F06418[0] == 52
         || dword_4D0485C && word_4F06418[0] == 25
         || word_4F06418[0] == 156) )
      {
        goto LABEL_28;
      }
LABEL_21:
      v153 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v152.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v153.m128i_i8[1] |= 0x20u;
      v154 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v152.m128i_i64[1] = *(_QWORD *)dword_4F07508;
      v155 = _mm_loadu_si128(&xmmword_4F06660[3]);
      goto LABEL_22;
    }
    if ( dword_4F077C4 == 2
      && ((word_4D04A10 & 0x200) != 0 || (unsigned int)sub_7C0F00(0, 0, (__int64)&dword_4F077C4, v10, v8, v9))
      && (unk_4D04A12 & 1) != 0 )
    {
      goto LABEL_21;
    }
  }
LABEL_28:
  if ( (v7 & 0x230) != 0 )
  {
    v14 = *(_QWORD *)(v2 + 288);
    v15 = *(_BYTE *)(v14 + 140);
    v16 = v15 - 9;
    if ( ((unsigned __int8)(v15 - 9) <= 2u || v15 == 2 && (*(_BYTE *)(v14 + 161) & 8) != 0) && word_4F06418[0] == 75 )
    {
      if ( (*(_BYTE *)(v2 + 124) & 0x20) != 0 )
      {
        sub_6451E0(v2);
        v14 = *(_QWORD *)(v2 + 288);
      }
      v47 = *(_QWORD *)v14;
      if ( !*(_DWORD *)(a1 + 52) && *(_BYTE *)(v2 + 268) )
      {
        v148 = *(_QWORD *)v14;
        sub_684AA0(7u, 0x9D3u, (_DWORD *)(v2 + 260));
        v47 = v148;
      }
      v48 = *(_BYTE *)(v47 + 80);
      if ( (unsigned __int8)(v48 - 4) <= 1u )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(v47 + 88) + 177LL) & 0x10) == 0 )
          goto LABEL_94;
      }
      else if ( v48 != 6 || (*(_BYTE *)(*(_QWORD *)(v47 + 88) + 162LL) & 0x10) == 0 )
      {
LABEL_94:
        if ( (*(_BYTE *)(v47 + 81) & 0x20) == 0 )
          sub_6854E0(0x318u, v47);
        goto LABEL_21;
      }
      v149 = v47;
      sub_88DC10(v47, (FILE *)(v2 + 32), a1);
      v53 = *(_QWORD *)(v2 + 288);
      if ( v16 > 2u )
        *(_BYTE *)(v53 + 162) |= 0x80u;
      else
        *(_BYTE *)(v53 + 178) |= 1u;
      v54 = *(_QWORD *)(v2 + 288);
      if ( (v7 & 0x20) != 0 )
      {
        *(_BYTE *)(v54 + 143) |= 8u;
        sub_7294B0(v157, (__int64 *)(*(_QWORD *)(*(_QWORD *)(v2 + 288) + 72LL) + 48LL));
      }
      else
      {
        sub_86A3D0(v54, *(_QWORD *)(v2 + 288), 0, 9, v156);
      }
      if ( v16 <= 2u )
      {
        v55 = *(_QWORD *)(*(_QWORD *)(v149 + 96) + 72LL);
        if ( v55 )
        {
          if ( (*(_BYTE *)(*(_QWORD *)(v55 + 88) + 266LL) & 0x20) != 0 )
            sub_6854E0(0x318u, v55);
        }
      }
      goto LABEL_21;
    }
  }
  v138 = dword_4F04C3C;
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  sub_87E3B0((__int64)&v158);
  v17 = *(_DWORD *)(a1 + 20);
  v18 = 4621;
  v131 = v7 & 2;
  v160 = (2 * ((v7 & 2) != 0)) | v160 & 0xFD;
  if ( v17 )
  {
    v18 = (-(__int64)((v7 & 0x400) == 0) & 0xFFFFFFFFFFFFFFE0LL) + 4653;
    if ( (v7 & 1) != 0 )
      goto LABEL_35;
  }
  else if ( (v7 & 1) != 0 )
  {
    goto LABEL_35;
  }
  if ( (*(_BYTE *)(v2 + 120) & 0x7F) == 0 )
    v18 |= 0x10000u;
LABEL_35:
  v19 = 0;
  sub_626F50(v18, v2, *(_QWORD *)(a1 + 240), (__int64)&v152, (__int64)&v158, v156);
  sub_645270(v2, v2, v20, v21);
  v139 = *(_QWORD *)(v2 + 16);
  if ( (v153.m128i_i8[1] & 0x20) == 0 )
  {
    v19 = v153.m128i_i64[1];
    if ( !v153.m128i_i64[1] )
      v19 = sub_7D5DD0(&v152, 0x20u, v22, v23, v24);
  }
  v25 = (__int64)&v152;
  sub_88F5B0(v2, (__int64)&v152);
  v26 = (__int64)&v158;
  sub_87E350((__int64)&v158);
  if ( (v153.m128i_i8[1] & 0x20) != 0 )
    goto LABEL_74;
  v29 = *(unsigned int *)(a1 + 44);
  if ( ((_DWORD)v29 || (v26 = *(unsigned int *)(a1 + 48), (_DWORD)v26)) && (v25 = dword_4D04740) == 0
    || (v135 = *(_DWORD *)(a1 + 52)) != 0 )
  {
    v49 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v50 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v152 = _mm_loadu_si128(xmmword_4F06660);
    v51 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v153 = v49;
    v154 = v50;
    v153.m128i_i8[1] = v49.m128i_i8[1] | 0x20;
    v152.m128i_i64[1] = *(_QWORD *)dword_4F07508;
    v155 = v51;
    goto LABEL_74;
  }
  if ( !v19 )
  {
    v25 = (__int64)&v152.m128i_i64[1];
    v26 = 801;
    sub_6851A0(0x321u, &v152.m128i_i32[2], *(_QWORD *)(v152.m128i_i64[0] + 8));
    goto LABEL_74;
  }
  v31 = sub_88F8A0(v19, (__int64)&v152);
  if ( *(_BYTE *)(v2 + 268) )
  {
    if ( (v153.m128i_i8[2] & 2) == 0 || (*(_BYTE *)(v2 + 121) & 0x40) != 0 )
    {
      if ( !dword_4F077BC || (LOBYTE(v32) = 4, qword_4F077A8 > 0x9D6Bu) )
      {
        LOBYTE(v32) = 7;
        if ( (_DWORD)qword_4F077B4 )
          v32 = ((*(_BYTE *)(v31 + 80) - 7) & 0xFD) == 0 ? 4 : 7;
      }
    }
    else
    {
      LOBYTE(v32) = 7;
      if ( dword_4F077BC )
        v32 = qword_4F077A8 < 0x9D6Cu ? 4 : 7;
    }
    v147 = v32;
    sub_684AA0(v32, 0x9D3u, (_DWORD *)(v2 + 260));
    if ( sub_67D370((int *)0x9D3, v147, (_DWORD *)(v2 + 260)) )
      *(_BYTE *)(v2 + 269) = 0;
  }
  v33 = (unsigned int)sub_8D2310(*(_QWORD *)(v2 + 288)) == 0;
  v34 = *(unsigned __int8 *)(v31 + 80);
  if ( v33 || (unsigned __int8)v34 > 0x14u || (v35 = 1182720, !_bittest64(&v35, v34)) )
  {
    if ( (((_BYTE)v34 - 7) & 0xFD) != 0 || !sub_892240(v31) )
      goto LABEL_126;
    v116 = *(_BYTE *)(v31 + 80);
    v26 = *(_QWORD *)(v2 + 288);
    v117 = *(_QWORD *)(v2 + 8) & 0x80000LL;
    if ( v116 == 9 )
    {
      v118 = *(_QWORD *)(v31 + 88);
    }
    else
    {
      if ( v116 == 7 )
      {
        v118 = *(_QWORD *)(v31 + 88);
        if ( !v117 )
        {
LABEL_388:
          *(_QWORD *)(v118 + 120) = v26;
          if ( (unsigned int)sub_8B1260(v26, *(_BYTE *)(v118 + 136), v2, 1) )
            *(_QWORD *)(v118 + 120) = sub_72C930();
          if ( *(_BYTE *)(v31 + 80) == 7 )
          {
LABEL_339:
            v27 = (*(_BYTE *)(v118 + 176) & 8) != 0;
            if ( (_BYTE)v27 != ((*(_QWORD *)(v2 + 8) & 0x400000LL) != 0) )
            {
              v26 = 8;
              v25 = (unsigned int)((*(_BYTE *)(v118 + 176) & 8) != 0) + 2502;
              sub_6854F0(8u, v25, &v152.m128i_i32[2], (_QWORD *)(v118 + 64));
              *(_QWORD *)(v2 + 8) &= ~0x400000uLL;
              goto LABEL_74;
            }
            v28 = (unsigned int *)qword_4F04C68;
            v38 = 0;
            v25 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
            if ( *(_DWORD *)v25 == *(_DWORD *)(v31 + 40) )
              goto LABEL_62;
LABEL_58:
            if ( (*(_BYTE *)(v31 + 81) & 0x10) != 0 || *(_QWORD *)(v31 + 64) )
            {
              v26 = v31;
              if ( (unsigned int)sub_85ED80(v31, v25) )
                goto LABEL_61;
              v28 = &dword_4F077BC;
              v126 = *(_DWORD *)(a1 + 52);
              v27 = dword_4F077BC;
              if ( dword_4F077BC )
              {
                if ( (*(_BYTE *)(v31 + 81) & 0x10) != 0 )
                {
                  if ( *(_BYTE *)(v31 + 80) == 9 || (v27 = (__int64)&qword_4F077A8, qword_4F077A8 <= 0x9CA3u) )
                  {
                    if ( !v126 )
                    {
                      sub_6853B0(5u, 0x1F7u, (FILE *)&v152.m128i_u64[1], v31);
LABEL_61:
                      if ( !v38 )
                        goto LABEL_62;
                      goto LABEL_352;
                    }
LABEL_74:
                    LOBYTE(v27) = word_4F06418[0] == 55;
                    if ( word_4F06418[0] == 73 || word_4F06418[0] == 55 )
                    {
                      sub_7C9610((unsigned __int8)v27);
                      **(_WORD **)(a1 + 184) = 74;
                    }
                    else if ( word_4F06418[0] == 56 )
                    {
                      ++*(_BYTE *)(qword_4F061C8 + 83LL);
                      sub_7BE180(v26, v25, v27, (__int64)v28, v29, v30);
                      --*(_BYTE *)(qword_4F061C8 + 83LL);
                    }
                    sub_854B40();
                    goto LABEL_79;
                  }
                }
              }
              if ( v126 )
                goto LABEL_74;
            }
            else if ( *(_DWORD *)(a1 + 52) )
            {
              goto LABEL_74;
            }
            *(_DWORD *)(a1 + 52) = 1;
            v25 = 503;
            v26 = 8;
            sub_6853B0(8u, 0x1F7u, (FILE *)&v152.m128i_u64[1], v31);
            goto LABEL_74;
          }
          v26 = *(_QWORD *)(v2 + 288);
LABEL_337:
          v121 = *(_QWORD *)(v118 + 120);
          if ( v121 != v26 && !(unsigned int)sub_8DED30(v26, v121, 5) )
          {
            v25 = (__int64)&v152.m128i_i64[1];
            v26 = 147;
            sub_6854C0(0x93u, (FILE *)&v152.m128i_u64[1], v31);
            goto LABEL_74;
          }
          goto LABEL_339;
        }
LABEL_334:
        if ( (*(_BYTE *)(v26 + 140) & 0xFB) == 8 )
        {
          if ( (sub_8D4C10(v26, dword_4F077C4 != 2) & 1) != 0 )
          {
            v120 = *(_BYTE *)(v31 + 80);
            v26 = *(_QWORD *)(v2 + 288);
LABEL_336:
            if ( v120 != 7 )
              goto LABEL_337;
            goto LABEL_388;
          }
          v26 = *(_QWORD *)(v2 + 288);
        }
        v119 = sub_73C570((const __m128i *)v26, 1);
        *(_QWORD *)(v2 + 288) = v119;
        v26 = (__int64)v119;
        v120 = *(_BYTE *)(v31 + 80);
        goto LABEL_336;
      }
      v118 = 0;
      if ( v116 == 21 )
        v118 = *(_QWORD *)(*(_QWORD *)(v31 + 88) + 192LL);
    }
    if ( !v117 )
      goto LABEL_337;
    goto LABEL_334;
  }
  v36 = *(_QWORD *)(a1 + 240);
  if ( v36 && *(char *)(v36 + 177) < 0 )
  {
    if ( word_4F06418[0] == 73 || word_4F06418[0] == 163 || word_4F06418[0] == 55 || (v160 & 0x18) != 0 )
    {
      *(_BYTE *)(v2 + 122) |= 1u;
      v160 |= 4u;
    }
    sub_5EDE90(*(_DWORD *)(a1 + 16), 1, &v152, v2, (__int64)&v158, (__int64)v156);
    if ( (*(_BYTE *)(v2 + 122) & 1) == 0 )
      goto LABEL_79;
    sub_88F1C0((__int64 *)a1, &v158);
    goto LABEL_82;
  }
  v25 = (__int64)&v152;
  v26 = sub_8B8140(
          (_BYTE *)v31,
          v2,
          (__int64)&v152,
          *(_DWORD *)(a1 + 20),
          1,
          0,
          *(_DWORD *)(a1 + 168) + *(_DWORD *)(a1 + 172),
          8u,
          &v151);
  v37 = sub_88F8A0(v26, (__int64)&v152);
  v31 = v37;
  if ( !v37 )
    goto LABEL_74;
  if ( !*(_QWORD *)(v37 + 96) )
  {
LABEL_126:
    v25 = (__int64)&v152.m128i_i64[1];
    v26 = 792;
    sub_6854C0(0x318u, (FILE *)&v152.m128i_u64[1], v31);
    goto LABEL_74;
  }
  v28 = (unsigned int *)qword_4F04C68;
  v38 = 1;
  v25 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  if ( *(_DWORD *)(v37 + 40) != *(_DWORD *)v25 )
    goto LABEL_58;
LABEL_352:
  if ( (*(_BYTE *)(v31 + 82) & 8) != 0 )
  {
    v123 = *(_BYTE *)(v31 + 80);
    v124 = 0;
    v125 = v31;
    if ( (v153.m128i_i8[2] & 2) == 0 )
      v124 = v154.m128i_i64[0];
    if ( v123 == 16 )
    {
      v125 = **(_QWORD **)(v31 + 88);
      v123 = *(_BYTE *)(v125 + 80);
    }
    if ( v123 == 24 )
      v125 = *(_QWORD *)(v125 + 88);
    if ( !(unsigned int)sub_880800(v125, *(_QWORD *)(v124 + 128)) )
      sub_887650((__int64)&v152);
  }
LABEL_62:
  if ( ((*(_BYTE *)(v31 + 80) - 7) & 0xFD) == 0 )
  {
    v141 = *(_QWORD *)(v31 + 88);
    *(_QWORD *)(v30 + 120) = sub_8D79B0(*(_QWORD *)(v141 + 120), *(_QWORD *)(v2 + 288));
    if ( *(_BYTE *)(v31 + 80) != 7 || (v56 = 1, (*(_BYTE *)(v31 + 81) & 0x10) != 0) )
    {
      v56 = (word_4F06418[0] == 56) | v139 & 1;
      if ( !v56 )
        v56 = dword_4D04428 != 0 && word_4F06418[0] == 73;
    }
    *(_BYTE *)(v2 + 122) = v56 | *(_BYTE *)(v2 + 122) & 0xFE;
    if ( (*(_DWORD *)(v2 + 120) & 0x14000) == 0x10000 && *(char *)(v141 + 170) < 0 && *(char *)(v141 + 176) < 0 )
    {
      if ( *(int *)(a1 + 176) <= 1 )
      {
        v57 = 1;
        if ( (*(_BYTE *)(v2 + 121) & 0x40) == 0 )
        {
          sub_6581B0(v141, v2, v56);
          v40 = v141;
LABEL_238:
          *(_BYTE *)(v2 + 127) &= ~0x10u;
          v45 = 0;
          v141 = v40;
          goto LABEL_69;
        }
        goto LABEL_133;
      }
      *(_BYTE *)(v141 + 176) &= ~0x80u;
      v57 = 0;
      *(_BYTE *)(v141 + 172) &= ~0x20u;
    }
    else
    {
      v57 = *(_BYTE *)(v141 + 170) >> 7;
    }
    if ( (*(_BYTE *)(v2 + 121) & 0x40) == 0 )
    {
      if ( *(char *)(v141 + 176) >= 0 )
        *(_BYTE *)(v141 + 176) &= ~1u;
      v58 = *(_BYTE *)(v2 + 122) & 1;
LABEL_134:
      sub_6581B0(v141, v2, v58);
      v40 = v141;
      if ( !v57 )
      {
        *(_BYTE *)(v31 + 81) &= ~2u;
        v45 = 0;
        *(_BYTE *)(v2 + 127) |= 0x10u;
        v132 = 1;
        goto LABEL_136;
      }
      goto LABEL_238;
    }
LABEL_133:
    *(_BYTE *)(v141 + 176) = (word_4F06418[0] != 75) | *(_BYTE *)(v141 + 176) & 0xFE;
    v58 = *(_BYTE *)(v2 + 122) & 1;
    goto LABEL_134;
  }
  v140 = *(_QWORD *)(v31 + 88);
  v39 = sub_877F80(v31);
  v40 = v140;
  v41 = v39;
  v42 = word_4F06418[0];
  if ( word_4F06418[0] == 56 )
  {
    v112 = sub_651030(&v151);
    v40 = v140;
    v42 = word_4F06418[0];
    if ( v112 )
    {
      if ( (_DWORD)v151 )
        v160 |= 8u;
      else
        v160 |= 0x10u;
    }
  }
  v43 = v42 == 163 || v42 == 73;
  if ( !v43 )
  {
    v43 = v42 == 55 && v41 == 1;
    if ( !v43 )
      v43 = (v160 & 0x18) != 0;
  }
  *(_BYTE *)(v2 + 122) = *(_BYTE *)(v2 + 122) & 0xFE | v43;
  if ( (*(_DWORD *)(v2 + 120) & 0x14000) == 0x10000
    && *(int *)(a1 + 176) > 1
    && (*(_DWORD *)(v40 + 192) & 0x22002000) == 0x22000000 )
  {
    *(_DWORD *)(v40 + 192) &= 0xDFFFFF7F;
    *(_BYTE *)(v31 + 81) &= ~2u;
    *(_QWORD *)(v40 + 264) = 0;
    if ( (*(_BYTE *)(v40 + 206) & 0x10) == 0 )
    {
LABEL_291:
      *(_BYTE *)(v2 + 127) |= 0x10u;
      v45 = v40;
      v141 = 0;
      v132 = 1;
      goto LABEL_136;
    }
LABEL_290:
    *(_BYTE *)(v40 + 206) &= ~0x10u;
    *(_BYTE *)(v40 + 193) &= ~0x20u;
    *(_BYTE *)(v31 + 81) &= ~2u;
    *(_QWORD *)(v40 + 264) = 0;
    goto LABEL_291;
  }
  v44 = (*(_BYTE *)(v40 + 195) & 2) != 0;
  if ( (*(_BYTE *)(v40 + 206) & 0x10) != 0 )
  {
    if ( (*(_BYTE *)(v40 + 195) & 2) == 0 )
      goto LABEL_290;
    *(_BYTE *)(v2 + 127) &= ~0x10u;
    v45 = v40;
    v141 = 0;
LABEL_69:
    v137 = 1;
    if ( (*(_BYTE *)(v2 + 122) & 1) == 0 )
    {
      v151 = *(_QWORD *)(v31 + 48);
      *(_QWORD *)v2 = v31;
      goto LABEL_181;
    }
    goto LABEL_70;
  }
  *(_BYTE *)(v2 + 127) = *(_BYTE *)(v2 + 127) & 0xEF | (16 * ((*(_BYTE *)(v40 + 195) & 2) == 0));
  v132 = 1 - v44;
  v45 = v40;
  v135 = v132;
  v141 = 0;
  if ( v44 )
    goto LABEL_69;
LABEL_136:
  v136 = v40;
  sub_899850(v31, (FILE *)&v152.m128i_u64[1]);
  v40 = v136;
  if ( (*(_BYTE *)(v136 + 88) & 4) == 0 )
  {
    v137 = 0;
    v135 = v132;
    if ( (*(_BYTE *)(v2 + 122) & 1) == 0 )
    {
LABEL_142:
      *(_QWORD *)(v40 + 64) = *(_QWORD *)(v2 + 48);
      v61 = *(_BYTE *)(v2 + 122) & 1;
      goto LABEL_143;
    }
LABEL_70:
    if ( (*(_BYTE *)(v31 + 81) & 2) != 0 && (!v141 || (*(_BYTE *)(v141 + 172) & 0x20) == 0) )
    {
      v26 = (__int64)&v152.m128i_i64[1];
      v25 = v31;
      sub_685920(&v152.m128i_i32[2], (FILE *)v31, 8u);
      goto LABEL_74;
    }
    goto LABEL_142;
  }
  v59 = sub_892240(v31);
  v40 = v136;
  if ( (*(_BYTE *)(v59 + 80) & 1) != 0 )
  {
    v60 = (_QWORD *)(v59 + 92);
    if ( !dword_4F077BC || *(_BYTE *)(v31 + 80) != 9 )
    {
      v25 = 1449;
      v26 = 8;
      sub_686A30(8u, 0x5A9u, &v152.m128i_i32[2], v60, v31);
      goto LABEL_74;
    }
    sub_686A30(5u, 0x5A9u, &v152.m128i_i32[2], v60, v31);
    v137 = 0;
    v40 = v136;
    v135 = v132;
    v61 = *(_BYTE *)(v2 + 122) & 1;
  }
  else
  {
    v137 = 0;
    v135 = v132;
    v61 = *(_BYTE *)(v2 + 122) & 1;
  }
LABEL_143:
  v151 = *(_QWORD *)(v31 + 48);
  *(_QWORD *)v2 = v31;
  if ( !v61 )
  {
LABEL_181:
    v62 = 1;
    goto LABEL_146;
  }
  v62 = 3;
  if ( ((*(_BYTE *)(v31 + 80) - 7) & 0xFD) == 0 )
  {
    *(_BYTE *)(v40 + 88) |= 4u;
    v62 = 2051;
  }
LABEL_146:
  v130 = v40;
  v133 = v62;
  sub_854980(v31, 0);
  sub_8756F0(v133, v31, &v152.m128i_i64[1], *(_QWORD **)(v2 + 352));
  v63 = v130;
  if ( (*(_BYTE *)(v2 + 122) & 1) != 0
    || (*(_QWORD *)(v31 + 48) = *(_QWORD *)(v130 + 64), (*(_BYTE *)(v2 + 122) & 1) != 0)
    || (v135 & 1) != 0 )
  {
    sub_729470(v130, v156);
    v63 = v130;
  }
  v134 = v63;
  sub_88DC10(v31, (FILE *)&v152.m128i_u64[1], a1);
  v64 = v134;
  if ( ((*(_BYTE *)(v31 + 80) - 7) & 0xFD) == 0 )
  {
    if ( (*(_BYTE *)(v2 + 122) & 1) != 0 )
    {
      if ( dword_4F04C64 != -1
        && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0
        && dword_4F077C4 == 2
        && (*(_BYTE *)(v134 - 8) & 1) != 0
        && (v153.m128i_i8[2] & 0x40) == 0 )
      {
        v122 = sub_7CAFF0((__int64)&v152, v134);
        if ( v122 )
          v122[2].m128i_i8[1] |= 0x10u;
      }
      if ( v135 || !*(_QWORD *)(v141 + 256) )
        *(_QWORD *)(v141 + 256) = *(_QWORD *)(v2 + 280);
      *(_BYTE *)(v141 + 137) = *(_BYTE *)(v2 + 268);
    }
    else
    {
      v65 = 0;
      if ( dword_4F04C64 != -1
        && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0
        && dword_4F077C4 == 2
        && (*(_BYTE *)(v134 - 8) & 1) != 0
        && (v153.m128i_i8[2] & 0x40) == 0 )
      {
        v65 = sub_7CAFF0((__int64)&v152, v134);
      }
      sub_86A3D0(v141, *(_QWORD *)(v2 + 288), (__int64)v65, v135 == 0 ? 8 : 24, v156);
    }
    *(_BYTE *)(v141 + 170) |= 0x80u;
    if ( (*(_BYTE *)(v2 + 121) & 0x40) != 0 )
      *(_BYTE *)(v141 + 176) |= 0x80u;
    if ( v131 )
    {
      if ( dword_4D04820 || dword_4F077BC && (unsigned int)sub_657F30((unsigned int *)(v2 + 88)) )
        *(_BYTE *)(v141 + 172) |= 0x20u;
      else
        sub_6851C0(0x145u, (_DWORD *)(v2 + 88));
    }
    v66 = *(_BYTE *)(v2 + 122) & 1;
    sub_644920((_QWORD *)v2, v66);
    v67 = v2;
    sub_65C210(v2);
    if ( (*(_BYTE *)(v2 + 122) & 1) == 0 )
      goto LABEL_178;
    v150 = 0;
    if ( HIDWORD(qword_4F077B4)
      && !(_DWORD)qword_4F077B4
      && qword_4F077A8
      && *(_BYTE *)(v31 + 80) == 7
      && (*(_BYTE *)(v141 + 172) & 0x20) == 0
      && (v67 = *(_QWORD *)(v2 + 288), (*(_BYTE *)(v67 + 140) & 0xFB) == 8) )
    {
      v66 = dword_4F077C4 != 2;
      v127 = sub_8D4C10(v67, v66);
      v71 = &dword_4F077C4;
      if ( (v127 & 1) != 0 )
      {
        v67 = *(_QWORD *)(v2 + 288);
        if ( (*(_BYTE *)(v67 + 140) & 0xFB) != 8
          || (v66 = dword_4F077C4 != 2, v129 = sub_8D4C10(v67, v66), v71 = &dword_4F077C4, (v129 & 2) == 0) )
        {
          v68 = v141;
          v128 = *(_BYTE *)(v141 + 88);
          *(_BYTE *)(v141 + 168) &= 0xF8u;
          *(_BYTE *)(v141 + 136) = 2;
          *(_BYTE *)(v141 + 88) = v128 & 0x8F | 0x10;
          goto LABEL_170;
        }
      }
    }
    else
    {
      v71 = &dword_4F077C4;
    }
    if ( *(_BYTE *)(v141 + 136) != 2 )
      *(_BYTE *)(v141 + 136) = 0;
LABEL_170:
    if ( dword_4F077C4 == 2 )
    {
      v113 = *(_QWORD *)(v141 + 120);
      v67 = v113;
      if ( (unsigned int)sub_8D23B0(v113) )
      {
        v67 = v113;
        sub_8AE000(v113);
      }
    }
    if ( word_4F06418[0] == 75 )
    {
      if ( *(_BYTE *)(v141 + 177) )
      {
LABEL_178:
        v72 = *(_DWORD *)(v31 + 48);
        v73 = *(_WORD *)(v31 + 52);
        *(_QWORD *)(v31 + 48) = v151;
        sub_648B20((_BYTE *)v2);
        *(_DWORD *)(v31 + 48) = v72;
        *(_WORD *)(v31 + 52) = v73;
        sub_65C470(v2, v66, v74, v75, v76);
        goto LABEL_79;
      }
      if ( (*(_BYTE *)(v2 + 10) & 0x20) != 0 )
        *(_BYTE *)(v141 + 172) |= 0x10u;
      v66 = v31 + 48;
      if ( !(unsigned int)sub_63BB10(v31, v31 + 48) )
      {
        v66 = *(_QWORD *)(v141 + 120);
        sub_640330(v31, v66, 0, 0);
      }
    }
    else
    {
      if ( (v139 & 1) == 0 && word_4F06418[0] == 56 )
        sub_7B8B50(v67, (unsigned int *)v66, (__int64)v71, v68, v69, v70);
      else
        *(_BYTE *)(v2 + 127) |= 8u;
      v66 = (__int64)&v152.m128i_i64[1];
      sub_638AC0(v2, &v152.m128i_i64[1], 2u, v139 & 1, &v150, v156);
    }
    if ( dword_4F077BC && (v139 & 1) != 0 && word_4F06418[0] == 142 )
    {
      v66 = v2;
      sub_650EA0(v141, v2);
    }
    goto LABEL_178;
  }
  *(_BYTE *)(v45 + 207) = (2 * *(_BYTE *)(v2 + 125)) & 0x10 | *(_BYTE *)(v45 + 207) & 0xEF;
  v77 = *(_BYTE *)(v45 + 174);
  if ( v77 == 2 || v77 == 5 && ((*(_BYTE *)(v45 + 176) - 2) & 0xFD) == 0 )
  {
    sub_5F93D0(v45, (__int64 *)(v2 + 288));
    v64 = v134;
  }
  if ( !dword_4F077BC )
  {
    v145 = v64;
    sub_894C00(v31);
    sub_6464A0(*(_QWORD *)(v2 + 288), v31, v159, 0);
    v64 = v145;
  }
  v142 = v64;
  v78 = sub_624310(*(_QWORD *)(v2 + 288), (__int64)&v158);
  if ( dword_4F04C64 == -1
    || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) == 0
    || dword_4F077C4 != 2
    || (*(_BYTE *)(v142 - 8) & 1) == 0
    || (v153.m128i_i8[2] & 0x40) != 0 )
  {
    if ( v135 )
    {
      if ( (*(_BYTE *)(v2 + 122) & 1) == 0 )
      {
        v106 = 0;
        LOBYTE(v107) = 24;
        goto LABEL_265;
      }
    }
    else if ( (*(_BYTE *)(v2 + 122) & 1) == 0 )
    {
      v106 = 0;
      LOBYTE(v107) = 8;
      goto LABEL_265;
    }
LABEL_191:
    sub_64A300(v45, v78);
    *(_BYTE *)(v45 + 173) = *(_BYTE *)(v2 + 268);
    goto LABEL_192;
  }
  v114 = v142;
  v146 = v78;
  v115 = sub_7CAFF0((__int64)&v152, v114);
  v78 = v146;
  v106 = (__int64)v115;
  v107 = v135 == 0 ? 8 : 24;
  if ( (*(_BYTE *)(v2 + 122) & 1) != 0 )
  {
    if ( v115 )
      v115[2].m128i_i8[1] |= 0x10u;
    goto LABEL_191;
  }
LABEL_265:
  v108 = sub_86A3D0(v45, v78, v106, v107, v156);
  if ( v108 )
    *(_BYTE *)(v108 + 56) = *(_BYTE *)(v2 + 268);
LABEL_192:
  if ( (*(_DWORD *)(v45 + 192) & 0x2000080) != 0x2000080 )
    sub_736C90(v45, (v160 & 2) != 0);
  v79 = *(_QWORD *)(v2 + 8);
  if ( (*(_BYTE *)(v45 + 195) & 2) != 0 )
  {
    if ( (*(_BYTE *)(v45 + 193) & 1) != ((v79 & 0x80000) != 0)
      || ((*(_BYTE *)(v45 + 193) & 4) != 0) != ((v79 & 0x100000) != 0) )
    {
      v101 = *(_DWORD *)(v31 + 48);
      v102 = 2930;
      v103 = *(_WORD *)(v31 + 52);
      *(_QWORD *)(v31 + 48) = v151;
      v104 = *(_BYTE *)(v45 + 193);
      if ( (v104 & 4) == 0 )
      {
        v102 = 2383;
        if ( (v104 & 2) == 0 )
          v102 = (*(_QWORD *)(v2 + 8) & 0x100000LL) == 0 ? 2384 : 2931;
      }
      v144 = v101;
      v105 = (FILE *)(v2 + 112);
      if ( (*(_BYTE *)(v45 + 193) & 1) != 0 )
        v105 = (FILE *)(v2 + 48);
      sub_6854C0(v102, v105, v31);
      *(_WORD *)(v31 + 52) = v103;
      *(_DWORD *)(v31 + 48) = v144;
    }
  }
  else if ( (v79 & 0x180000) != 0 )
  {
    v33 = (v79 & 0x100000) == 0;
    v80 = *(_BYTE *)(v45 + 193);
    if ( v33 )
      v81 = v80 | 1;
    else
      v81 = v80 | 4;
    *(_BYTE *)(v45 + 193) = v81;
    v82 = *(char *)(v45 + 192) < 0;
    *(_BYTE *)(v45 + 193) = v81 | 2;
    if ( !v82 )
      sub_736C90(v45, 1);
  }
  else
  {
    *(_BYTE *)(v45 + 193) &= 0xF8u;
  }
  *(_BYTE *)(v45 + 195) |= 2u;
  *(_QWORD *)(v45 + 216) = 0;
  v83 = *(_BYTE *)(v2 + 269);
  if ( v83 && !*(_DWORD *)(a1 + 20) )
  {
    if ( (*(_BYTE *)(v45 + 89) & 4) == 0 )
      goto LABEL_206;
    if ( v83 == 2 )
      sub_684AA0(8u, 0x62Bu, (_DWORD *)(v2 + 32));
  }
  *(_BYTE *)(v2 + 269) = *(_BYTE *)(v45 + 172);
LABEL_206:
  if ( (v160 & 2) != 0 && !dword_4D04824 || *(_BYTE *)(v2 + 269) == 2 )
  {
    v84 = *(_BYTE *)(v45 + 88);
    *(_BYTE *)(v45 + 200) &= 0xF8u;
    *(_BYTE *)(v45 + 172) = 2;
    *(_BYTE *)(v45 + 88) = v84 & 0x8F | 0x10;
  }
  else
  {
    *(_BYTE *)(v45 + 172) = (*(_BYTE *)(v2 + 122) & 1) == 0;
    *(_BYTE *)(v45 + 88) = *(_BYTE *)(v45 + 88) & 0x8F | 0x20;
  }
  v85 = *(_QWORD *)(v2 + 288);
  if ( *(_BYTE *)(v85 + 140) == 12 && (unsigned int)sub_8D4970(v85) )
  {
    v111 = *(_QWORD *)(v2 + 288);
    for ( v160 |= 0x80u; *(_BYTE *)(v111 + 140) == 12; v111 = *(_QWORD *)(v111 + 160) )
      ;
    *(_QWORD *)(v2 + 288) = v111;
  }
  if ( *(_QWORD *)(v2 + 184) || *(_QWORD *)(v2 + 200) )
  {
    if ( v137 )
    {
      sub_644920((_QWORD *)v2, *(_BYTE *)(v2 + 122) & 1);
    }
    else
    {
      v86 = *(_BYTE *)(v45 + 200);
      *(_BYTE *)(v45 + 200) = v86 & 0xF8;
      v87 = v86 & 7;
      sub_644920((_QWORD *)v2, *(_BYTE *)(v2 + 122) & 1);
      v88 = *(_BYTE *)(v45 + 200);
      if ( (v88 & 7) == 0 )
        *(_BYTE *)(v45 + 200) = v88 & 0xF8 | v87;
    }
  }
  sub_65C210(v2);
  if ( (*(_BYTE *)(v45 + 199) & 1) != 0 )
  {
    v89 = *(const char **)(v2 + 240);
    if ( v89 )
    {
      v90 = strlen(v89);
      v91 = (char *)sub_7247C0(v90 + 1);
      *(_QWORD *)(v45 + 136) = v91;
      strcpy(v91, *(const char **)(v2 + 240));
    }
  }
  if ( (*(_BYTE *)(v45 + 198) & 0x20) != 0 )
  {
    v109 = *(_QWORD *)(v45 + 152);
    if ( v109 )
    {
      v110 = *(_QWORD *)(v2 + 288);
      if ( v110 )
      {
        while ( *(_BYTE *)(v109 + 140) == 12 )
          v109 = *(_QWORD *)(v109 + 160);
        for ( ; *(_BYTE *)(v110 + 140) == 12; v110 = *(_QWORD *)(v110 + 160) )
          ;
        sub_826B90(**(__int64 ****)(v109 + 168), **(__int64 ****)(v110 + 168), 0xE74u, dword_4F07508, &v151);
      }
    }
  }
  sub_826060(v45, &v152.m128i_i32[2]);
  v92 = *(_DWORD *)(v31 + 48);
  v93 = *(_WORD *)(v31 + 52);
  *(_QWORD *)(v31 + 48) = v151;
  v143 = v92;
  sub_648B00(v45, (_BYTE *)(v2 + 224), (__int64)&v152.m128i_i64[1]);
  *(_WORD *)(v31 + 52) = v93;
  *(_DWORD *)(v31 + 48) = v143;
  if ( (*(_BYTE *)(v2 + 122) & 1) == 0 )
  {
    sub_876830((__int64)&v158);
    goto LABEL_79;
  }
  v82 = v160 < 0;
  v160 |= 4u;
  if ( v82 )
    sub_6851C0(0x5Du, dword_4F07508);
  if ( dword_4F0690C )
  {
    for ( i = *(_QWORD *)(v45 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    for ( j = *(_QWORD *)(v2 + 288); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v95 = **(_QWORD **)(j + 168);
    v94 = **(_QWORD **)(i + 168);
    if ( v95 && v94 )
    {
      do
      {
        *(_DWORD *)(v94 + 32) = *(_DWORD *)(v95 + 32) & 0x3F800 | *(_DWORD *)(v94 + 32) & 0xFFFC07FF;
        v94 = *(_QWORD *)v94;
        v95 = *(_QWORD *)v95;
      }
      while ( v94 && v95 );
    }
  }
  if ( (v160 & 0x18) != 0 )
  {
    sub_71DEE0((_BYTE *)v2, (__int64)&v158, v94, v95);
    **(_WORD **)(a1 + 184) = 75;
LABEL_79:
    if ( dword_4F04C64 == -1
      || (v46 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v46 + 7) & 1) == 0)
      || dword_4F04C44 == -1 && (*(_BYTE *)(v46 + 6) & 2) == 0 )
    {
      if ( (v161 & 8) == 0 )
        sub_87E280((_QWORD **)&v158.m128i_i64[1]);
    }
    goto LABEL_82;
  }
  if ( !*(_DWORD *)(a1 + 20) )
  {
    if ( (*(_BYTE *)(v45 + 89) & 4) != 0 )
      *(_BYTE *)(v45 + 203) |= 1u;
    v98 = sub_880920(v31);
    sub_71E0E0(v45, (__int64)&v158, (-(__int64)(v98 == 0) & 0xFFFFFFFFFFFFFFE0LL) + 36, v99, v100);
    **(_WORD **)(a1 + 184) = 74;
    goto LABEL_79;
  }
  sub_88F1C0((__int64 *)a1, &v158);
  sub_736C90(v45, 1);
LABEL_82:
  --*(_BYTE *)(qword_4F061C8 + 83LL);
  dword_4F04C3C = v138;
LABEL_22:
  sub_643EB0(v2, 0);
  result = dword_4D04870;
  if ( dword_4D04870 )
  {
    if ( *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 456) )
      sub_87DD20(dword_4F04C40);
    result = (__int64)&dword_4F077C4;
    if ( dword_4F077C4 == 2 )
    {
      v52 = (int)dword_4F04C40;
      result = 776 * v52;
      *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) &= ~8u;
      if ( *(_QWORD *)(qword_4F04C68[0] + 776 * v52 + 456) )
        return sub_8845B0(v52);
    }
  }
  return result;
}
