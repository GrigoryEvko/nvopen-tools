// Function: sub_7B8B50
// Address: 0x7b8b50
//
__int64 __fastcall sub_7B8B50(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v6; // r8
  __int64 v7; // rbx
  char v8; // dl
  __int64 v9; // rdx
  unsigned int *v10; // rcx
  unsigned __int64 *v11; // r13
  unsigned __int64 v12; // rdx
  char *v13; // r11
  char v14; // r12
  unsigned __int16 v15; // ax
  __int64 v16; // rax
  int v17; // ecx
  unsigned __int16 v18; // dx
  const char *v19; // r9
  int v21; // eax
  unsigned int v22; // r12d
  __int16 v23; // r9
  __int16 v24; // ax
  char *v25; // rdi
  int v26; // edx
  __int64 v27; // rax
  __m128i v28; // xmm1
  __m128i v29; // xmm2
  __m128i v30; // xmm3
  __m128i v31; // xmm4
  __int64 v32; // rdi
  char v33; // al
  char v34; // al
  const __m128i *v35; // rax
  _QWORD *v36; // rsi
  const __m128i *v37; // rbx
  __int8 v38; // cl
  unsigned __int16 v39; // ax
  __int64 v40; // rax
  __int64 v41; // rax
  char v42; // al
  char *v43; // rdx
  char v44; // al
  char *v45; // rdx
  unsigned __int8 v46; // di
  __int64 v47; // rdx
  __int64 v48; // r10
  __m128i v49; // xmm5
  __m128i v50; // xmm6
  __m128i v51; // xmm7
  _BYTE *v52; // rdx
  __m128i v53; // xmm0
  __int64 v54; // r12
  char v55; // al
  __int64 v56; // rax
  __int64 v57; // r12
  char v58; // al
  unsigned __int16 v59; // ax
  _BYTE *v60; // rdx
  unsigned __int16 v61; // ax
  int v62; // edx
  __int64 v63; // rax
  __int16 v64; // r9
  __int16 v65; // cx
  __int16 v66; // ax
  char *v67; // rax
  char *v68; // rdi
  size_t v69; // rax
  __int64 v70; // rax
  unsigned int v71; // esi
  unsigned __int8 v72; // di
  __int16 v73; // ax
  int v74; // r15d
  _BYTE *v75; // rsi
  __int64 v76; // rcx
  char *v77; // rdi
  bool v78; // cf
  bool v79; // zf
  const char *v80; // r12
  unsigned __int64 v81; // rcx
  const char *v82; // rax
  _BYTE *v83; // rsi
  __int64 v84; // rdx
  char v85; // al
  _BYTE *v86; // rax
  int v87; // eax
  __int64 v88; // rdx
  char v89; // al
  char v90; // al
  char v91; // al
  char v92; // al
  char v93; // al
  char v94; // al
  int v95; // eax
  int v96; // eax
  _BYTE *v97; // rdi
  int v98; // eax
  char v99; // al
  unsigned __int16 v100; // ax
  __int16 v101; // r10
  __int16 v102; // ax
  __int16 v103; // r9
  __int16 v104; // ax
  unsigned __int16 v105; // ax
  char *v106; // rax
  char *v107; // rcx
  __m128i v108; // xmm5
  __m128i v109; // xmm6
  __m128i v110; // xmm7
  __m128i v111; // xmm0
  __int64 v112; // rdi
  char v113; // al
  __int16 v114; // r9
  __int16 v115; // dx
  char *v116; // rax
  char v117; // al
  unsigned __int8 *v118; // r12
  __int16 v119; // r9
  __int16 v120; // dx
  _BYTE *v121; // rax
  unsigned __int8 v122; // dl
  bool v123; // cf
  bool v124; // zf
  bool v125; // cf
  bool v126; // zf
  _BYTE *v127; // rsi
  __int64 v128; // rcx
  __int64 v129; // rdx
  __int64 v130; // rdx
  __int64 v131; // rcx
  __int16 v132; // ax
  _QWORD *v133; // rcx
  __int64 v134; // rdx
  __int64 v135; // rdx
  __int64 v136; // rcx
  __int16 v137; // ax
  _BYTE *v138; // rdx
  __int64 v139; // rcx
  _DWORD *v140; // rcx
  __int64 v141; // rdx
  __int64 v142; // rcx
  unsigned __int16 v143; // ax
  _BYTE *v144; // rax
  unsigned __int16 v145; // ax
  unsigned __int16 v146; // ax
  unsigned __int16 v147; // ax
  unsigned __int16 v148; // dx
  __int64 v149; // rdx
  unsigned int v150; // r13d
  __int64 v151; // rdx
  __int64 v152; // rcx
  __int16 v153; // ax
  int v154; // eax
  int v155; // r10d
  _BYTE *v156; // r15
  int v157; // ecx
  _BYTE *v158; // rsi
  __int64 v159; // rdx
  unsigned __int64 v160; // rdi
  unsigned int v161; // eax
  __int64 v162; // rax
  unsigned __int16 v163; // [rsp+8h] [rbp-A8h]
  unsigned __int64 *v164; // [rsp+8h] [rbp-A8h]
  int v165; // [rsp+14h] [rbp-9Ch]
  __int64 v166; // [rsp+20h] [rbp-90h]
  __int64 v167; // [rsp+20h] [rbp-90h]
  int v168; // [rsp+20h] [rbp-90h]
  _BYTE *v169; // [rsp+28h] [rbp-88h]
  int v170; // [rsp+30h] [rbp-80h]
  int v171; // [rsp+34h] [rbp-7Ch]
  __int64 v172; // [rsp+48h] [rbp-68h]
  unsigned __int16 srca; // [rsp+58h] [rbp-58h]
  unsigned __int16 srcb; // [rsp+58h] [rbp-58h]
  unsigned __int16 srcd; // [rsp+58h] [rbp-58h]
  unsigned __int16 srcc; // [rsp+58h] [rbp-58h]
  unsigned __int16 srce; // [rsp+58h] [rbp-58h]
  unsigned __int16 srcf; // [rsp+58h] [rbp-58h]
  char *src; // [rsp+58h] [rbp-58h]
  int v180; // [rsp+6Ch] [rbp-44h] BYREF
  _BYTE *v181; // [rsp+70h] [rbp-40h] BYREF
  __int64 v182[7]; // [rsp+78h] [rbp-38h] BYREF

  v6 = &dword_4F061FC;
  if ( !dword_4F061FC )
    goto LABEL_11;
  v7 = qword_4F08560;
  if ( !qword_4F08560 )
  {
    if ( !unk_4D03E88 )
      goto LABEL_10;
    a2 = &dword_4D03D18;
    if ( dword_4D03D18 | unk_4D03D20 || (qword_4F061C0[7] & 8) != 0 )
      goto LABEL_10;
    goto LABEL_69;
  }
  v8 = *(_BYTE *)(qword_4F08560 + 26);
  if ( v8 == 7 )
    goto LABEL_11;
  if ( !unk_4D03E88 )
    goto LABEL_16;
  if ( !(unk_4D03D20 | dword_4D03D18) && (qword_4F061C0[7] & 8) == 0 )
  {
LABEL_69:
    sub_854430();
    v7 = qword_4F08560;
    v6 = &dword_4F061FC;
    if ( !unk_4D03E88 && !qword_4F08560 )
      goto LABEL_71;
    dword_4F061FC = 1;
    if ( qword_4F08560 )
      goto LABEL_15;
    goto LABEL_10;
  }
  do
  {
LABEL_15:
    v8 = *(_BYTE *)(v7 + 26);
LABEL_16:
    qword_4F08560 = *(_QWORD *)v7;
    if ( v8 != 3 )
    {
      if ( v8 == 4 && !unk_4D03D20 )
      {
        do
        {
          v7 = *(_QWORD *)v7;
          if ( !v7 )
          {
LABEL_657:
            sub_6851C0(0xCu, (_DWORD *)8);
            BUG();
          }
        }
        while ( *(_BYTE *)(v7 + 26) == 4 );
        sub_6851C0(0xCu, (_DWORD *)(v7 + 8));
        v8 = *(_BYTE *)(v7 + 26);
        qword_4F08560 = *(_QWORD *)v7;
      }
      v15 = *(_WORD *)(v7 + 24);
      v11 = (unsigned __int64 *)&qword_4F06410;
      qword_4F063F0 = *(_QWORD *)(v7 + 16);
      word_4F06418[0] = v15;
      v16 = *(_QWORD *)(v7 + 8);
      qword_4F06410 = 0;
      *(_QWORD *)dword_4F07508 = v16;
      v17 = *(_DWORD *)(v7 + 32);
      *(_QWORD *)&dword_4F063F8 = v16;
      dword_4F06650[0] = *(_DWORD *)(v7 + 28);
      dword_4F0664C = v17;
      unk_4F06640 = *(_QWORD *)(v7 + 40);
      qword_4F06408 = 0;
      unk_4F06400 = 0;
      switch ( v8 )
      {
        case 4:
          qword_4F06410 = *(const char **)(v7 + 48);
          qword_4F06408 = *(_QWORD *)(v7 + 56);
          goto LABEL_23;
        case 1:
          v28 = _mm_loadu_si128((const __m128i *)(v7 + 48));
          v29 = _mm_loadu_si128((const __m128i *)(v7 + 64));
          v30 = _mm_loadu_si128((const __m128i *)(v7 + 80));
          v31 = _mm_loadu_si128((const __m128i *)(v7 + 96));
          *(__m128i *)&qword_4D04A00 = v28;
          v32 = v28.m128i_i64[0];
          unk_4D04A10 = v29;
          xmmword_4D04A20 = v30;
          unk_4D04A30 = v31;
          if ( v28.m128i_i64[0] )
          {
            v33 = *(_BYTE *)(v28.m128i_i64[0] + 73);
            if ( (v33 & 0x20) != 0 )
            {
              if ( unk_4D03FE8 )
              {
                if ( (v33 & 0x40) != 0 )
                  break;
              }
              else
              {
                if ( !(unsigned int)((__int64 (*)(void))sub_889670)() )
                  break;
                v32 = qword_4D04A00;
              }
              sub_889E70(v32);
            }
          }
          break;
        case 6:
          unk_4F061F0 = *(_QWORD *)(v7 + 48);
          goto LABEL_23;
        case 2:
          sub_72A510(*(const __m128i **)(v7 + 48), xmmword_4F06300);
          break;
        case 8:
          sub_72A510(*(const __m128i **)(v7 + 48), xmmword_4F06300);
          sub_72A510(*(const __m128i **)(v7 + 56), xmmword_4F06220);
          v68 = *(char **)(v7 + 72);
          qword_4F06218 = *(_QWORD **)(v7 + 64);
          v69 = strlen(v68);
          sub_87A880(v68, v69);
          unk_4F06210 = *(_QWORD *)(v7 + 80);
          break;
        default:
LABEL_23:
          *(_QWORD *)v7 = qword_4F08558;
          qword_4F08558 = v7;
          if ( !qword_4F08560 )
          {
            if ( unk_4D03E88 )
              dword_4F061FC = 1;
            else
              dword_4F061FC = qword_4F08538 != 0;
          }
          v18 = word_4F06418[0];
          v19 = qword_4F06410;
          v172 = 0;
          goto LABEL_27;
      }
      v34 = *(_BYTE *)(v7 + 26);
      if ( v34 == 2 )
      {
        *(_QWORD *)(*(_QWORD *)(v7 + 48) + 120LL) = qword_4F08550;
        qword_4F08550 = *(_QWORD *)(v7 + 48);
      }
      else if ( v34 == 8 )
      {
        *(_QWORD *)(*(_QWORD *)(v7 + 48) + 120LL) = qword_4F08550;
        *(_QWORD *)(*(_QWORD *)(v7 + 56) + 120LL) = *(_QWORD *)(v7 + 48);
        qword_4F08550 = *(_QWORD *)(v7 + 56);
      }
      goto LABEL_23;
    }
    unk_4D03E88 = *(_QWORD *)(v7 + 48);
    *(_QWORD *)v7 = qword_4F08558;
    qword_4F08558 = v7;
    v7 = qword_4F08560;
  }
  while ( qword_4F08560 );
  word_4F06418[0] = 0;
  if ( unk_4D03E88 )
  {
    dword_4F061FC = 1;
    goto LABEL_10;
  }
LABEL_71:
  dword_4F061FC = qword_4F08538 != 0;
LABEL_10:
  v9 = qword_4F08538;
  if ( qword_4F08538 )
  {
    v35 = *(const __m128i **)(qword_4F08538 + 16);
    if ( v35 )
    {
      while ( 1 )
      {
        v36 = qword_4F061C0;
        do
        {
          v37 = v35;
          v35 = (const __m128i *)v35->m128i_i64[0];
          *(_QWORD *)(v9 + 16) = v35;
          v38 = v37[1].m128i_i8[10];
          if ( v38 != 3 )
          {
            if ( v38 == 4 && !unk_4D03D20 )
            {
              do
              {
                v37 = (const __m128i *)v37->m128i_i64[0];
                if ( !v37 )
                  goto LABEL_657;
              }
              while ( v37[1].m128i_i8[10] == 4 );
              sub_6851C0(0xCu, &v37->m128i_i32[2]);
              v9 = qword_4F08538;
              *(_QWORD *)(qword_4F08538 + 16) = v37->m128i_i64[0];
              v38 = v37[1].m128i_i8[10];
            }
            v39 = v37[1].m128i_u16[4];
            v11 = (unsigned __int64 *)&qword_4F06410;
            qword_4F063F0 = v37[1].m128i_i64[0];
            word_4F06418[0] = v39;
            v40 = v37->m128i_i64[1];
            qword_4F06410 = 0;
            *(_QWORD *)dword_4F07508 = v40;
            *(_QWORD *)&dword_4F063F8 = v40;
            dword_4F06650[0] = v37[1].m128i_u32[3];
            dword_4F0664C = v37[2].m128i_i32[0];
            unk_4F06640 = v37[2].m128i_i64[1];
            qword_4F06408 = 0;
            unk_4F06400 = 0;
            if ( v38 == 4 )
            {
              qword_4F06410 = (const char *)v37[3].m128i_i64[0];
              qword_4F06408 = v37[3].m128i_i64[1];
            }
            else
            {
              if ( v38 != 1 )
              {
                switch ( v38 )
                {
                  case 6:
                    unk_4F061F0 = v37[3].m128i_i64[0];
                    break;
                  case 2:
                    sub_72A510((const __m128i *)v37[3].m128i_i64[0], xmmword_4F06300);
                    v9 = qword_4F08538;
                    break;
                  case 8:
                    sub_72A510((const __m128i *)v37[3].m128i_i64[0], xmmword_4F06300);
                    sub_72A510((const __m128i *)v37[3].m128i_i64[1], xmmword_4F06220);
                    src = (char *)v37[4].m128i_i64[1];
                    strlen(src);
                    qword_4F06218 = (_QWORD *)sub_881010(src);
                    unk_4F06210 = v37[5].m128i_i64[0];
                    v9 = qword_4F08538;
                    break;
                }
                goto LABEL_90;
              }
              v108 = _mm_loadu_si128(v37 + 3);
              v109 = _mm_loadu_si128(v37 + 4);
              v110 = _mm_loadu_si128(v37 + 5);
              v111 = _mm_loadu_si128(v37 + 6);
              *(__m128i *)&qword_4D04A00 = v108;
              v112 = v108.m128i_i64[0];
              unk_4D04A10 = v109;
              xmmword_4D04A20 = v110;
              unk_4D04A30 = v111;
              if ( v108.m128i_i64[0] )
              {
                v113 = *(_BYTE *)(v108.m128i_i64[0] + 73);
                if ( (v113 & 0x20) != 0 )
                {
                  if ( unk_4D03FE8 )
                  {
                    if ( (v113 & 0x40) == 0 )
                      goto LABEL_421;
                  }
                  else
                  {
                    v154 = ((__int64 (*)(void))sub_889670)();
                    v9 = qword_4F08538;
                    if ( v154 )
                    {
                      v112 = qword_4D04A00;
LABEL_421:
                      sub_889E70(v112);
                      v9 = qword_4F08538;
                    }
                  }
                }
              }
            }
LABEL_90:
            v41 = v9;
            if ( v9 )
            {
              do
              {
                if ( *(_QWORD *)(v41 + 16) )
                  break;
                if ( *(_DWORD *)(v41 + 64) )
                  break;
                sub_7AEB70();
                v41 = qword_4F08538;
              }
              while ( qword_4F08538 );
            }
            v18 = word_4F06418[0];
            v19 = qword_4F06410;
            v172 = 0;
            goto LABEL_27;
          }
        }
        while ( (v36[7] & 8) != 0 );
        v70 = sub_853ED0(v37[3].m128i_i64[0]);
        v9 = qword_4F08538;
        unk_4D03E88 = v70;
        v35 = *(const __m128i **)(qword_4F08538 + 16);
      }
    }
    v11 = (unsigned __int64 *)&qword_4F06410;
    dword_4F06650[0] = 0;
    v19 = qword_4F06410;
    word_4F06418[0] = 9;
    dword_4F0664C = 0;
    if ( !qword_4F06410 )
    {
      qword_4F06200 = 0;
      goto LABEL_32;
    }
    v172 = 0;
    v18 = 9;
LABEL_152:
    if ( unk_4D0420C && (qword_4F06498 > (unsigned __int64)v19 || qword_4F06490 <= (unsigned __int64)v19) )
    {
      srca = v18;
      sub_7AF060(1);
      v19 = (const char *)*v11;
      v18 = srca;
    }
LABEL_29:
    unk_4F06400 = qword_4F06408 - (_QWORD)v19 + 1LL;
    goto LABEL_30;
  }
LABEL_11:
  v10 = dword_4F06650;
  v171 = 0;
  LODWORD(v172) = 0;
  v11 = (unsigned __int64 *)&qword_4F06410;
  dword_4F06648 += 2;
  v12 = (unsigned __int64)&dword_4F0664C;
  dword_4F06650[0] = dword_4F06648;
  dword_4F0664C = dword_4F06648;
  unk_4F06640 = 0;
LABEL_12:
  v13 = qword_4F06460;
  if ( *qword_4F06460 == 32 )
  {
    v67 = qword_4F06460 + 1;
    do
    {
      qword_4F06460 = v67;
      v13 = v67++;
    }
    while ( *(v67 - 1) == 32 );
  }
  while ( 2 )
  {
    *v11 = (unsigned __int64)v13;
    v14 = *v13;
    dword_4F084E8 = *(_DWORD *)&word_4F06480;
    switch ( v14 )
    {
      case 0:
        v95 = (unsigned __int8)v13[1];
        v12 = (unsigned __int8)v95 & 0xFD;
        if ( (v13[1] & 0xFD) == 1 || (_BYTE)v95 == 9 )
        {
          ((void (*)(void))sub_7BC390)();
          v13 = qword_4F06460;
          if ( *qword_4F06460 || (qword_4F06460[1] & 0xFD) != 1 )
            continue;
          goto LABEL_378;
        }
        switch ( (_BYTE)v95 )
        {
          case 2:
            v10 = &dword_4D03D18;
            v12 = unk_4F061F8 | dword_4D03D18;
            if ( !(unk_4F061F8 | dword_4D03D18) )
              goto LABEL_117;
            v144 = v13 + 2;
            v19 = v13++;
            qword_4F06460 = v144;
            v18 = 10;
            goto LABEL_284;
          case 4:
            qword_4F06460 = v13 + 2;
            goto LABEL_12;
          case 5:
          case 8:
            v13 += 2;
            qword_4F06460 = v13;
            *v11 = (unsigned __int64)v13;
            if ( (_BYTE)v95 == 8 )
              v172 = 0x100000001LL;
            else
              HIDWORD(v172) = 1;
            goto LABEL_44;
        }
        v12 = (unsigned int)(v95 - 6);
        if ( (unsigned __int8)(v95 - 6) > 1u && (unsigned __int8)(v95 - 10) > 3u )
          sub_721090();
        goto LABEL_117;
      case 9:
      case 11:
      case 12:
      case 13:
      case 32:
        goto LABEL_117;
      case 10:
        ((void (*)(void))sub_7BC390)();
        v13 = qword_4F06460;
        if ( *qword_4F06460 != 10 )
          continue;
LABEL_378:
        v97 = qword_4F06460;
        dword_4F17FA4 = 1;
        v98 = *(_DWORD *)&word_4F06480;
        *v11 = (unsigned __int64)qword_4F06460;
        dword_4F084E8 = v98;
        qword_4F06408 = v97 + 1;
        sub_7B2A40((unsigned __int64)v97, (__int64)&dword_4F063F8);
        *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
        if ( qword_4F06460[1] != 1 )
        {
          v19 = (const char *)*v11;
          xmmword_4F06300[4].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
          if ( v19 )
          {
            HIDWORD(v172) = 0;
            goto LABEL_151;
          }
          goto LABEL_552;
        }
        if ( (unsigned __int64)qword_4F06460 < qword_4F06498 + 2LL || *(qword_4F06460 - 2) )
          v148 = word_4F063FC[0] - 1;
        else
          v148 = word_4F063FC[0] - 2;
        word_4F063FC[0] = v148;
        v19 = (const char *)*v11;
        LOWORD(dword_4F07508[1]) = v148;
        xmmword_4F06300[4].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
        if ( !v19 )
        {
LABEL_552:
          HIDWORD(v172) = 0;
          v147 = 9;
          goto LABEL_553;
        }
        HIDWORD(v172) = 0;
LABEL_151:
        v47 = *(_QWORD *)&dword_4F063F8;
        xmmword_4F06300[7].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
        qword_4F063F0 = v47;
        v18 = 9;
        goto LABEL_152;
      case 33:
        v19 = v13;
        if ( v13[1] != 61 )
        {
          v18 = 38;
          goto LABEL_283;
        }
        v18 = 48;
        goto LABEL_298;
      case 34:
        if ( unk_4D03D00 )
          goto LABEL_413;
        dword_4F17FA4 = 1;
        if ( !dword_4F17FA0
          && (qword_4F06498 > (unsigned __int64)v13
           || (unsigned __int64)v13 >= qword_4F06490
           || unk_4F06458
           || dword_4F17F78) )
        {
          if ( (_DWORD)qword_4F061D0 )
          {
            *(_QWORD *)&dword_4F063F8 = qword_4F061D0;
          }
          else
          {
            a2 = &dword_4F063F8;
            sub_7B0EB0((unsigned __int64)v13, (__int64)&dword_4F063F8);
          }
        }
        else
        {
          v114 = (_WORD)v13 - qword_4F06498;
          dword_4F063F8 = unk_4F0647C;
          if ( *(_DWORD *)&word_4F06480 && (unsigned __int64)v13 < qword_4F06488[*(int *)&word_4F06480 - 1] )
            v115 = sub_7AB680((unsigned __int64)v13);
          else
            v115 = word_4F06480;
          word_4F063FC[0] = v114 + 1 - v115;
        }
        *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
        if ( unk_4D0420C && (qword_4F06498 > *v11 || *v11 >= qword_4F06490) )
          sub_7AF060(0);
        a1 = 17;
        ++qword_4F06460;
        v18 = sub_7B78D0(0x11u);
        goto LABEL_327;
      case 35:
        qword_4F06460 = v13 + 1;
        v44 = v14;
        v43 = (char *)qword_4F06498;
        goto LABEL_102;
      case 36:
        if ( !unk_4D04748 )
          goto LABEL_363;
        goto LABEL_43;
      case 37:
        v42 = v13[1];
        if ( v42 == 61 )
        {
          v19 = v13;
          v18 = 59;
          goto LABEL_298;
        }
        if ( v42 != 58 )
        {
          if ( v42 != 62 )
          {
LABEL_571:
            v19 = v13;
            v18 = 40;
LABEL_283:
            qword_4F06460 = v13 + 1;
            goto LABEL_284;
          }
          v19 = v13;
          if ( !unk_4D04388 )
          {
            v18 = 40;
            goto LABEL_283;
          }
          v18 = 74;
LABEL_298:
          v86 = v13 + 2;
          ++v13;
          qword_4F06460 = v86;
          goto LABEL_284;
        }
        if ( !unk_4D04388 )
          goto LABEL_571;
        if ( v13[2] != 58 || v13[3] == 58 || dword_4F077C4 != 2 )
        {
          v43 = (char *)qword_4F06498;
LABEL_101:
          qword_4F06460 = v13 + 2;
          v44 = v14;
          goto LABEL_102;
        }
        a2 = dword_4F07508;
        sub_684B30(0x2D3u, dword_4F07508);
        v13 = qword_4F06460;
        v44 = *qword_4F06460;
        v43 = (char *)qword_4F06498;
        ++qword_4F06460;
        if ( v44 == 37 )
          goto LABEL_101;
LABEL_102:
        if ( unk_4D03D04 && !unk_4F06420 )
        {
          a1 = (unsigned __int64)&v180;
          sub_821C60(&v180, a2, v43);
          if ( !v180 )
          {
            v19 = (const char *)*v11;
            v18 = 0;
            goto LABEL_142;
          }
          goto LABEL_12;
        }
        if ( dword_4D03D18 && !unk_4D03D10 )
        {
          v13 = qword_4F06460;
          v19 = (const char *)*v11;
          if ( *qword_4F06460 == 35 )
          {
            if ( v44 != 37 )
            {
              v18 = 69;
              ++qword_4F06460;
              goto LABEL_284;
            }
          }
          else if ( *qword_4F06460 == 37 && qword_4F06460[1] == 58 && v44 == 37 )
          {
            v18 = 69;
            v13 = qword_4F06460 + 1;
            qword_4F06460 += 2;
            goto LABEL_284;
          }
LABEL_634:
          --v13;
          v18 = 68;
          goto LABEL_284;
        }
        a1 = *v11;
        if ( unk_4D04954 && v13 != v43 || dword_4F17F78 | dword_4F17FA4 )
        {
          v19 = (const char *)*v11;
          unk_4F06208 = 10;
          if ( !unk_4D03D20 )
          {
            sub_7B0EB0(a1, (__int64)dword_4F07508);
            sub_684AC0(8u, 0xAu);
            v19 = (const char *)*v11;
          }
          v18 = 0;
          v13 = qword_4F06460 - 1;
LABEL_284:
          qword_4F06408 = v13;
LABEL_142:
          dword_4F17FA4 = 1;
          if ( !dword_4F17FA0
            && (qword_4F06498 > (unsigned __int64)v19
             || qword_4F06490 <= (unsigned __int64)v19
             || unk_4F06458
             || dword_4F17F78) )
          {
            if ( (_DWORD)qword_4F061D0 )
            {
              *(_QWORD *)&dword_4F063F8 = qword_4F061D0;
            }
            else
            {
              srcd = v18;
              sub_7B0EB0((unsigned __int64)v19, (__int64)&dword_4F063F8);
              v19 = (const char *)*v11;
              v18 = srcd;
            }
          }
          else
          {
            v101 = (_WORD)v19 - qword_4F06498;
            v102 = word_4F06480;
            dword_4F063F8 = unk_4F0647C;
            if ( *(_DWORD *)&word_4F06480 && qword_4F06488[*(int *)&word_4F06480 - 1] > (unsigned __int64)v19 )
            {
              srcc = v18;
              v102 = sub_7AB680((unsigned __int64)v19);
              v18 = srcc;
            }
            word_4F063FC[0] = v101 + 1 - v102;
          }
          HIDWORD(v172) = 0;
          *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
LABEL_149:
          xmmword_4F06300[4].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
          if ( v19 )
          {
            if ( v18 == 9 )
              goto LABEL_151;
LABEL_198:
            if ( !dword_4F17FA0
              && (qword_4F06408 < qword_4F06498 || qword_4F06408 >= qword_4F06490 || unk_4F06458 || dword_4F17F78) )
            {
              if ( (_DWORD)qword_4F061D0 )
              {
                v19 = (const char *)*v11;
                qword_4F063F0 = qword_4F061D0;
              }
              else
              {
                srce = v18;
                sub_7B0EB0(qword_4F06408, (__int64)&qword_4F063F0);
                v19 = (const char *)*v11;
                v18 = srce;
              }
            }
            else
            {
              v64 = qword_4F06408 - qword_4F06498;
              LODWORD(qword_4F063F0) = unk_4F0647C;
              v65 = word_4F06480;
              if ( *(_DWORD *)&word_4F06480 && qword_4F06408 < qword_4F06488[*(int *)&word_4F06480 - 1] )
              {
                srcb = v18;
                v73 = sub_7AB680(qword_4F06408);
                v18 = srcb;
                v65 = v73;
              }
              v66 = v64 + 1;
              v19 = (const char *)*v11;
              WORD2(qword_4F063F0) = v66 - v65;
            }
            xmmword_4F06300[7].m128i_i64[0] = qword_4F063F0;
LABEL_27:
            if ( v19 )
            {
              if ( !v18 )
                goto LABEL_29;
              goto LABEL_152;
            }
          }
LABEL_30:
          word_4F06418[0] = v18;
          qword_4F06200 = v172;
          if ( v18 == 156 )
            dword_4F084C0 = dword_4F06650[0];
          goto LABEL_32;
        }
        a2 = &dword_4D03CE4;
        if ( dword_4D03CE4 )
        {
          v13 = qword_4F06460;
          v19 = (const char *)*v11;
          goto LABEL_634;
        }
        dword_4F17FA4 = 1;
        if ( dword_4F17FA0 || a1 >= (unsigned __int64)v43 && a1 < qword_4F06490 && !unk_4F06458 )
        {
          v103 = a1 - (_WORD)v43;
          dword_4F063F8 = unk_4F0647C;
          v104 = word_4F06480;
          if ( *(_DWORD *)&word_4F06480 && a1 < qword_4F06488[*(int *)&word_4F06480 - 1] )
            v104 = sub_7AB680(a1);
          word_4F063FC[0] = v103 + 1 - v104;
        }
        else if ( (_DWORD)qword_4F061D0 )
        {
          *(_QWORD *)&dword_4F063F8 = qword_4F061D0;
        }
        else
        {
          a2 = &dword_4F063F8;
          sub_7B0EB0(a1, (__int64)&dword_4F063F8);
        }
        *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
        sub_859ED0();
LABEL_117:
        sub_7BC390(a1, a2, v12, v10);
        v13 = qword_4F06460;
        continue;
      case 38:
        v94 = v13[1];
        v19 = v13;
        if ( v94 == 38 )
        {
          v18 = 52;
        }
        else
        {
          if ( v94 != 61 )
          {
            v18 = 33;
            goto LABEL_283;
          }
          v18 = 64;
        }
        goto LABEL_298;
      case 39:
        v100 = sub_7B74C0(1);
        v19 = (const char *)*v11;
        v18 = v100;
        goto LABEL_142;
      case 40:
        v19 = v13;
        v18 = 27;
        goto LABEL_283;
      case 41:
        v19 = v13;
        v18 = 28;
        goto LABEL_283;
      case 42:
        v19 = v13;
        if ( v13[1] != 61 )
        {
          v18 = 34;
          goto LABEL_283;
        }
        v18 = 57;
        goto LABEL_298;
      case 43:
        v99 = v13[1];
        v19 = v13;
        if ( v99 == 43 )
        {
          v18 = 31;
        }
        else
        {
          if ( v99 != 61 )
          {
            v18 = 35;
            goto LABEL_283;
          }
          v18 = 60;
        }
        goto LABEL_298;
      case 44:
        v19 = v13;
        v18 = 67;
        goto LABEL_283;
      case 45:
        v89 = v13[1];
        if ( v89 == 45 )
        {
          v19 = v13;
          v18 = 32;
          goto LABEL_298;
        }
        if ( v89 != 62 )
        {
          v19 = v13;
          if ( v89 != 61 )
          {
            v18 = 36;
            goto LABEL_283;
          }
          v18 = 61;
          goto LABEL_298;
        }
        if ( dword_4F077C4 != 2 || v13[2] != 42 )
        {
          v19 = v13;
          v18 = 30;
          goto LABEL_298;
        }
        v116 = v13;
        v18 = 148;
        goto LABEL_455;
      case 46:
        v87 = (unsigned __int8)v13[1];
        v88 = (unsigned int)(v87 - 48);
        if ( (unsigned int)v88 <= 9 )
        {
          v146 = sub_7B40D0(a1, (__int64)a2, v88, (__int64)v10, (__int64)v6, a6);
          v19 = (const char *)*v11;
          v18 = v146;
          goto LABEL_142;
        }
        if ( (_BYTE)v87 != 46 )
        {
          if ( (_BYTE)v87 == 42 )
          {
            v19 = v13;
            if ( dword_4F077C4 == 2 )
            {
              v18 = 147;
              goto LABEL_298;
            }
LABEL_303:
            v18 = 29;
            goto LABEL_283;
          }
LABEL_428:
          v19 = v13;
          goto LABEL_303;
        }
        if ( v13[2] != 46 )
          goto LABEL_428;
        v116 = v13;
        v18 = 76;
        goto LABEL_455;
      case 47:
        v85 = v13[1];
        if ( v85 != 42 && (!unk_4D042A8 || v85 != 47)
          || (qword_4F06498 > (unsigned __int64)v13 || qword_4F06490 <= (unsigned __int64)v13)
          && (!unk_4D04954 || dword_4F084D8 || v85 != 42) )
        {
          goto LABEL_296;
        }
        ((void (*)(void))sub_7BC390)();
        v13 = qword_4F06460;
        if ( *qword_4F06460 != 47 )
          continue;
        *v11 = (unsigned __int64)qword_4F06460;
        dword_4F084E8 = *(_DWORD *)&word_4F06480;
        if ( unk_4D03D20 && v13[1] == 47 )
        {
          v19 = v13;
          v18 = 0;
        }
        else
        {
LABEL_296:
          v13 = qword_4F06460;
          v19 = (const char *)*v11;
          v18 = 39;
          if ( qword_4F06460[1] != 61 )
            goto LABEL_283;
          v18 = 58;
        }
        goto LABEL_298;
      case 48:
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 55:
      case 56:
      case 57:
        if ( unk_4D03CF8 )
        {
          v106 = v13;
          do
          {
            v107 = v106++;
            qword_4F06460 = v106;
          }
          while ( (unsigned int)(unsigned __int8)*v106 - 48 <= 9 );
          v19 = v13;
          v18 = 13;
          qword_4F06408 = v107;
          goto LABEL_142;
        }
        if ( dword_4F051C0[v13[1] + 128] )
        {
          qword_4F06408 = v13;
          sub_724C70((__int64)xmmword_4F06300, 1);
          v46 = 5;
          if ( unk_4D03D04 )
          {
            if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 || (v46 = 5, HIDWORD(qword_4F077B4)) )
              v46 = unk_4F06AC9;
          }
          xmmword_4F06380[0].m128i_i64[0] = (__int64)sub_72BA30(v46);
          if ( v14 == 48 )
            byte_4F063A9[0] |= 2u;
          sub_620DE0(word_4F063B0, (unsigned __int8)(v14 - 48));
          ++qword_4F06460;
          if ( !unk_4D03D04 )
            goto LABEL_141;
        }
        else
        {
          v18 = sub_7B40D0(a1, (__int64)a2, (__int64)dword_4F051C0, (__int64)v10, (__int64)v6, a6);
          if ( !unk_4D03D04 || v18 != 4 )
          {
LABEL_445:
            v19 = (const char *)*v11;
            goto LABEL_142;
          }
        }
        sub_7AC2A0();
LABEL_141:
        v19 = (const char *)*v11;
        v18 = 4;
        goto LABEL_142;
      case 58:
        v92 = v13[1];
        if ( v92 == 58 )
        {
          if ( dword_4F077C4 == 2 )
          {
            v19 = v13;
            v18 = 146;
            goto LABEL_298;
          }
          v19 = v13;
          if ( unk_4F07778 > 202310 )
          {
            v18 = 146;
            goto LABEL_298;
          }
        }
        else
        {
          v19 = v13;
          if ( v92 == 62 && unk_4D04388 )
          {
            v18 = 26;
            goto LABEL_298;
          }
        }
        v18 = 55;
        goto LABEL_283;
      case 59:
        v19 = v13;
        v18 = 75;
        goto LABEL_283;
      case 60:
        if ( unk_4D03CFC )
        {
LABEL_413:
          v105 = sub_7B7810();
          v19 = (const char *)*v11;
          v18 = v105;
          goto LABEL_142;
        }
        v93 = v13[1];
        if ( v93 != 60 )
        {
          if ( v93 != 61 )
          {
            if ( v93 != 37 )
            {
              if ( v93 == 58 )
              {
                if ( unk_4D04388 )
                {
                  if ( v13[2] != 58 || (v13[3] & 0xFB) == 0x3A || dword_4F077C4 != 2 )
                  {
                    v19 = v13;
                    v18 = 25;
                    goto LABEL_298;
                  }
                  if ( unk_4F07778 <= 201102 && !dword_4F07774
                    || HIDWORD(qword_4F077B4) && !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x9F5Fu )
                  {
                    sub_684B30(0x2D2u, dword_4F07508);
                    v13 = qword_4F06460;
                    v18 = 25;
                    v19 = (const char *)*v11;
                    goto LABEL_298;
                  }
                }
              }
              else if ( v93 == 63 )
              {
                v19 = v13;
                if ( dword_4F077BC )
                {
                  v18 = 70;
                  goto LABEL_298;
                }
                goto LABEL_348;
              }
              v19 = v13;
              goto LABEL_348;
            }
            v19 = v13;
            if ( unk_4D04388 )
            {
              v18 = 73;
              goto LABEL_298;
            }
LABEL_348:
            v18 = 43;
            goto LABEL_283;
          }
          if ( v13[2] != 62 || !unk_4D041C0 )
          {
            v19 = v13;
            v18 = 45;
            goto LABEL_298;
          }
          v116 = v13;
          v18 = 49;
LABEL_455:
          v19 = v13;
          v13 = v116 + 2;
          qword_4F06460 = v116 + 3;
          goto LABEL_284;
        }
        v117 = v13[2];
        if ( v117 == 61 )
        {
          v116 = v13;
          v18 = 62;
          goto LABEL_455;
        }
        if ( v117 == 60 )
        {
          if ( !dword_4F084C0 || dword_4F084C0 + 2 != dword_4F06650[0] )
          {
            v116 = v13;
            v18 = 72;
            goto LABEL_455;
          }
        }
        else if ( v117 == 32 && v13[3] == 60 && (!dword_4F084C0 || dword_4F084C0 + 2 != dword_4F06650[0]) )
        {
          v116 = v13 + 1;
          v18 = 72;
          goto LABEL_455;
        }
        v19 = v13;
        v18 = 41;
        goto LABEL_298;
      case 61:
        v19 = v13;
        if ( v13[1] != 61 )
        {
          v18 = 56;
          goto LABEL_283;
        }
        v18 = 47;
        goto LABEL_298;
      case 62:
        v90 = v13[1];
        if ( v90 != 62 )
        {
          if ( v90 == 61 )
          {
            v19 = v13;
            v18 = 46;
          }
          else
          {
            v19 = v13;
            if ( v90 != 63 || !dword_4F077BC )
            {
              v18 = 44;
              goto LABEL_283;
            }
            v18 = 71;
          }
          goto LABEL_298;
        }
        if ( v13[2] != 61 )
        {
          v19 = v13;
          v18 = 42;
          goto LABEL_298;
        }
        v116 = v13;
        v18 = 63;
        goto LABEL_455;
      case 63:
        v19 = v13;
        v18 = 54;
        goto LABEL_283;
      case 65:
      case 66:
      case 67:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
      case 74:
      case 75:
      case 77:
      case 78:
      case 79:
      case 80:
      case 81:
      case 83:
      case 84:
      case 86:
      case 87:
      case 88:
      case 89:
      case 90:
      case 95:
      case 97:
      case 98:
      case 99:
      case 100:
      case 102:
      case 103:
      case 104:
      case 106:
      case 107:
      case 108:
      case 110:
      case 111:
      case 112:
      case 113:
      case 114:
      case 115:
      case 116:
      case 118:
      case 119:
      case 120:
      case 121:
      case 122:
        goto LABEL_43;
      case 76:
        goto LABEL_41;
      case 82:
        HIDWORD(v172) = unk_4F07710;
        if ( !unk_4F07710 )
          goto LABEL_44;
        goto LABEL_41;
      case 85:
      case 117:
        HIDWORD(v172) = unk_4D043A8;
        if ( !unk_4D043A8 )
          goto LABEL_44;
LABEL_41:
        v21 = sub_7B7F70(v13);
        v22 = v21;
        if ( v21 == -1 )
        {
          v13 = (char *)*v11;
LABEL_43:
          HIDWORD(v172) = 0;
          goto LABEL_44;
        }
        if ( (v21 & 0x10) == 0 )
        {
          v18 = sub_7B74C0(v21);
          goto LABEL_445;
        }
        v160 = *v11;
        dword_4F17FA4 = 1;
        a2 = &dword_4F063F8;
        sub_7B2A40(v160, (__int64)&dword_4F063F8);
        *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
        if ( unk_4D0420C && (qword_4F06498 > *v11 || *v11 >= qword_4F06490) )
          sub_7AF060(0);
        v161 = v22 & 7;
        if ( v161 > 2 )
        {
          v161 = 1;
        }
        else
        {
          a2 = 0;
          if ( v161 != 2 )
            v161 = 0;
        }
        a1 = v22;
        qword_4F06460 += (int)(v161 + ((v22 >> 3) & 1) + 1);
        v18 = sub_7B78D0(v22);
LABEL_327:
        if ( !unk_4D03D20 )
          goto LABEL_328;
        if ( v18 != 7 )
        {
          HIDWORD(v172) = 0;
          v19 = (const char *)*v11;
          goto LABEL_149;
        }
        v118 = qword_4F06460;
        goto LABEL_495;
      case 91:
        v19 = v13;
        v18 = 25;
        goto LABEL_283;
      case 92:
        if ( (v13[1] & 0xDF) != 0x55 || !unk_4D042A0 )
          goto LABEL_363;
        goto LABEL_43;
      case 93:
        v19 = v13;
        v18 = 26;
        goto LABEL_283;
      case 94:
        v19 = v13;
        if ( v13[1] != 61 )
        {
          v18 = 50;
          goto LABEL_283;
        }
        v18 = 65;
        goto LABEL_298;
      case 101:
      case 105:
      case 109:
        v74 = *(_DWORD *)&word_4F06480;
        HIDWORD(v172) = unk_4D04458;
        if ( !unk_4D04458 )
          goto LABEL_44;
        HIDWORD(v172) = dword_4D03D18;
        if ( dword_4D03D18 )
        {
          HIDWORD(v172) = 0;
        }
        else if ( word_4F06418[0] == 171 || !dword_4F17FA4 )
        {
          v75 = v13 + 1;
          v76 = 5;
          v77 = "mport";
          v78 = (unsigned __int8)v14 < 0x69u;
          v79 = v14 == 105;
          if ( v14 != 105 )
          {
            v78 = (unsigned __int8)v14 < 0x65u;
            v79 = v14 == 101;
            if ( v14 == 101 )
            {
              v75 = v13 + 1;
              v76 = 5;
              v77 = "xport";
            }
            else
            {
              v78 = (unsigned __int8)v14 < 0x6Du;
              v79 = v14 == 109;
              if ( v14 != 109 )
                goto LABEL_44;
              v75 = v13 + 1;
              v76 = 5;
              v77 = "odule";
            }
          }
          do
          {
            if ( !v76 )
              break;
            v78 = *v75 < (unsigned __int8)*v77;
            v79 = *v75++ == (unsigned __int8)*v77++;
            --v76;
          }
          while ( v79 );
          if ( (!v78 && !v79) == v78 )
          {
            if ( dword_4F055C0[v13[6] + 128] )
            {
              v80 = v13;
              goto LABEL_266;
            }
            v77 = v13 + 6;
            if ( !(unsigned int)sub_7B3CF0((unsigned __int8 *)v13 + 6, 0, 0) )
            {
              v80 = qword_4F06460;
              v13 = (char *)*v11;
              v74 = *(_DWORD *)&word_4F06480;
LABEL_266:
              v81 = (unsigned int)dword_4F17FA0;
              v170 = unk_4D03D20;
              v165 = dword_4D03D1C;
              if ( !dword_4F17FA0
                && (qword_4F06498 > (unsigned __int64)v13
                 || (unsigned __int64)v13 >= qword_4F06490
                 || unk_4F06458
                 || dword_4F17F78) )
              {
                if ( (_DWORD)qword_4F061D0 )
                {
                  v182[0] = qword_4F061D0;
                  v82 = v80;
                }
                else
                {
                  v77 = v13;
                  sub_7B0EB0((unsigned __int64)v13, (__int64)v182);
                  v82 = qword_4F06460;
                }
              }
              else
              {
                v119 = (_WORD)v13 - qword_4F06498;
                LODWORD(v182[0]) = unk_4F0647C;
                if ( v74 && (v81 = (unsigned __int64)&qword_4F06488, (unsigned __int64)v13 < qword_4F06488[v74 - 1]) )
                {
                  v77 = v13;
                  v120 = sub_7AB680((unsigned __int64)v13);
                }
                else
                {
                  v120 = v74;
                }
                WORD2(v182[0]) = v119 + 1 - v120;
                v82 = v80;
              }
              unk_4D03D20 = 1;
              dword_4D03D18 = 1;
              v83 = &dword_4D03D1C;
              dword_4D03D1C = 0;
              v84 = *(unsigned __int8 *)v82;
              if ( (_BYTE)v84 != 101 )
              {
                if ( (_BYTE)v84 != 105 )
                {
                  if ( (_BYTE)v84 != 109 )
                    goto LABEL_275;
                  qword_4F06460 = v82 + 6;
                  sub_7BC390(v77, &dword_4D03D1C, v84, v81);
                  v168 = 0;
                  v169 = qword_4F06460;
LABEL_510:
                  v132 = sub_7B8B50(v77, v83, v130, v131);
                  LOBYTE(v133) = v132 == 55 || v132 == 1;
                  if ( !(_BYTE)v133 )
                  {
                    if ( v132 != 75 )
                    {
                      sub_684B10(0xC75u, v182, (__int64)"module");
                      *v11 = (unsigned __int64)v80;
                      goto LABEL_276;
                    }
                    goto LABEL_512;
                  }
                  if ( v132 == 75 )
                  {
LABEL_512:
                    v134 = 173;
LABEL_513:
                    v163 = v134;
                    sub_7BC390(v77, v83, v134, v133);
                    v137 = sub_7B8B50(v77, v83, v135, v136);
                    v18 = v163;
                    if ( v137 != 10 )
                      goto LABEL_514;
                    if ( v168 || v163 != 173 )
                    {
                      v19 = v80;
                      v155 = v74;
                      v156 = v169;
                      *v11 = (unsigned __int64)v80;
                    }
                    else
                    {
                      if ( unk_4D03CD8 != -1 )
                      {
                        sub_6851C0(0xC78u, v182);
                        *v11 = (unsigned __int64)v80;
                        goto LABEL_276;
                      }
                      v19 = v80;
                      v155 = v74;
                      v156 = v169;
                      *v11 = (unsigned __int64)v80;
                      v18 = 173;
                    }
                    qword_4F06460 = v156;
                    if ( v168 )
                      v18 = 171;
                    qword_4F06408 = v19 + 5;
                    dword_4D03D18 = 0;
                    unk_4D03D20 = v170;
                    dword_4D03D1C = v165;
                    *(_DWORD *)&word_4F06480 = v155;
                    goto LABEL_142;
                  }
                  v149 = 173;
LABEL_560:
                  v83 = &qword_4F06498;
                  if ( qword_4F06498 > *v11 || (v133 = &qword_4F06490, *v11 >= qword_4F06490) )
                  {
                    sub_6851C0(0xC77u, v182);
                    goto LABEL_275;
                  }
                  v164 = v11;
                  v150 = v149;
                  while ( 1 )
                  {
                    sub_7BC390(v77, &qword_4F06498, v149, v133);
                    v153 = sub_7B8B50(v77, &qword_4F06498, v151, v152);
                    if ( v153 == 75 )
                      break;
                    if ( (unsigned __int16)(v153 - 9) <= 1u )
                    {
                      v11 = v164;
LABEL_514:
                      sub_684B30(0xC76u, v182);
LABEL_275:
                      *v11 = (unsigned __int64)v80;
LABEL_276:
                      qword_4F06460 = v80;
                      v13 = (char *)v80;
                      dword_4D03D18 = 0;
                      unk_4D03D20 = v170;
                      dword_4D03D1C = v165;
                      *(_DWORD *)&word_4F06480 = v74;
                      goto LABEL_44;
                    }
                  }
                  v134 = v150;
                  v11 = v164;
                  goto LABEL_513;
                }
                qword_4F06460 = v82 + 6;
                sub_7BC390(v77, &dword_4D03D1C, v84, v81);
                v168 = 0;
                v169 = qword_4F06460;
LABEL_523:
                v143 = sub_7B8B50(v77, v83, v141, v142);
                if ( v143 > 0x37u || (v149 = 172, ((0x80080000000082uLL >> v143) & 1) == 0) )
                {
                  sub_684B10(0xC75u, v182, (__int64)"import");
                  *v11 = (unsigned __int64)v80;
                  goto LABEL_276;
                }
                goto LABEL_560;
              }
              qword_4F06460 = v82 + 6;
              ((void (*)(void))sub_7BC390)();
              v121 = qword_4F06460;
              v122 = *qword_4F06460;
              v169 = qword_4F06460;
              v123 = *qword_4F06460 < 0x69u;
              v124 = *qword_4F06460 == 105;
              if ( *qword_4F06460 == 105 )
              {
                v138 = qword_4F06460;
                v139 = 5;
                v77 = "mport";
                v83 = qword_4F06460 + 1;
                do
                {
                  if ( !v139 )
                    break;
                  v123 = *v83 < (unsigned __int8)*v77;
                  v124 = *v83++ == (unsigned __int8)*v77++;
                  --v139;
                }
                while ( v124 );
                if ( (!v123 && !v124) != v123 )
                  goto LABEL_275;
                v140 = dword_4F055C0;
                if ( dword_4F055C0[(char)qword_4F06460[6] + 128]
                  || (v77 = qword_4F06460 + 6, v83 = 0, !(unsigned int)sub_7B3CF0(qword_4F06460 + 6, 0, 0)) )
                {
                  qword_4F06460 += 6;
                  sub_7BC390(v77, v83, v138, v140);
                  v168 = 1;
                  goto LABEL_523;
                }
                v121 = qword_4F06460;
                v122 = *qword_4F06460;
              }
              v125 = v122 < 0x6Du;
              v126 = v122 == 109;
              if ( v122 != 109 )
                goto LABEL_275;
              v127 = v121 + 1;
              v128 = 5;
              v77 = "odule";
              do
              {
                if ( !v128 )
                  break;
                v125 = *v127 < (unsigned __int8)*v77;
                v126 = *v127++ == (unsigned __int8)*v77++;
                --v128;
              }
              while ( v126 );
              if ( (!v125 && !v126) != v125 )
                goto LABEL_275;
              v83 = dword_4F055C0;
              v129 = (char)v121[6] + 128;
              if ( !dword_4F055C0[v129] )
              {
                v83 = 0;
                v77 = v121 + 6;
                if ( (unsigned int)sub_7B3CF0(v121 + 6, 0, 0) )
                  goto LABEL_275;
              }
              qword_4F06460 += 6;
              sub_7BC390(v77, v83, v129, v128);
              v168 = 1;
              goto LABEL_510;
            }
            HIDWORD(v172) = 0;
            v13 = (char *)*v11;
          }
        }
LABEL_44:
        dword_4F17FA4 = 1;
        if ( !dword_4F17FA0
          && (qword_4F06498 > (unsigned __int64)v13
           || qword_4F06490 <= (unsigned __int64)v13
           || unk_4F06458
           || dword_4F17F78) )
        {
          if ( (_DWORD)qword_4F061D0 )
            *(_QWORD *)&dword_4F063F8 = qword_4F061D0;
          else
            sub_7B0EB0((unsigned __int64)v13, (__int64)&dword_4F063F8);
        }
        else
        {
          v23 = (_WORD)v13 - qword_4F06498;
          v24 = word_4F06480;
          dword_4F063F8 = unk_4F0647C;
          if ( *(_DWORD *)&word_4F06480 && qword_4F06488[*(int *)&word_4F06480 - 1] > (unsigned __int64)v13 )
            v24 = sub_7AB680((unsigned __int64)v13);
          word_4F063FC[0] = v23 + 1 - v24;
        }
        v25 = qword_4F06460;
        unk_4F061E4 = 0;
        *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
        while ( 1 )
        {
          while ( 1 )
          {
            while ( 1 )
            {
              v26 = *v25;
              v27 = v26 + 128;
              if ( !dword_4F05DC0[v27] )
                break;
              if ( dword_4F05DC0[v25[1] + 128] )
                v25 += 2;
              else
                ++v25;
              qword_4F06460 = v25;
            }
            if ( (_BYTE)v26 == 92 )
              break;
            if ( dword_4F055C0[v27] || !(unsigned int)sub_7B3CF0((unsigned __int8 *)v25, (int *)v182, 0) )
              goto LABEL_156;
            unk_4F061E4 = 1;
            if ( SLODWORD(v182[0]) > 1
              && qword_4F06498 <= (unsigned __int64)qword_4F06460
              && qword_4F06490 > (unsigned __int64)qword_4F06460 )
            {
              v25 = qword_4F06460 + 1;
              v62 = 1;
              do
              {
                ++v62;
                qword_4F06460 = v25 + 1;
                v63 = *(int *)&word_4F06480;
                ++*(_DWORD *)&word_4F06480;
                qword_4F06488[v63] = v25;
                v25 = qword_4F06460;
              }
              while ( SLODWORD(v182[0]) > v62 );
            }
            else
            {
              v25 = &qword_4F06460[SLODWORD(v182[0])];
              qword_4F06460 = v25;
            }
          }
          v45 = (char *)*v11;
          if ( (v25[1] & 0xDF) != 0x55 || !unk_4D042A0 )
            break;
          unk_4F061E4 = 1;
          sub_7B39D0((unsigned __int64 *)&qword_4F06460, 1, v25 == v45, 1);
          v171 = 1;
          v25 = qword_4F06460;
        }
LABEL_156:
        v48 = *v11;
        v49 = _mm_loadu_si128(xmmword_4F06660);
        v50 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v51 = _mm_loadu_si128(&xmmword_4F06660[2]);
        qword_4F06408 = qword_4F06460 - 1;
        v52 = &qword_4F06460[-v48 - 1];
        v53 = _mm_loadu_si128(&xmmword_4F06660[3]);
        *(__m128i *)&qword_4D04A00 = v49;
        v181 = &qword_4F06460[-v48];
        qword_4D04A08 = *(_QWORD *)&dword_4F063F8;
        unk_4D04A10 = v50;
        xmmword_4D04A20 = v51;
        unk_4D04A30 = v53;
        if ( dword_4D03D18 | unk_4D03D20 && !(unk_4D03D10 | dword_4D03D1C) )
        {
          if ( unk_4D041A0 && dword_4D0432C )
          {
            sub_7AC440();
            v79 = *v11 == 0;
            xmmword_4F06300[4].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
            if ( !v79 )
              goto LABEL_652;
          }
          else
          {
            xmmword_4F06300[4].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
            if ( v48 )
            {
LABEL_652:
              v18 = 1;
              goto LABEL_198;
            }
          }
          v147 = 1;
LABEL_553:
          word_4F06418[0] = v147;
          qword_4F06200 = v172;
          goto LABEL_32;
        }
        if ( unk_4D04788 && v52 == (_BYTE *)10 )
        {
          if ( !memcmp((const void *)v48, "__VA_ARGS__", 0xBu) )
          {
            v166 = v48;
            sub_6851C0(0x3C9u, dword_4F07508);
            v48 = v166;
          }
        }
        else if ( unk_4D041B8 && v52 == (_BYTE *)9 && !memcmp((const void *)v48, "__VA_OPT__", 0xAu) )
        {
          v167 = v48;
          sub_6851C0(0xB7Bu, dword_4F07508);
          v48 = v167;
        }
        if ( unk_4F061E4 )
          v48 = sub_7B3EE0((unsigned __int8 *)v48, &v181);
        v54 = sub_87A100(v48, v181, &qword_4D04A00);
        v55 = *(_BYTE *)(v54 + 73);
        if ( unk_4D041A0 )
        {
          if ( dword_4D0432C )
          {
            if ( (v55 & 0x10) == 0 )
            {
              sub_7AC440();
              v55 = *(_BYTE *)(v54 + 73);
              if ( !v171 )
              {
                v55 |= 0x10u;
                *(_BYTE *)(v54 + 73) = v55;
              }
            }
          }
        }
        if ( (v55 & 0x20) != 0 )
        {
          if ( unk_4D03FE8 )
          {
            if ( (v55 & 0x40) != 0 )
              goto LABEL_172;
          }
          else if ( !(unsigned int)sub_889670(v54) )
          {
            goto LABEL_172;
          }
          if ( !(dword_4D03D18 | unk_4D03D20) && (!qword_4F08560 || *(_BYTE *)(qword_4F08560 + 26) != 7) )
            sub_889E70(v54);
        }
LABEL_172:
        v56 = *(_QWORD *)(v54 + 32);
        if ( !unk_4D04968 )
          v56 = *(_QWORD *)(v54 + 24);
        v57 = v56;
        if ( !v56 )
        {
LABEL_195:
          if ( unk_4D03D04 )
          {
            v18 = sub_81B740(0);
            byte_4F063A9[0] |= 0x80u;
            v19 = (const char *)*v11;
            goto LABEL_149;
          }
          v18 = 1;
LABEL_197:
          v79 = *v11 == 0;
          xmmword_4F06300[4].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
          if ( v79 )
            goto LABEL_30;
          goto LABEL_198;
        }
        while ( 1 )
        {
          v58 = *(_BYTE *)(v57 + 80);
          if ( v58 == 1 )
            break;
          if ( v58 || unk_4D03D14 | unk_4D03D20 )
            goto LABEL_194;
          if ( !dword_4D03D18 || (*(_BYTE *)(v57 + 90) & 1) != 0 || unk_4D03D10 && unk_4D03D0C )
          {
            v59 = *(_WORD *)(v57 + 88);
          }
          else
          {
            if ( !unk_4D03D04 )
              goto LABEL_194;
            v59 = *(_WORD *)(v57 + 88);
            if ( (unsigned __int16)(v59 - 181) > 1u )
              goto LABEL_194;
          }
          if ( (*(_BYTE *)(v57 + 83) & 0x40) == 0 )
          {
            if ( v59 == 24 )
            {
              v71 = *(_DWORD *)(v57 + 92);
              if ( v71 )
              {
                v72 = 4;
                if ( dword_4D04964 )
                  v72 = byte_4F07472[0];
                sub_685440(v72, v71, v57);
                *(_DWORD *)(v57 + 92) = 0;
              }
            }
            else
            {
              if ( (unsigned __int16)(v59 - 181) <= 1u )
              {
                srcf = v59;
                sub_724C70((__int64)xmmword_4F06300, 1);
                xmmword_4F06380[0].m128i_i64[0] = sub_72C390();
                sub_620D80(word_4F063B0, srcf == 182);
                byte_4F063A9[0] |= 1u;
                v18 = srcf;
                goto LABEL_197;
              }
              if ( !(_DWORD)qword_4F077B4 )
              {
                v19 = (const char *)*v11;
                v18 = v59;
                goto LABEL_149;
              }
              if ( word_4F06418[0] != 101
                || (v60 = *(&off_4B6DFA0 + v59), *v60 != 95)
                || v60[1] != 95
                || v60[2] != 105
                || v60[3] != 115
                || v60[4] != 95 )
              {
                v19 = (const char *)*v11;
                v18 = v59;
                goto LABEL_149;
              }
              sub_685490(0x9F5u, (FILE *)&dword_4F063F8, v57);
              *(_BYTE *)(v57 + 83) |= 0x40u;
            }
          }
LABEL_194:
          v57 = *(_QWORD *)(v57 + 8);
          if ( !v57 )
            goto LABEL_195;
        }
        if ( HIDWORD(v172) || !dword_4D03D1C )
          goto LABEL_194;
        if ( unk_4D0420C && (qword_4F06498 > *v11 || *v11 >= qword_4F06490) )
          sub_7AF060(1);
        a2 = (unsigned int *)&v180;
        a1 = v57;
        v61 = sub_81B8F0(v57, &v180);
        if ( v180 )
          goto LABEL_12;
        if ( v61 == 1 )
          goto LABEL_194;
        v18 = v61;
        if ( v61 != 7 )
        {
          if ( !unk_4D03D04 || v61 != 4 )
          {
            HIDWORD(v172) = 0;
            v19 = (const char *)*v11;
            goto LABEL_149;
          }
          sub_7AC2A0();
          HIDWORD(v172) = 0;
          v18 = 4;
          goto LABEL_197;
        }
        a2 = (unsigned int *)unk_4D03D20;
        if ( !unk_4D03D20 )
        {
LABEL_328:
          if ( dword_4D03D18 && (HIDWORD(v172) = unk_4D03D10) == 0 || (HIDWORD(v172) = unk_4D03D08) == 0 )
          {
            v19 = (const char *)*v11;
            goto LABEL_149;
          }
          v145 = sub_7B8270(0);
          HIDWORD(v172) = 0;
          v19 = (const char *)*v11;
          v18 = v145;
          goto LABEL_27;
        }
        v118 = qword_4F06460;
LABEL_495:
        HIDWORD(v172) = dword_4F07718;
        if ( !dword_4F07718 || (HIDWORD(v172) = sub_7B3E40(a1, a2)) == 0 )
        {
          v18 = 7;
          goto LABEL_197;
        }
        if ( unk_4F07714 )
        {
          v182[0] = qword_4F06460 - v118;
          v162 = sub_7B3EE0(v118, v182);
          if ( (unsigned int)sub_7ABF60(v162, v182[0]) )
          {
            qword_4F06460 = v118;
            v18 = 7;
            HIDWORD(v172) = 0;
            goto LABEL_197;
          }
        }
        v79 = *v11 == 0;
        qword_4F06408 = qword_4F06460 - 1;
        xmmword_4F06300[4].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
        if ( !v79 )
        {
          HIDWORD(v172) = 0;
          v18 = 8;
          goto LABEL_198;
        }
        word_4F06418[0] = 8;
        qword_4F06200 = (unsigned int)v172;
LABEL_32:
        if ( *((_DWORD *)qword_4F061C0 + 2)
          && (qword_4F061C0[7] & 4) == 0
          && !dword_4F08518
          && *((_DWORD *)qword_4F061C0 + 3) < dword_4F06650[0] )
        {
          sub_7AE360((__int64)(qword_4F061C0 + 3));
          *((_DWORD *)qword_4F061C0 + 3) = dword_4F06650[0];
        }
        if ( !dword_4D03D18 )
          dword_4F064B8[0] = 1;
        return word_4F06418[0];
      case 123:
        v19 = v13;
        v18 = 73;
        goto LABEL_283;
      case 124:
        v91 = v13[1];
        v19 = v13;
        if ( v91 == 124 )
        {
          v18 = 53;
        }
        else
        {
          if ( v91 != 61 )
          {
            v18 = 51;
            goto LABEL_283;
          }
          v18 = 66;
        }
        goto LABEL_298;
      case 125:
        v19 = v13;
        v18 = 74;
        goto LABEL_283;
      case 126:
        v19 = v13;
        v18 = 37;
        goto LABEL_283;
      default:
        HIDWORD(v172) = dword_4F055C0[v14 + 128];
        if ( !HIDWORD(v172) && (unsigned int)sub_7B3CF0((unsigned __int8 *)v13, 0, 1) )
        {
          v13 = (char *)*v11;
          goto LABEL_44;
        }
LABEL_363:
        unk_4F06208 = 7;
        if ( !unk_4D03D20 )
        {
          sub_7B0EB0(*v11, (__int64)dword_4F07508);
          sub_684AC0(8u, 7u);
        }
        v13 = qword_4F06460;
        if ( !dword_4D0432C )
          goto LABEL_549;
        if ( (char)*qword_4F06460 < 0 )
        {
          v96 = sub_721AB0(qword_4F06460, v182, unk_4F064A8 == 0);
          if ( LODWORD(v182[0]) )
          {
            v13 = qword_4F06460;
LABEL_549:
            v19 = (const char *)*v11;
            v18 = 0;
            goto LABEL_283;
          }
          if ( v96 > 1
            && qword_4F06498 <= (unsigned __int64)qword_4F06460
            && qword_4F06490 > (unsigned __int64)qword_4F06460 )
          {
            v157 = 1;
            ++qword_4F06460;
            do
            {
              v158 = qword_4F06460;
              ++v157;
              ++qword_4F06460;
              v159 = *(int *)&word_4F06480;
              ++*(_DWORD *)&word_4F06480;
              qword_4F06488[v159] = v158;
            }
            while ( v96 != v157 );
            goto LABEL_369;
          }
        }
        else
        {
          v96 = 1;
        }
        qword_4F06460 += v96;
LABEL_369:
        v18 = 0;
        v19 = (const char *)*v11;
        v13 = qword_4F06460 - 1;
        goto LABEL_284;
    }
  }
}
