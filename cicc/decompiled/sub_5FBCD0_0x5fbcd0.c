// Function: sub_5FBCD0
// Address: 0x5fbcd0
//
__int64 __fastcall sub_5FBCD0(__m128i *a1, __int64 a2, __int64 a3, __int64 *a4, unsigned int a5)
{
  const __m128i *v5; // r14
  __int64 v8; // rcx
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // rax
  char v12; // cl
  _QWORD *v13; // rax
  char v14; // dl
  __int64 v15; // rax
  int v16; // r13d
  _QWORD *v17; // rax
  __int64 v18; // r15
  char v19; // al
  char *v20; // r15
  int v21; // r8d
  int v22; // eax
  __m128i v23; // xmm7
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // rcx
  __int64 v28; // r8
  __int8 v29; // al
  char v30; // al
  _QWORD *v31; // rax
  _QWORD *v32; // rax
  char v33; // al
  char v34; // al
  __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int8 v40; // al
  __int64 v41; // rdi
  char v42; // al
  __int8 v43; // al
  char v44; // al
  __int64 v45; // r9
  __int64 v46; // rcx
  __int64 v47; // rsi
  __int64 v48; // rax
  __int64 v49; // rdx
  char v50; // al
  _QWORD *v51; // rax
  __int64 v52; // rdx
  _QWORD *i; // rdi
  __int64 v55; // rax
  __int64 v56; // rax
  __int8 v57; // al
  __int64 v58; // rax
  __int64 v59; // rax
  _QWORD **v60; // r8
  __int64 v61; // rdx
  __int64 v62; // rdi
  __int64 v63; // rsi
  __int64 v64; // rax
  __int64 v65; // r12
  char v66; // al
  char v67; // al
  _BYTE *v68; // rax
  __int64 v69; // rax
  __int64 j; // rax
  _QWORD *v71; // r12
  _BYTE *v72; // rax
  _QWORD *v73; // rdi
  __int64 v74; // r12
  __int64 jj; // rax
  _BYTE *v76; // rdi
  _QWORD *v77; // r8
  char v78; // al
  __int64 v79; // r12
  __int64 v80; // rax
  int v81; // eax
  __int64 v82; // rax
  int v83; // eax
  __int64 v84; // rax
  _BYTE *v85; // r10
  __int64 v86; // r11
  _QWORD **v87; // rax
  _QWORD *v88; // r12
  _BYTE *v89; // r13
  __int64 v90; // r14
  __int64 v91; // rax
  __int64 v92; // rdi
  __int64 v93; // rax
  __int64 v94; // r14
  __int64 ii; // rax
  __int64 v96; // rax
  char v97; // al
  char v98; // al
  __int64 v99; // rdi
  __int64 v100; // rax
  const char *v101; // r8
  __int64 v102; // rax
  __int64 v103; // rax
  char v104; // al
  __int64 v105; // rax
  char v106; // cl
  __int64 v107; // rax
  __int64 v108; // rax
  char v109; // al
  char v110; // al
  __int64 v111; // rax
  __int64 v112; // rax
  int v113; // eax
  _BYTE *v114; // rax
  __int64 v115; // rax
  __int64 v116; // r12
  __int64 v117; // r15
  int v118; // eax
  _BYTE *v119; // r12
  int v120; // r13d
  __int64 v121; // rbx
  __int64 v122; // r14
  __int64 v123; // rdi
  __int64 k; // rax
  __int64 v125; // r11
  __int64 v126; // r14
  __int64 v127; // rax
  char v128; // al
  __int64 v129; // rdx
  __int64 v130; // rdx
  char m; // al
  int v132; // eax
  int v133; // eax
  int v134; // eax
  int v135; // eax
  __int64 v136; // rax
  char v137; // dl
  int v138; // eax
  _BYTE *v139; // rax
  int v140; // eax
  unsigned int v141; // eax
  __int64 v142; // rax
  char v143; // dl
  __int64 v144; // rax
  int v145; // eax
  int v146; // eax
  __int64 n; // rax
  int v148; // eax
  unsigned int v149; // eax
  int v150; // eax
  int v151; // eax
  unsigned int v152; // [rsp+0h] [rbp-100h]
  __int64 v153; // [rsp+0h] [rbp-100h]
  __int64 v154; // [rsp+8h] [rbp-F8h]
  __int64 v155; // [rsp+8h] [rbp-F8h]
  unsigned int v156; // [rsp+8h] [rbp-F8h]
  __int64 v157; // [rsp+8h] [rbp-F8h]
  __int64 v158; // [rsp+10h] [rbp-F0h]
  __int64 v159; // [rsp+10h] [rbp-F0h]
  __int64 v160; // [rsp+18h] [rbp-E8h]
  __int64 v161; // [rsp+18h] [rbp-E8h]
  int v162; // [rsp+18h] [rbp-E8h]
  __int64 v163; // [rsp+18h] [rbp-E8h]
  __int64 v164; // [rsp+18h] [rbp-E8h]
  __int64 v165; // [rsp+18h] [rbp-E8h]
  __int64 *v166; // [rsp+20h] [rbp-E0h]
  __int64 v167; // [rsp+20h] [rbp-E0h]
  __int64 v168; // [rsp+20h] [rbp-E0h]
  char v169; // [rsp+20h] [rbp-E0h]
  __int64 v170; // [rsp+20h] [rbp-E0h]
  _QWORD **v171; // [rsp+28h] [rbp-D8h]
  __int64 v172; // [rsp+30h] [rbp-D0h]
  __int64 *v173; // [rsp+38h] [rbp-C8h]
  const char *v174; // [rsp+40h] [rbp-C0h]
  _QWORD **v175; // [rsp+40h] [rbp-C0h]
  __int32 v176; // [rsp+48h] [rbp-B8h]
  const __m128i *v177; // [rsp+48h] [rbp-B8h]
  __int16 v178; // [rsp+50h] [rbp-B0h]
  __int64 v179; // [rsp+50h] [rbp-B0h]
  __int64 v180; // [rsp+58h] [rbp-A8h]
  __int64 v181; // [rsp+58h] [rbp-A8h]
  __int64 v182; // [rsp+58h] [rbp-A8h]
  const char *v183; // [rsp+58h] [rbp-A8h]
  __int64 v184; // [rsp+58h] [rbp-A8h]
  __int64 v185; // [rsp+58h] [rbp-A8h]
  __int64 v186; // [rsp+58h] [rbp-A8h]
  __int64 v187; // [rsp+60h] [rbp-A0h]
  __int64 v188; // [rsp+60h] [rbp-A0h]
  char v189; // [rsp+6Fh] [rbp-91h]
  __int64 v190; // [rsp+70h] [rbp-90h]
  _QWORD **v191; // [rsp+70h] [rbp-90h]
  _QWORD **v192; // [rsp+70h] [rbp-90h]
  _QWORD **v193; // [rsp+70h] [rbp-90h]
  __int64 v194; // [rsp+70h] [rbp-90h]
  __int64 v195; // [rsp+70h] [rbp-90h]
  __int64 v196; // [rsp+70h] [rbp-90h]
  __int64 v197; // [rsp+70h] [rbp-90h]
  __int64 v198; // [rsp+70h] [rbp-90h]
  _QWORD **v199; // [rsp+70h] [rbp-90h]
  __int64 v200; // [rsp+70h] [rbp-90h]
  __int64 v201; // [rsp+70h] [rbp-90h]
  __int64 v202; // [rsp+70h] [rbp-90h]
  unsigned int v203; // [rsp+78h] [rbp-88h]
  int v204; // [rsp+78h] [rbp-88h]
  _BYTE *v205; // [rsp+78h] [rbp-88h]
  _BYTE *v206; // [rsp+78h] [rbp-88h]
  _BYTE *v207; // [rsp+78h] [rbp-88h]
  _BYTE *v208; // [rsp+78h] [rbp-88h]
  _BYTE *v209; // [rsp+78h] [rbp-88h]
  __int64 v210; // [rsp+80h] [rbp-80h]
  _QWORD *v213; // [rsp+90h] [rbp-70h]
  _QWORD *v214; // [rsp+98h] [rbp-68h]
  _QWORD *v215; // [rsp+98h] [rbp-68h]
  _QWORD *v216; // [rsp+98h] [rbp-68h]
  _QWORD *v217; // [rsp+98h] [rbp-68h]
  __int8 *v218; // [rsp+A0h] [rbp-60h]
  __int64 v219; // [rsp+A8h] [rbp-58h]
  char v220[4]; // [rsp+BCh] [rbp-44h] BYREF
  __int64 v221; // [rsp+C0h] [rbp-40h] BYREF
  __int64 v222[7]; // [rsp+C8h] [rbp-38h] BYREF

  v5 = (const __m128i *)a2;
  v8 = *(_QWORD *)a3;
  v9 = *(_QWORD *)(*(_QWORD *)a3 + 168LL);
  v10 = *(_BYTE *)(*(_QWORD *)a3 + 140LL) == 12;
  v219 = *(_QWORD *)a3;
  v221 = 0;
  v180 = v9;
  v203 = *(_DWORD *)(*(_QWORD *)(v9 + 152) + 240LL);
  v214 = (_QWORD *)a4[36];
  if ( v10 )
  {
    v11 = v8;
    do
      v11 = *(_QWORD *)(v11 + 160);
    while ( *(_BYTE *)(v11 + 140) == 12 );
  }
  else
  {
    v11 = v219;
  }
  v12 = *((_BYTE *)a4 + 269);
  v187 = *(_QWORD *)(*(_QWORD *)v11 + 96LL);
  v13 = (_QWORD *)a4[36];
  v14 = *((_BYTE *)v214 + 140);
  if ( v14 == 12 )
  {
    do
      v13 = (_QWORD *)v13[20];
    while ( *((_BYTE *)v13 + 140) == 12 );
  }
  else
  {
    v13 = (_QWORD *)a4[36];
  }
  v15 = *(_QWORD *)v13[21];
  if ( v15 && (*(_BYTE *)(v15 + 35) & 1) != 0 )
  {
    if ( v12 == 2 )
    {
      v16 = 1;
      sub_684AA0(7, 3210, a4 + 4);
      v189 = 1;
    }
    else
    {
      v189 = 1;
      v16 = 1;
      if ( (a4[1] & 4) != 0 )
        sub_6851C0(3236, a4 + 12);
    }
  }
  else if ( v12 == 2 )
  {
    v189 = 1;
    v16 = 1;
  }
  else if ( (*((_BYTE *)a4 + 10) & 8) == 0 || (a4[70] & 2) != 0 )
  {
    v189 = 0;
    v16 = 0;
  }
  else
  {
    v189 = 0;
    v16 = unk_4D0487C;
    if ( unk_4D0487C )
    {
      for ( i = (_QWORD *)a4[36]; v14 == 12; v14 = *((_BYTE *)i + 140) )
        i = (_QWORD *)i[20];
      v189 = 0;
      v16 = 0;
      if ( v14 == 7 )
      {
        v55 = i[21];
        if ( *(_QWORD *)(v55 + 40) )
        {
          if ( (*(_BYTE *)(v55 + 18) & 1) == 0 )
          {
            v56 = sub_73EDA0(i, 0);
            a4[36] = v56;
            *(_BYTE *)(*(_QWORD *)(v56 + 168) + 18LL) |= 1u;
          }
        }
      }
    }
  }
  v218 = &a1->m128i_i8[8];
  if ( (a1[1].m128i_i8[0] & 0x40) != 0 )
  {
    sub_6851C0(2488, &a1->m128i_u64[1]);
    *a1 = _mm_loadu_si128(xmmword_4F06660);
    a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    v59 = *(_QWORD *)dword_4F07508;
    a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
    a1[1].m128i_i8[1] |= 0x20u;
    a1->m128i_i64[1] = v59;
  }
  sub_5F1DC0((__int64)a1, (__int64)a4, v16);
  *((_BYTE *)a4 + 122) = ((*(_BYTE *)(a2 + 64) & 4) != 0) | *((_BYTE *)a4 + 122) & 0xFE;
  v17 = v214;
  if ( *((_BYTE *)v214 + 140) == 12 )
  {
    do
      v17 = (_QWORD *)v17[20];
    while ( *((_BYTE *)v17 + 140) == 12 );
  }
  else
  {
    v17 = v214;
  }
  v210 = v17[21];
  sub_646070(v214, v219, a1);
  v18 = sub_5EDB40((__int64)a1, v219, (__int64)a4, &v221);
  v19 = *(_BYTE *)(v18 + 84) & 0xEF | (16 * ((*(_BYTE *)(v18 + 84) & 0x10) != 0 || unk_4D03A10 != 0));
  *(_BYTE *)(v18 + 84) = v19;
  if ( (v19 & 0x10) != 0 )
  {
    if ( !*(_QWORD *)(v18 + 88) )
      goto LABEL_23;
    v176 = a1->m128i_i32[2];
    v178 = a1->m128i_i16[6];
    v174 = *(const char **)(a1->m128i_i64[0] + 8);
    v190 = *(_QWORD *)(a1->m128i_i64[0] + 16);
    v20 = (char *)sub_7279A0(v190 + 22);
    v21 = unk_4D03A0C++;
    v22 = snprintf(v20, v190 + 21, "%s$$OMP_VARIANT%06d", v174, v21);
    *a1 = _mm_loadu_si128(xmmword_4F06660);
    a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    v23 = _mm_loadu_si128(&xmmword_4F06660[3]);
    a1->m128i_i32[2] = v176;
    a1->m128i_i16[6] = v178;
    a1[3] = v23;
    sub_878540(v20, v22);
    v24 = sub_5EDB40((__int64)a1, v219, (__int64)a4, &v221);
    *(_DWORD *)(v24 + 81) |= 0x10000001u;
    v18 = v24;
    ++unk_4F07488;
  }
  if ( *(_QWORD *)(v18 + 88) )
  {
    sub_6854C0(403, v218, v18);
    a1[1].m128i_i8[1] |= 0x20u;
    a1[1].m128i_i64[1] = 0;
    v18 = sub_647630(10, a1, unk_4F04C5C, 1);
  }
LABEL_23:
  if ( (*((_BYTE *)a4 + 133) & 2) != 0 )
    *(_BYTE *)(v18 + 104) |= 1u;
  *a4 = v18;
  v25 = 0xFFFFFFFFLL;
  if ( (a4[70] & 4) == 0 )
    v25 = v203;
  v26 = sub_646F50(v214, 2, v25);
  *(_BYTE *)(v26 + 207) = (2 * *((_BYTE *)a4 + 125)) & 0x10 | *(_BYTE *)(v26 + 207) & 0xEF;
  *(_QWORD *)(v18 + 88) = v26;
  if ( !unk_4D04538 && (unsigned int)sub_8D3C40(v219) )
    *(_BYTE *)(v26 + 198) |= 0x18u;
  sub_877D80(v26, v18);
  sub_877E20(v18, v26, v219);
  *(_BYTE *)(v26 + 88) = *(_BYTE *)(a3 + 12) & 3 | *(_BYTE *)(v26 + 88) & 0xFC;
  v29 = a1[1].m128i_i8[0];
  if ( (v29 & 8) != 0 )
  {
    sub_725ED0(v26, 5);
    v57 = a1[3].m128i_i8[8];
    *(_BYTE *)(v26 + 176) = v57;
    if ( v57 == 11 )
    {
      *(_BYTE *)(v219 + 179) |= 0x80u;
    }
    else if ( v57 == 42 && (*(_BYTE *)(v180 + 109) & 0x20) != 0 )
    {
      v58 = *(_QWORD *)(v26 + 152);
      *(_BYTE *)(v26 + 206) |= 2u;
      *(_QWORD *)(*(_QWORD *)(v58 + 168) + 8LL) = v26;
    }
  }
  else if ( (v29 & 0x10) != 0 )
  {
    sub_725ED0(v26, 3);
    if ( (*((_BYTE *)a4 + 125) & 8) != 0 )
      *(_BYTE *)(v187 + 178) |= 8u;
  }
  else
  {
    v30 = *((_BYTE *)a4 + 560);
    if ( (v30 & 2) != 0 )
    {
      sub_725ED0(v26, 1);
      if ( dword_4D04428 )
        sub_5E4BC0(v26, v219);
      *(_BYTE *)(v26 + 194) = (*((_BYTE *)a4 + 131) << 7) | *(_BYTE *)(v26 + 194) & 0x7F;
      if ( *((char *)a4 + 130) < 0 )
        *(_QWORD *)(v26 + 232) = a4[40];
    }
    else if ( (v30 & 8) != 0 )
    {
      sub_725ED0(v26, 2);
    }
  }
  v31 = (_QWORD *)a4[50];
  if ( v31 )
  {
    *(_QWORD *)(v26 + 216) = v31;
    a4[50] = 0;
  }
  sub_5F06F0(a4, a2, v218, v27, v28);
  if ( (*(_BYTE *)(v26 + 206) & 8) != 0 )
    *(_BYTE *)(a3 + 11) |= 1u;
  v32 = (_QWORD *)a4[1];
  if ( ((unsigned int)v32 & 0x180000) != 0 )
  {
    v10 = ((unsigned int)v32 & 0x100000) == 0;
    v33 = *(_BYTE *)(v26 + 193);
    if ( v10 )
      v34 = v33 | 1;
    else
      v34 = v33 | 4;
    *(_BYTE *)(v26 + 193) = v34;
    *(_BYTE *)(v26 + 193) = v34 | 2;
    if ( (*(_BYTE *)(v219 + 177) & 0x20) == 0 && (*(_BYTE *)(*(_QWORD *)v18 + 73LL) & 2) != 0 )
    {
      v101 = *(const char **)(*(_QWORD *)v18 + 8LL);
      if ( (!strcmp(v101, "allocate") || !strcmp(v101, "deallocate"))
        && (*(_BYTE *)(***(_QWORD ***)(v18 + 64) + 73LL) & 2) != 0 )
      {
        v183 = v101;
        v196 = **(_QWORD **)(v18 + 64);
        if ( (unsigned int)sub_879C10(v196, unk_4D049B8) )
        {
          if ( !strcmp(*(const char **)(*(_QWORD *)v196 + 8LL), "allocator") )
          {
            *(_BYTE *)(*(_QWORD *)(v18 + 88) + 193LL) |= 8u;
            sub_77FA10((unsigned int)(*v183 != 97) + 2);
          }
        }
      }
    }
    if ( !v189 )
      *(_BYTE *)(v187 + 183) |= 8u;
    sub_736C90(v26, 1);
  }
  if ( *(char *)(v26 + 192) >= 0 && (*(_BYTE *)(a2 + 64) & 2) != 0 )
    sub_736C90(v26, 1);
  if ( a5 )
    *(_BYTE *)(v26 + 193) |= 0x10u;
  *(_BYTE *)(v210 + 17) = *(_BYTE *)(v210 + 17) & 0x8F | 0x20;
  v35 = (*(_BYTE *)(a2 + 64) & 2) != 0;
  sub_5E68B0(v18, v35, (__int64)v218);
  if ( (a1[1].m128i_i8[1] & 0x20) == 0 )
  {
    v40 = v5[4].m128i_i8[1];
    if ( (v40 & 2) != 0 )
    {
      *(_BYTE *)(v26 + 192) |= 0x10u;
      v40 = v5[4].m128i_i8[1];
    }
    v36 = *(_BYTE *)(v26 + 192) & 0xDF;
    *(_BYTE *)(v26 + 192) = *(_BYTE *)(v26 + 192) & 0xDF | (8 * v40) & 0x20;
  }
  if ( unk_4D03CB8 )
  {
    if ( a5 )
      goto LABEL_55;
    sub_858370(a4 + 23);
  }
  else if ( a5 )
  {
    goto LABEL_55;
  }
  sub_8756F0((-(__int64)((v5[4].m128i_i8[0] & 4) == 0) & 0xFFFFFFFFFFFFFFFELL) + 3, v18, v218, v5[4].m128i_i64[1]);
  sub_729470(v26, a4 + 59);
  v60 = (_QWORD **)(a4 + 59);
  if ( (v5[4].m128i_i8[0] & 4) != 0 )
  {
    if ( dword_4F04C64 != -1
      && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0
      && dword_4F077C4 == 2
      && (*(_BYTE *)(v26 - 8) & 1) != 0
      && (a1[1].m128i_i8[2] & 0x40) == 0 )
    {
      v111 = sub_7CAFF0(a1, v26, qword_4F04C68);
      v60 = (_QWORD **)(a4 + 59);
      if ( v111 )
        *(_BYTE *)(v111 + 33) |= 0x10u;
    }
    v191 = v60;
    sub_64A300(v26, v5[5].m128i_i64[0]);
    v60 = v191;
    *(_BYTE *)(v26 + 173) = *((_BYTE *)a4 + 268);
    v5[5].m128i_i64[0] = *(_QWORD *)(v26 + 264);
  }
  if ( (v5[4].m128i_i16[0] & 0x1004) == 4 )
  {
    *(_BYTE *)(v26 + 90) = (16 * *((_BYTE *)a4 + 126)) & 0x40 | *(_BYTE *)(v26 + 90) & 0xBF;
  }
  else
  {
    v61 = 0;
    if ( dword_4F04C64 != -1
      && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0
      && dword_4F077C4 == 2
      && (*(_BYTE *)(v26 - 8) & 1) != 0
      && (a1[1].m128i_i8[2] & 0x40) == 0 )
    {
      v193 = v60;
      v69 = sub_7CAFF0(a1, v26, 0);
      v60 = v193;
      v61 = v69;
    }
    if ( (v5[4].m128i_i8[1] & 0x10) != 0 )
    {
      v62 = *(_QWORD *)(v26 + 264);
      v63 = *(_QWORD *)(v26 + 152);
      *(_BYTE *)(v26 + 203) |= 1u;
      if ( v62 != v63 )
      {
        v181 = v61;
        v192 = v60;
        v64 = sub_73EDA0(v62, 0);
        v60 = v192;
        v61 = v181;
        v5[5].m128i_i64[0] = v64;
        v63 = v64;
      }
    }
    else
    {
      v185 = v61;
      v199 = v60;
      v107 = sub_64A1A0(v214, v5[5].m128i_i64[0]);
      v61 = v185;
      v60 = v199;
      v5[5].m128i_i64[0] = v107;
      v63 = v107;
    }
    if ( !sub_86A3D0(v26, v63, v61, (*((_BYTE *)a4 + 126) & 4) == 0 ? 16 : 80, v60) )
      v5[5].m128i_i64[0] = 0;
  }
  sub_65C210(a4);
  sub_652340(a4);
  if ( (v5[4].m128i_i8[0] & 4) != 0 )
    sub_64F530(v18);
  v35 = 0;
  sub_854980(v18, 0);
LABEL_55:
  v41 = v221;
  if ( v221 )
  {
    v35 = (__int64)&a1->m128i_i64[1];
    sub_5EE660(v221, (__int64)v218);
    v41 = v221;
    sub_5EE4B0(v221);
  }
  v42 = *(_BYTE *)(v219 + 177);
  if ( (v42 & 0x40) != 0 )
    goto LABEL_61;
  if ( (v42 & 0x20) != 0 || (*(_BYTE *)(a3 + 9) & 1) != 0 )
  {
    if ( (*(_BYTE *)(v219 + 178) & 4) != 0 )
    {
LABEL_61:
      *(_BYTE *)(v26 + 195) |= 8u;
      goto LABEL_62;
    }
    v102 = sub_880C60();
    *(_QWORD *)(v18 + 96) = v102;
    *(_QWORD *)(v102 + 32) = v18;
    *(_QWORD *)(v102 + 24) = v18;
    v41 = *(unsigned __int8 *)(v18 + 80);
    v197 = v102;
    v103 = sub_87E420(v41);
    v37 = v197;
    v38 = v103;
    *(_QWORD *)(v197 + 56) = v103;
    *(_QWORD *)(v103 + 176) = v26;
    *(__m128i *)(v103 + 184) = _mm_loadu_si128(v5);
    *(__m128i *)(v103 + 200) = _mm_loadu_si128(v5 + 1);
    *(__m128i *)(v103 + 216) = _mm_loadu_si128(v5 + 2);
    *(__m128i *)(v103 + 232) = _mm_loadu_si128(v5 + 3);
    *(__m128i *)(v103 + 248) = _mm_loadu_si128(v5 + 4);
    *(__m128i *)(v103 + 264) = _mm_loadu_si128(v5 + 5);
    *(_QWORD *)(v103 + 280) = v5[6].m128i_i64[0];
    if ( *((_BYTE *)v214 + 140) == 7 )
    {
      v114 = *(_BYTE **)(v210 + 56);
      if ( v114 )
      {
        if ( (*v114 & 0x20) != 0 )
        {
          v186 = v197;
          v202 = v38;
          v115 = sub_8661A0(1);
          v41 = v202 + 336;
          v35 = *(_QWORD *)(*(_QWORD *)(v210 + 56) + 8LL);
          sub_879080(v202 + 336, v35, v115);
          v37 = v186;
          v38 = v202;
          *(_BYTE *)(qword_4CF8008 + 184) |= 8u;
        }
      }
    }
    v104 = (8 * (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1)) | *(_BYTE *)(v38 + 160) & 0xF7;
    *(_BYTE *)(v38 + 160) = v104;
    v36 = *(_BYTE *)(v26 + 195) & 0xF6;
    *(_BYTE *)(v26 + 195) = *(_BYTE *)(v26 + 195) & 0xF6 | (8 * ((unsigned __int8)~v104 >> 7) + 1);
    *(_QWORD *)(v37 + 64) = v5->m128i_i64[0];
    v5[4].m128i_i8[1] |= 8u;
    if ( (a4[70] & 4) == 0 )
    {
      v184 = v38;
      v105 = sub_727340();
      *(_BYTE *)(v105 + 120) = 4;
      v198 = v105;
      sub_877D80(v105, v18);
      sub_877E20(0, v198, v219);
      *(_BYTE *)(v198 + 88) = *(_BYTE *)(v198 + 88) & 0x8F | 0x20;
      *(_QWORD *)(v184 + 104) = v198;
      v106 = 0;
      *(_BYTE *)(v198 + 88) = *(_BYTE *)(a3 + 12) & 3 | *(_BYTE *)(v198 + 88) & 0xFC;
      if ( *(char *)(v219 + 177) < 0 && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v219 + 168) + 160LL) + 121LL) & 1) != 0 )
        v106 = (((unsigned __int8)v5[4].m128i_i8[0] >> 1) ^ 1) & 1;
      v35 = v203;
      v41 = v198;
      *(_BYTE *)(v198 + 121) = v106 | *(_BYTE *)(v198 + 121) & 0xFE;
      sub_7344C0(v198, v203);
      v36 = (__int64)&dword_4F07590;
      if ( dword_4F07590 || (v38 = v184, *(char *)(v184 + 160) < 0) )
        *(_QWORD *)(v198 + 192) = v26;
      *(_QWORD *)(v198 + 200) = v198;
      if ( (v5[4].m128i_i8[0] & 4) != 0 )
        *(_QWORD *)(v198 + 208) = v198;
      *(_QWORD *)(v26 + 248) = v198;
    }
  }
  else if ( *((_BYTE *)v214 + 140) == 7 )
  {
    v68 = *(_BYTE **)(v210 + 56);
    if ( v68 )
    {
      if ( (*v68 & 0x20) != 0 )
        *(_BYTE *)(qword_4CF8008 + 184) |= 8u;
    }
  }
LABEL_62:
  if ( (a1[1].m128i_i8[1] & 0x20) != 0 )
    goto LABEL_88;
  v43 = a1[1].m128i_i8[0];
  if ( (v43 & 8) != 0 )
  {
    v44 = *(_BYTE *)(v26 + 176);
    switch ( v44 )
    {
      case 15:
        v36 = v221;
        v84 = *(_QWORD *)(v187 + 32);
        if ( v84 )
        {
          if ( *(_BYTE *)(v84 + 80) != 17 )
            *(_QWORD *)(v187 + 32) = v221;
        }
        else
        {
          if ( !v221 )
            v36 = v18;
          *(_QWORD *)(v187 + 32) = v36;
        }
        break;
      case 1:
        *(_BYTE *)(v187 + 179) |= 4u;
        break;
      case 3:
        *(_BYTE *)(v187 + 179) |= 8u;
        break;
      case 2:
        v35 = (__int64)(a4 + 36);
        v41 = v26;
        *(_BYTE *)(v187 + 179) |= 0x10u;
        sub_5F93D0(v26, a4 + 36);
        break;
      case 4:
        v35 = (__int64)(a4 + 36);
        v41 = v26;
        *(_BYTE *)(v187 + 179) |= 0x20u;
        sub_5F93D0(v26, a4 + 36);
        break;
    }
  }
  else if ( (v43 & 0x10) != 0 )
  {
    if ( sub_5F1C40(*(_QWORD *)(v26 + 152)) )
    {
      v35 = v187;
      sub_5E6C40(v18, v187);
      v41 = sub_73D790(*(_QWORD *)(v26 + 152));
      v65 = v41;
      if ( (unsigned int)sub_8D32E0(v41) )
        v65 = sub_8D46C0(v41);
      while ( 1 )
      {
        v66 = *(_BYTE *)(v65 + 140);
        if ( v66 != 12 )
          break;
        v65 = *(_QWORD *)(v65 + 160);
      }
      if ( (unsigned __int8)(v66 - 9) <= 2u )
      {
        v41 = v65;
        sub_5E48B0(v65);
      }
    }
    else
    {
      v35 = (__int64)&a1->m128i_i64[1];
      v41 = 554;
      sub_685490(554, v218, v18);
    }
  }
  if ( !dword_4D048B8 || *((char *)a4 + 130) < 0 )
    goto LABEL_72;
  if ( (*(_BYTE *)(v219 + 177) & 0x20) == 0 )
  {
    for ( j = *(_QWORD *)(v26 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v71 = *(_QWORD **)(j + 168);
    if ( !v71[7] )
    {
      if ( (*(_BYTE *)(v26 + 193) & 0x10) == 0 )
      {
        v39 = (unsigned int)dword_4D048B0;
        if ( !dword_4D048B0 || *(_BYTE *)(v26 + 174) != 5 )
          goto LABEL_153;
        goto LABEL_307;
      }
      v113 = *(unsigned __int8 *)(v26 + 174);
      v36 = (unsigned int)(v113 - 1);
      if ( (unsigned __int8)(v113 - 1) <= 1u )
      {
        v36 = (__int64)&dword_4D04964;
        if ( !dword_4D04964
          || (_BYTE)v113 == 1 && !*v71 && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v219 + 96LL) + 183LL) & 0x60) != 0 )
        {
          goto LABEL_387;
        }
      }
      else
      {
        if ( (_BYTE)v113 != 5 )
          goto LABEL_153;
        if ( *(_BYTE *)(v26 + 176) != 15 )
        {
          if ( !dword_4D048B0 )
            goto LABEL_153;
LABEL_307:
          if ( ((*(_BYTE *)(v26 + 176) - 2) & 0xFD) == 0 )
            sub_5F1D90((__int64)v71);
          goto LABEL_153;
        }
        if ( !dword_4D04964 )
        {
LABEL_387:
          v139 = (_BYTE *)sub_725E60(v41, v35, v36);
          *v139 |= 0xAu;
          v71[7] = v139;
          v71[1] = v26;
          goto LABEL_153;
        }
      }
      sub_5F8DB0(v26, 0);
    }
  }
LABEL_153:
  v72 = *(_BYTE **)(v210 + 56);
  if ( v72 && (*v72 & 0x22) == 0 )
  {
    v73 = v214;
    if ( *((_BYTE *)v214 + 140) == 12 )
    {
      do
        v73 = (_QWORD *)v73[20];
      while ( *((_BYTE *)v73 + 140) == 12 );
    }
    else
    {
      v73 = v214;
    }
    if ( (unsigned int)sub_8D76D0(v73) )
      *(_BYTE *)(v26 + 195) |= 0x10u;
  }
LABEL_72:
  if ( (*(_BYTE *)(v18 + 81) & 0x20) != 0 )
    goto LABEL_73;
  v67 = 0;
  if ( (a4[1] & 4) != 0 )
    v67 = ((*((_BYTE *)a4 + 560) >> 4) ^ 1) & 1;
  v38 = a5;
  if ( a5 && (*(_BYTE *)(v26 + 195) & 8) != 0 )
  {
    if ( (*(_BYTE *)(v26 + 192) & 2) != 0 )
    {
      v100 = *(_QWORD *)(v26 + 216);
      if ( v100 )
        goto LABEL_232;
    }
    sub_644920(a4, (v5[4].m128i_i8[0] & 4) != 0, v36, v37, a5, v39);
    goto LABEL_78;
  }
  v37 = a3;
  v171 = (_QWORD **)(a4 + 6);
  v85 = *(_BYTE **)(*a4 + 88);
  v172 = *a4;
  v86 = *(_QWORD *)a3;
  v194 = *(_QWORD *)*a4;
  v36 = v85[192] & 0xFD;
  v10 = (v85[193] & 0x10) == 0;
  v85[192] = v85[192] & 0xFD | (2 * v67);
  v87 = (_QWORD **)(v172 + 48);
  if ( v10 )
    v87 = (_QWORD **)(a4 + 6);
  v175 = v87;
  v173 = (__int64 *)(a3 + 16);
  v222[0] = *(_QWORD *)(a3 + 16);
  v39 = **(_QWORD **)(v86 + 168);
  if ( !v39 )
    goto LABEL_215;
  v182 = v26;
  v88 = **(_QWORD ***)(v86 + 168);
  v89 = v85;
  v204 = 0;
  v188 = v18;
  v179 = v86;
  v177 = v5;
LABEL_190:
  while ( 2 )
  {
    v36 = v88[5];
    if ( v89[174] == 2 )
    {
      if ( (*(_BYTE *)(v36 + 177) & 0x20) == 0 )
      {
        while ( *(_BYTE *)(v36 + 140) == 12 )
          v36 = *(_QWORD *)(v36 + 160);
        v36 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v36 + 96LL) + 24LL);
        if ( v36 && (*(_BYTE *)(*(_QWORD *)(v36 + 88) + 192LL) & 2) != 0 )
        {
          sub_5F81D0(a3, (__int64)a4, v36, (__int64)v88, 0);
          *((_BYTE *)a4 + 129) |= 1u;
          v204 = 1;
        }
        goto LABEL_189;
      }
LABEL_188:
      *((_BYTE *)a4 + 129) |= 1u;
      goto LABEL_189;
    }
    v90 = *(_QWORD *)(*(_QWORD *)(v36 + 168) + 152LL);
    v91 = v88[5];
    if ( *(_BYTE *)(v36 + 140) == 12 )
    {
      do
        v91 = *(_QWORD *)(v91 + 160);
      while ( *(_BYTE *)(v91 + 140) == 12 );
    }
    if ( !v90 || (*(_BYTE *)(v90 + 29) & 0x20) != 0 || (*(_BYTE *)(v36 + 177) & 0x20) != 0 )
      goto LABEL_188;
    v92 = *(_QWORD *)(*(_QWORD *)v91 + 96LL);
    if ( (v89[195] & 8) != 0 )
      *((_BYTE *)a4 + 129) |= 1u;
    v93 = sub_883800(v92 + 192, v194);
    if ( !v93 )
      goto LABEL_189;
    v37 = *(unsigned int *)(v90 + 24);
    while ( 1 )
    {
      v94 = v93;
      v93 = *(_QWORD *)(v93 + 32);
      if ( *(_DWORD *)(v94 + 40) != (_DWORD)v37 )
        goto LABEL_200;
      v36 = *(unsigned __int8 *)(v94 + 80);
      if ( (_BYTE)v36 == 17 )
        break;
      if ( (_BYTE)v36 == 10 )
      {
        v117 = v94;
        v39 = 0;
        goto LABEL_324;
      }
      v36 = (unsigned int)(v36 - 8);
      if ( (unsigned __int8)v36 <= 1u )
      {
        v88 = (_QWORD *)*v88;
        if ( !v88 )
          goto LABEL_206;
        goto LABEL_190;
      }
LABEL_200:
      if ( !v93 )
        goto LABEL_189;
    }
    v117 = *(_QWORD *)(v94 + 88);
    if ( !v117 )
      goto LABEL_189;
    v36 = *(unsigned __int8 *)(v117 + 80);
    v39 = 1;
LABEL_324:
    v38 = 0;
    v160 = (__int64)v88;
    v118 = 0;
    v119 = v89;
    v166 = a4;
    v120 = v39;
    v121 = 0;
    v158 = v94;
    while ( 2 )
    {
      if ( v120 )
        v121 = *(_QWORD *)(v117 + 8);
      if ( (_BYTE)v36 != 10 )
        goto LABEL_326;
      v122 = *(_QWORD *)(v117 + 88);
      if ( (*(_BYTE *)(v122 + 192) & 2) == 0 )
        goto LABEL_326;
      if ( !(unsigned int)sub_8DE890(*((_QWORD *)v119 + 19), *(_QWORD *)(v122 + 152), 0x2000, 0) )
        goto LABEL_325;
      v123 = *((_QWORD *)v119 + 19);
      for ( k = v123; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      if ( !*(_QWORD *)(*(_QWORD *)(k + 168) + 40LL) )
      {
        v18 = v188;
        v26 = v182;
        v5 = v177;
        a4 = v166;
        sub_6851C0(314, v175);
        v86 = v179;
        v85 = v119;
        goto LABEL_207;
      }
      if ( !(unsigned int)sub_8D7820(v123, *(_QWORD *)(v122 + 152), 0, 0) )
      {
LABEL_325:
        v118 = 1;
LABEL_326:
        if ( !v121 )
        {
          v89 = v119;
          v88 = (_QWORD *)v160;
          a4 = v166;
          if ( v118 && (v89[193] & 0x10) == 0 )
            sub_5E6A70(v173, v222, v158, v172, v160);
          goto LABEL_189;
        }
        v36 = *(unsigned __int8 *)(v121 + 80);
        v117 = v121;
        continue;
      }
      break;
    }
    v125 = v122;
    v89 = v119;
    v126 = v158;
    v127 = *((_QWORD *)v119 + 19);
    v88 = (_QWORD *)v160;
    a4 = v166;
    v159 = *(_QWORD *)(v125 + 152);
    v161 = *(_QWORD *)(v127 + 160);
    v167 = *(_QWORD *)(v159 + 160);
    if ( (unsigned int)sub_8D97D0(v161, v167, 512, v37, v161)
      || (unsigned int)sub_8DBE70(v161)
      || (unsigned int)sub_8DBE70(v167) )
    {
      goto LABEL_338;
    }
    v128 = *(_BYTE *)(v161 + 140);
    if ( v128 == 12 )
    {
      v129 = v161;
      do
      {
        v129 = *(_QWORD *)(v129 + 160);
        v128 = *(_BYTE *)(v129 + 140);
      }
      while ( v128 == 12 );
    }
    if ( !v128 )
      goto LABEL_338;
    v130 = v167;
    for ( m = *(_BYTE *)(v167 + 140); m == 12; m = *(_BYTE *)(v130 + 140) )
      v130 = *(_QWORD *)(v130 + 160);
    if ( !m )
      goto LABEL_338;
    v132 = sub_8D3230(v161, v167);
    v38 = v161;
    if ( v132 )
    {
      v154 = v161;
      v162 = sub_8D3110(v161);
      v133 = sub_8D3110(v167);
      v38 = v154;
      if ( v162 == v133 )
      {
LABEL_369:
        v164 = sub_8D46C0(v38);
        v168 = sub_8D46C0(v167);
        v135 = sub_8D3A70(v164);
        v38 = v168;
        if ( v135 && (v151 = sub_8D3A70(v168), v38 = v168, v151) )
        {
          v169 = 1;
        }
        else
        {
          if ( !dword_4F077BC )
            goto LABEL_399;
          v169 = 0;
        }
        if ( (*(_BYTE *)(v164 + 140) & 0xFB) == 8 )
        {
          v153 = v38;
          v149 = sub_8D4C10(v164, dword_4F077C4 != 2);
          v38 = v153;
          v36 = (__int64)&dword_4F077C4;
          v37 = v149;
          v146 = 0;
          if ( (*(_BYTE *)(v153 + 140) & 0xFB) == 8 )
          {
LABEL_409:
            v152 = v37;
            v157 = v38;
            v146 = sub_8D4C10(v38, dword_4F077C4 != 2);
            v37 = v152;
            v38 = v157;
          }
          if ( ((unsigned int)v37 & ~v146) != 0 )
            goto LABEL_396;
          for ( n = v164; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
            ;
          v164 = n;
        }
        else if ( (*(_BYTE *)(v38 + 140) & 0xFB) == 8 )
        {
          LODWORD(v37) = 0;
          goto LABEL_409;
        }
        while ( *(_BYTE *)(v38 + 140) == 12 )
          v38 = *(_QWORD *)(v38 + 160);
        v155 = v38;
        if ( (unsigned int)sub_8D97D0(v164, v38, 512, v37, v38) || (unsigned int)sub_8D2600(v155) )
        {
          if ( v169 )
          {
LABEL_338:
            v38 = 0;
          }
          else
          {
            v136 = v159;
            do
            {
              v137 = *(_BYTE *)(v136 + 140);
              v136 = *(_QWORD *)(v136 + 160);
            }
            while ( v137 == 12 );
            sub_686870(855, v175, v117, v136);
            v38 = 0;
          }
        }
        else
        {
          v38 = v155;
          if ( !v169 )
            goto LABEL_396;
          if ( dword_4F077C4 == 2 )
          {
            v150 = sub_8D23B0(v164);
            v38 = v155;
            if ( v150 )
            {
              sub_8AE000(v164);
              v38 = v155;
            }
          }
          v144 = sub_8D5CE0(v164, v38);
          if ( !v144 )
            goto LABEL_396;
          if ( (*(_BYTE *)(v144 + 96) & 4) != 0 )
            goto LABEL_396;
          v170 = v144;
          v145 = sub_87DF20(v144);
          v38 = v170;
          if ( !v145 )
            goto LABEL_396;
        }
        sub_5F81D0(a3, (__int64)a4, v117, (__int64)v88, v38);
        *((_BYTE *)a4 + 129) |= 1u;
        v204 = 1;
        if ( (v89[193] & 0x10) == 0 )
          sub_5E6A70(v173, v222, v126, 0, (__int64)v88);
        goto LABEL_189;
      }
    }
    v163 = v38;
    v134 = sub_8D2F30(v38, v167);
    v38 = v163;
    if ( v134 )
    {
      if ( (*(_BYTE *)(v163 + 140) & 0xFB) == 8 )
      {
        v148 = sub_8D4C10(v163, dword_4F077C4 != 2);
        v38 = v163;
        v37 = v148 & 0xFFFFFF8F;
        v141 = 0;
        if ( (*(_BYTE *)(v167 + 140) & 0xFB) == 8 )
          goto LABEL_394;
      }
      else
      {
        if ( (*(_BYTE *)(v167 + 140) & 0xFB) != 8 )
          goto LABEL_369;
        LODWORD(v37) = 0;
LABEL_394:
        v156 = v37;
        v165 = v38;
        v140 = sub_8D4C10(v167, dword_4F077C4 != 2);
        v37 = v156;
        v38 = v165;
        v141 = v140 & 0xFFFFFF8F;
      }
      if ( v141 == (_DWORD)v37 )
        goto LABEL_369;
    }
LABEL_396:
    if ( !dword_4F077BC
      || dword_4F04C44 == -1
      && (v36 = (__int64)qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0) )
    {
LABEL_399:
      v142 = v159;
      do
      {
        v143 = *(_BYTE *)(v142 + 140);
        v142 = *(_QWORD *)(v142 + 160);
      }
      while ( v143 == 12 );
      sub_686890(317, v175, v117, v142, v38);
    }
LABEL_189:
    v88 = (_QWORD *)*v88;
    if ( v88 )
      continue;
    break;
  }
LABEL_206:
  v85 = v89;
  v18 = v188;
  v26 = v182;
  v86 = v179;
  v5 = v177;
LABEL_207:
  if ( v204 )
  {
    v85[192] |= 4u;
    if ( (v85[193] & 0x10) == 0 && v85[174] != 2 )
    {
      for ( ii = v86; *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
        ;
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)ii + 96LL) + 181LL) & 0x10) != 0 )
      {
        v195 = v86;
        v205 = v85;
        v96 = sub_6440B0(11, a4);
        v85 = v205;
        v86 = v195;
        if ( !v96 )
        {
          sub_6851C0(1892, v171);
          v86 = v195;
          v85 = v205;
        }
      }
    }
  }
LABEL_215:
  if ( (*(_BYTE *)(v86 + 177) & 0x20) == 0 && (v5[4].m128i_i8[1] & 4) != 0 && (*((_BYTE *)a4 + 129) & 1) == 0 )
  {
    v201 = v86;
    v208 = v85;
    sub_684AA0(8, 1455, v175);
    v86 = v201;
    v85 = v208;
  }
  v97 = v85[192];
  if ( (v97 & 2) != 0 )
  {
    *(_WORD *)(v86 + 176) |= 0x140u;
    *(_BYTE *)(a3 + 8) |= 4u;
LABEL_219:
    if ( (v85[192] & 2) != 0 )
    {
      while ( *(_BYTE *)(v86 + 140) == 12 )
        v86 = *(_QWORD *)(v86 + 160);
      v36 = *(_QWORD *)(*(_QWORD *)v86 + 96LL);
      *(_DWORD *)(v36 + 176) = *(_DWORD *)(v36 + 176) & 0xFFF81FFF | 0x78000;
      *(_BYTE *)(a3 + 9) |= 0x10u;
      v98 = v85[193];
      if ( (v98 & 6) != 0 || (a4[1] & 0x180000) != 0 )
      {
        v36 = (__int64)&dword_4D04888;
        if ( !dword_4D04888 )
        {
          v99 = 2924;
          if ( (v98 & 4) == 0 )
            v99 = (a4[1] & 0x100000) == 0 ? 2390 : 2924;
          v206 = v85;
          sub_6851C0(v99, v175);
          v85 = v206;
          v206[193] &= 0xF9u;
          a4[1] &= 0xFFFFFFFFFFE7FFFFLL;
        }
      }
      if ( (v85[207] & 0x10) != 0 )
      {
        v209 = v85;
        sub_6851C0(2548, v175);
        v116 = *((_QWORD *)v209 + 19);
        *(_QWORD *)(v116 + 160) = sub_72C930();
        v209[207] &= ~0x10u;
        *((_BYTE *)a4 + 125) &= ~8u;
      }
    }
  }
  else if ( (v97 & 0x10) != 0 && (*((_BYTE *)a4 + 129) & 1) == 0 )
  {
    v200 = v86;
    v207 = v85;
    sub_6851C0(2614, v175);
    v85 = v207;
    v86 = v200;
    v207[192] &= ~0x10u;
    goto LABEL_219;
  }
  if ( (*(_BYTE *)(v26 + 192) & 2) != 0 )
  {
    v100 = *(_QWORD *)(v26 + 216);
    if ( v100 )
LABEL_232:
      sub_6851C0(3130, v100 + 8);
  }
LABEL_73:
  sub_644920(a4, (v5[4].m128i_i8[0] & 4) != 0, v36, v37, v38, v39);
  v46 = a5;
  if ( a5 )
  {
LABEL_78:
    v49 = *(unsigned __int8 *)(v26 + 174);
    if ( (_BYTE)v49 )
      goto LABEL_160;
    goto LABEL_79;
  }
  v47 = *(_QWORD *)(a3 + 32);
  if ( v47 )
  {
    v48 = *(_QWORD *)(v47 + 88);
    if ( *(_BYTE *)(v48 + 140) != 11 || !*(_BYTE *)(*(_QWORD *)(v48 + 168) + 113LL) )
    {
      sub_8960C0(v18);
      if ( *(char *)(v26 + 90) >= 0 )
        goto LABEL_159;
      goto LABEL_78;
    }
  }
  if ( *(char *)(v26 + 90) < 0 )
    goto LABEL_78;
LABEL_159:
  sub_8D9350(v214, v218);
  v49 = *(unsigned __int8 *)(v26 + 174);
  if ( !(_BYTE)v49 )
    goto LABEL_79;
LABEL_160:
  v74 = *(_QWORD *)a3;
  for ( jj = *(_QWORD *)a3; *(_BYTE *)(jj + 140) == 12; jj = *(_QWORD *)(jj + 160) )
    ;
  v76 = *(_BYTE **)(v18 + 88);
  v77 = *(_QWORD **)(*(_QWORD *)jj + 96LL);
  v78 = v76[174];
  if ( v78 == 1 )
  {
    if ( (a4[70] & 4) != 0 )
    {
      v77[2] = v18;
      v76[194] |= 2u;
      LOBYTE(v49) = *(_BYTE *)(v26 + 174);
      goto LABEL_177;
    }
    v108 = v77[1];
    if ( v108 )
    {
      v49 = v221;
      if ( *(_BYTE *)(v108 + 80) != 17 && v221 )
        v77[1] = v221;
    }
    else
    {
      v77[1] = v18;
    }
    v213 = v77;
    if ( (unsigned int)sub_72F310(v76, 1, v49, v46, v77, v45) )
    {
      if ( (v76[194] & 2) != 0 )
        v213[2] = v18;
      else
        *((_BYTE *)v213 + 176) |= 1u;
      v109 = v76[193];
      if ( (v109 & 0x10) != 0 )
        goto LABEL_285;
      v110 = *((_BYTE *)v213 + 176);
      *((_BYTE *)v213 + 176) = v110 | 2;
      if ( (v76[206] & 0x18) == 0 )
      {
        *((_BYTE *)v213 + 176) = v110 | 6;
        *(_BYTE *)(a3 + 8) |= 4u;
      }
    }
    v109 = v76[193];
LABEL_285:
    sub_5F03A0((__int64)v76, v74, (v109 & 0x10) != 0);
    LOBYTE(v49) = *(_BYTE *)(v26 + 174);
    goto LABEL_177;
  }
  if ( v78 == 2 )
  {
    v79 = v77[3];
    if ( !v79 )
      goto LABEL_301;
    if ( (*(_BYTE *)(v79 + 104) & 1) != 0 )
    {
      v215 = v77;
      v81 = sub_8796F0(v77[3]);
      v77 = v215;
    }
    else
    {
      v80 = *(_QWORD *)(v79 + 88);
      if ( *(_BYTE *)(v79 + 80) == 20 )
        v80 = *(_QWORD *)(v80 + 176);
      v81 = (*(_BYTE *)(v80 + 208) & 4) != 0;
    }
    if ( v81 )
    {
LABEL_301:
      v77[3] = v18;
      LOBYTE(v49) = *(_BYTE *)(v26 + 174);
    }
    else
    {
      v82 = *(_QWORD *)(v18 + 88);
      if ( !*(_QWORD *)(v82 + 248) )
        goto LABEL_176;
      if ( (*(_BYTE *)(v18 + 104) & 1) != 0 )
      {
        v217 = v77;
        v83 = sub_8796F0(v18);
        v77 = v217;
      }
      else
      {
        if ( *(_BYTE *)(v18 + 80) == 20 )
          v82 = *(_QWORD *)(v82 + 176);
        v83 = (*(_BYTE *)(v82 + 208) & 4) != 0;
      }
      if ( !v83 )
      {
        v216 = v77;
        v138 = sub_6F3270(**(_QWORD **)(*(_QWORD *)(v79 + 88) + 248LL), **(_QWORD **)(*(_QWORD *)(v18 + 88) + 248LL), 0);
        if ( v138 == -1 )
        {
          v216[3] = v18;
          *(_BYTE *)(*(_QWORD *)(v79 + 88) + 208LL) |= 4u;
          *(_BYTE *)(*(_QWORD *)(v79 + 96) + 80LL) |= 2u;
          LOBYTE(v49) = *(_BYTE *)(v26 + 174);
        }
        else
        {
          if ( v138 )
          {
            *((_BYTE *)a4 + 133) |= 4u;
            goto LABEL_176;
          }
          *(_BYTE *)(*(_QWORD *)(v79 + 88) + 208LL) |= 4u;
          *(_BYTE *)(*(_QWORD *)(v79 + 96) + 80LL) |= 2u;
          *(_BYTE *)(v79 + 82) |= 4u;
          *((_BYTE *)a4 + 133) |= 4u;
          *(_BYTE *)(v18 + 82) |= 4u;
          LOBYTE(v49) = *(_BYTE *)(v26 + 174);
        }
      }
      else
      {
LABEL_176:
        LOBYTE(v49) = *(_BYTE *)(v26 + 174);
      }
    }
  }
LABEL_177:
  if ( (_BYTE)v49 == 1 )
  {
    if ( (unsigned int)sub_72F570(v26) )
    {
      sub_73EA10(v26 + 152, v222);
      v112 = v222[0];
      *(_BYTE *)(**(_QWORD **)(v222[0] + 168) + 34LL) |= 0x20u;
      *(_BYTE *)(**(_QWORD **)(v112 + 168) + 34LL) |= 0x40u;
    }
    else if ( (unsigned int)sub_72F3C0(*(_QWORD *)(v26 + 152), *(_QWORD *)(*(_QWORD *)(v26 + 40) + 32LL), v220, 0, 1) )
    {
      sub_73EA10(v26 + 152, v222);
      *(_BYTE *)(**(_QWORD **)(v222[0] + 168) + 34LL) |= 0x40u;
    }
  }
  else if ( (unsigned int)sub_72F850(v26) )
  {
    sub_73EA10(v26 + 152, v222);
    *(_BYTE *)(**(_QWORD **)(v222[0] + 168) + 34LL) |= 0x20u;
  }
LABEL_79:
  if ( dword_4F077BC )
  {
    v50 = *(_BYTE *)(v26 + 200);
    if ( (v50 & 7) == 0 )
      *(_BYTE *)(v26 + 200) = *(_BYTE *)(*(_QWORD *)(v219 + 168) + 109LL) & 7 | v50 & 0xF8;
    v51 = (_QWORD *)a4[30];
    if ( v51 )
    {
      v52 = *(_QWORD *)(v26 + 256);
      if ( !v52 )
      {
        v52 = sub_726210(v26);
        v51 = (_QWORD *)a4[30];
      }
      *(_QWORD *)(v52 + 40) = v51;
    }
  }
  sub_648B00(v26, a4 + 28, v218, 0, (v5[4].m128i_i8[0] & 4) != 0, (v5[4].m128i_i8[0] & 2) != 0);
  if ( !v189 )
    *(_BYTE *)(v219 + 88) |= 4u;
LABEL_88:
  if ( (*(_BYTE *)(v26 + 198) & 0x10) != 0 && (*(_BYTE *)(v26 + 193) & 0x10) == 0 )
    sub_8E3700(*(_QWORD *)(v26 + 152));
  return sub_826060(v26, v218);
}
