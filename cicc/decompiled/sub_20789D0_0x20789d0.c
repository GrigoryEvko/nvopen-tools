// Function: sub_20789D0
// Address: 0x20789d0
//
void __fastcall sub_20789D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int8 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9)
{
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  int v17; // ebx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rbx
  int v24; // ebx
  __int64 v25; // rax
  __int64 v26; // rdx
  char v27; // dl
  _BYTE *v28; // r15
  unsigned __int64 v29; // r13
  __int64 (*v30)(); // rax
  __int64 **v31; // rbx
  char v32; // al
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r12
  int v36; // r12d
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 *v40; // r12
  char v41; // r15
  __int64 v42; // rdx
  __int64 (*v43)(); // rax
  __m128i *v44; // rsi
  __int8 v45; // cl
  __int64 (*v46)(); // rax
  __int64 *v47; // rax
  __int32 v48; // edx
  __m128i *v49; // rax
  __int64 v50; // rsi
  __int64 *v51; // rax
  __int64 v52; // r13
  __int64 v53; // rdx
  unsigned __int64 v54; // rbx
  _QWORD *v55; // r12
  char v56; // al
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rdx
  char v60; // al
  __int64 v61; // rdx
  char v62; // al
  __int64 v63; // rdx
  int v64; // eax
  const __m128i *v65; // rdi
  const __m128i *v66; // rsi
  const __m128i *v67; // rax
  __m128i *v68; // rax
  const __m128i *v69; // rax
  unsigned __int64 v70; // rbx
  _QWORD *v71; // rdi
  __int64 v72; // rax
  char v73; // al
  unsigned __int64 v74; // rbx
  __int64 *v75; // rax
  __int64 v76; // rdx
  __int64 *v77; // rax
  __int64 (*v78)(); // rax
  char v79; // al
  __int64 v80; // rdx
  char v81; // cl
  __int64 v82; // rax
  char v83; // al
  __int8 v84; // al
  char v85; // al
  __int64 v86; // rdx
  char v87; // al
  __int64 v88; // rdx
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // r12
  int v92; // r12d
  __int64 v93; // rax
  __int64 v94; // rdx
  unsigned int v95; // edx
  char v96; // al
  unsigned int v97; // r15d
  int v98; // eax
  __int64 v99; // r8
  __int64 v100; // r9
  _QWORD *v101; // rax
  __int64 v102; // rdx
  char v103; // al
  __int8 v104; // cl
  __int64 v105; // rbx
  char v106; // al
  unsigned __int64 v107; // rax
  __int64 v108; // r13
  __int64 v109; // rbx
  unsigned __int64 v110; // rax
  __int64 *v111; // r15
  int v112; // r12d
  __int64 v113; // r8
  __int128 v114; // rax
  __int64 *v115; // rbx
  __int64 v116; // rdx
  __int64 v117; // r15
  __int64 v118; // r12
  unsigned int v119; // [rsp+0h] [rbp-1080h]
  __int64 v122; // [rsp+18h] [rbp-1068h]
  __int64 v124; // [rsp+30h] [rbp-1050h]
  char v125; // [rsp+30h] [rbp-1050h]
  __int64 v126; // [rsp+40h] [rbp-1040h]
  _QWORD *v127; // [rsp+40h] [rbp-1040h]
  __int64 v128; // [rsp+48h] [rbp-1038h]
  unsigned __int64 v129; // [rsp+48h] [rbp-1038h]
  __m128i v130; // [rsp+50h] [rbp-1030h] BYREF
  _BYTE *v131; // [rsp+60h] [rbp-1020h]
  __int64 **v132; // [rsp+68h] [rbp-1018h]
  __int64 v133; // [rsp+70h] [rbp-1010h]
  __int64 v134; // [rsp+78h] [rbp-1008h]
  __int64 v135; // [rsp+80h] [rbp-1000h]
  __int64 v136; // [rsp+88h] [rbp-FF8h]
  __int64 v137; // [rsp+90h] [rbp-FF0h]
  __int64 v138; // [rsp+98h] [rbp-FE8h]
  __int64 *v139; // [rsp+A0h] [rbp-FE0h]
  __int64 v140; // [rsp+A8h] [rbp-FD8h]
  __m128i v141; // [rsp+B0h] [rbp-FD0h]
  __int64 *v142; // [rsp+C0h] [rbp-FC0h]
  __int64 v143; // [rsp+C8h] [rbp-FB8h]
  __int64 *v144; // [rsp+D0h] [rbp-FB0h]
  __int64 v145; // [rsp+D8h] [rbp-FA8h]
  _QWORD *v146; // [rsp+E0h] [rbp-FA0h]
  __int64 v147; // [rsp+E8h] [rbp-F98h]
  __int64 *v148; // [rsp+F0h] [rbp-F90h]
  __int64 v149; // [rsp+F8h] [rbp-F88h]
  __int64 v150; // [rsp+108h] [rbp-F78h] BYREF
  unsigned __int64 v151; // [rsp+118h] [rbp-F68h] BYREF
  const __m128i *v152; // [rsp+120h] [rbp-F60h] BYREF
  __m128i *v153; // [rsp+128h] [rbp-F58h]
  const __m128i *v154; // [rsp+130h] [rbp-F50h]
  __m128i v155; // [rsp+140h] [rbp-F40h] BYREF
  __m128i v156; // [rsp+150h] [rbp-F30h] BYREF
  __m128i v157; // [rsp+160h] [rbp-F20h] BYREF
  __m128i v158; // [rsp+170h] [rbp-F10h] BYREF
  __int64 v159; // [rsp+180h] [rbp-F00h]
  __int64 v160; // [rsp+188h] [rbp-EF8h]
  __int64 v161; // [rsp+190h] [rbp-EF0h]
  const __m128i *v162; // [rsp+198h] [rbp-EE8h]
  __m128i *v163; // [rsp+1A0h] [rbp-EE0h]
  const __m128i *v164; // [rsp+1A8h] [rbp-ED8h]
  __int64 *v165; // [rsp+1B0h] [rbp-ED0h]
  __int64 v166; // [rsp+1B8h] [rbp-EC8h] BYREF
  __int32 v167; // [rsp+1C0h] [rbp-EC0h]
  __int64 v168; // [rsp+1C8h] [rbp-EB8h]
  _BYTE *v169; // [rsp+1D0h] [rbp-EB0h]
  __int64 v170; // [rsp+1D8h] [rbp-EA8h]
  _BYTE v171[1536]; // [rsp+1E0h] [rbp-EA0h] BYREF
  __int64 **v172; // [rsp+7E0h] [rbp-8A0h]
  __int64 v173; // [rsp+7E8h] [rbp-898h]
  _BYTE v174[512]; // [rsp+7F0h] [rbp-890h] BYREF
  _BYTE *v175; // [rsp+9F0h] [rbp-690h]
  __int64 v176; // [rsp+9F8h] [rbp-688h]
  _BYTE v177[1536]; // [rsp+A00h] [rbp-680h] BYREF
  _BYTE *v178; // [rsp+1000h] [rbp-80h]
  __int64 v179; // [rsp+1008h] [rbp-78h]
  _BYTE v180[112]; // [rsp+1010h] [rbp-70h] BYREF

  v10 = *(_QWORD *)(a1 + 552);
  v150 = a2;
  v11 = *(_QWORD *)(v10 + 32);
  v130.m128i_i8[0] = a5;
  v122 = sub_1E0A0C0(v11);
  v12 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v128 = *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 64);
  v13 = *(_QWORD *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v152 = 0;
  v124 = v13;
  v153 = 0;
  v154 = 0;
  if ( (a2 & 4) != 0 )
  {
    if ( *(char *)(v12 + 23) >= 0 )
    {
      v20 = -24;
      goto LABEL_15;
    }
    v14 = sub_1648A40(v150 & 0xFFFFFFFFFFFFFFF8LL);
    v16 = v14 + v15;
    if ( *(char *)(v12 + 23) >= 0 )
    {
      if ( (unsigned int)(v16 >> 4) )
        goto LABEL_164;
    }
    else if ( (unsigned int)((v16 - sub_1648A40(v12)) >> 4) )
    {
      if ( *(char *)(v12 + 23) < 0 )
      {
        v17 = *(_DWORD *)(sub_1648A40(v12) + 8);
        if ( *(char *)(v12 + 23) >= 0 )
          BUG();
        v18 = sub_1648A40(v12);
        v12 = v150 & 0xFFFFFFFFFFFFFFF8LL;
        v20 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v18 + v19 - 4) - v17);
        goto LABEL_15;
      }
LABEL_164:
      BUG();
    }
    v20 = -24;
    v12 = v150 & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_15;
  }
  if ( *(char *)(v12 + 23) >= 0 )
  {
    v20 = -72;
    goto LABEL_15;
  }
  v21 = sub_1648A40(v150 & 0xFFFFFFFFFFFFFFF8LL);
  v23 = v21 + v22;
  if ( *(char *)(v12 + 23) >= 0 )
  {
    if ( (unsigned int)(v23 >> 4) )
LABEL_162:
      BUG();
LABEL_94:
    v20 = -72;
    v12 = v150 & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_15;
  }
  if ( !(unsigned int)((v23 - sub_1648A40(v12)) >> 4) )
    goto LABEL_94;
  if ( *(char *)(v12 + 23) >= 0 )
    goto LABEL_162;
  v24 = *(_DWORD *)(sub_1648A40(v12) + 8);
  if ( *(char *)(v12 + 23) >= 0 )
    BUG();
  v25 = sub_1648A40(v12);
  v12 = v150 & 0xFFFFFFFFFFFFFFF8LL;
  v20 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v25 + v26 - 4) - v24);
LABEL_15:
  sub_1FD3FA0(
    &v152,
    -1431655765
  * (unsigned int)((__int64)(v20 + (a2 & 0xFFFFFFFFFFFFFFF8LL) + 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF) - v12) >> 3));
  v27 = v150;
  v28 = *(_BYTE **)(*(_QWORD *)(a1 + 552) + 16LL);
  v29 = v150 & 0xFFFFFFFFFFFFFFF8LL;
  v30 = *(__int64 (**)())(*(_QWORD *)v28 + 1160LL);
  if ( v30 != sub_1D45FE0 )
  {
    v105 = *(_QWORD *)(*(_QWORD *)(v29 + 40) + 56LL);
    if ( ((unsigned __int8 (__fastcall *)(_QWORD))v30)(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL)) )
    {
      v157.m128i_i64[0] = *(_QWORD *)(v105 + 112);
      v106 = sub_1560490(&v157, 54, 0);
      v27 = v150;
      v29 = v150 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v106 )
        v130.m128i_i8[0] = 0;
    }
    else
    {
      v27 = v150;
      v29 = v150 & 0xFFFFFFFFFFFFFFF8LL;
    }
  }
  v31 = (__int64 **)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
  v32 = *(_BYTE *)(v29 + 23);
  if ( (v27 & 4) == 0 )
  {
    if ( v32 < 0 )
    {
      v89 = sub_1648A40(v29);
      v91 = v89 + v90;
      if ( *(char *)(v29 + 23) >= 0 )
      {
        if ( (unsigned int)(v91 >> 4) )
          goto LABEL_167;
      }
      else if ( (unsigned int)((v91 - sub_1648A40(v29)) >> 4) )
      {
        if ( *(char *)(v29 + 23) < 0 )
        {
          v92 = *(_DWORD *)(sub_1648A40(v29) + 8);
          if ( *(char *)(v29 + 23) >= 0 )
            BUG();
          v93 = sub_1648A40(v29);
          v39 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v93 + v94 - 4) - v92);
          goto LABEL_23;
        }
LABEL_167:
        BUG();
      }
    }
    v39 = -72;
    goto LABEL_23;
  }
  if ( v32 < 0 )
  {
    v33 = sub_1648A40(v29);
    v35 = v33 + v34;
    if ( *(char *)(v29 + 23) >= 0 )
    {
      if ( (unsigned int)(v35 >> 4) )
        goto LABEL_161;
    }
    else if ( (unsigned int)((v35 - sub_1648A40(v29)) >> 4) )
    {
      if ( *(char *)(v29 + 23) < 0 )
      {
        v36 = *(_DWORD *)(sub_1648A40(v29) + 8);
        if ( *(char *)(v29 + 23) >= 0 )
          BUG();
        v37 = sub_1648A40(v29);
        v39 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v37 + v38 - 4) - v36);
        goto LABEL_23;
      }
LABEL_161:
      BUG();
    }
  }
  v39 = -24;
LABEL_23:
  v126 = 0;
  v132 = (__int64 **)(v29 + v39);
  if ( (__int64 **)(v29 + v39) != v31 )
  {
    v131 = v28;
    do
    {
      v157 = 0u;
      v158 = 0u;
      LODWORD(v159) = 0;
      v40 = *v31;
      v41 = sub_1642FB0(**v31);
      if ( !v41 )
      {
        v148 = sub_20685E0(a1, v40, a7, a8, a9);
        v149 = v42;
        v157.m128i_i64[1] = (__int64)v148;
        v158.m128i_i32[0] = v42;
        v158.m128i_i64[1] = *v40;
        sub_20A1C00(
          &v157,
          &v150,
          0xAAAAAAAAAAAAAAABLL
        * ((__int64)((__int64)&v31[3 * (*(_DWORD *)((v150 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)]
                   - (v150 & 0xFFFFFFFFFFFFFFF8LL)) >> 3));
        if ( (v159 & 0x200) != 0 )
        {
          v43 = *(__int64 (**)())(*(_QWORD *)v131 + 1160LL);
          if ( v43 != sub_1D45FE0 )
          {
            if ( ((unsigned __int8 (__fastcall *)(_BYTE *))v43)(v131) )
            {
              v127 = *(_QWORD **)(a1 + 552);
              v95 = 8 * sub_15A9520(v122, 0);
              if ( v95 == 32 )
              {
                v96 = 5;
              }
              else if ( v95 > 0x20 )
              {
                v96 = 6;
                if ( v95 != 64 )
                {
                  v96 = 7;
                  if ( v95 != 128 )
                    v96 = v41;
                }
              }
              else
              {
                v96 = 3;
                if ( v95 != 8 )
                  v96 = 4 * (v95 == 16);
              }
              v97 = v119;
              LOBYTE(v97) = v96;
              v98 = sub_1FE6270(
                      *(_QWORD *)(a1 + 712),
                      v150 & 0xFFFFFFFFFFFFFFF8LL,
                      *(_QWORD *)(*(_QWORD *)(a1 + 712) + 784LL),
                      (__int64)v40);
              v101 = sub_1D2A660(v127, v98, v97, 0, v99, v100);
              v126 = (__int64)v40;
              v146 = v101;
              v147 = v102;
              v157.m128i_i64[1] = (__int64)v101;
              v158.m128i_i32[0] = v102;
            }
          }
        }
        v44 = v153;
        if ( v153 == v154 )
        {
          sub_1D27190(&v152, v153, &v157);
        }
        else
        {
          if ( v153 )
          {
            a8 = _mm_loadu_si128(&v157);
            *v153 = a8;
            a9 = _mm_loadu_si128(&v158);
            v44[1] = a9;
            v44[2].m128i_i64[0] = v159;
            v44 = v153;
          }
          v153 = (__m128i *)((char *)v44 + 40);
        }
        if ( (v159 & 8) != 0 )
        {
          v45 = v130.m128i_i8[0];
          if ( *((_BYTE *)v40 + 16) >= 0x18u )
            v45 = 0;
          v130.m128i_i8[0] = v45;
        }
      }
      v31 += 3;
    }
    while ( v31 != v132 );
    v28 = v131;
  }
  if ( v130.m128i_i8[0] )
    v130.m128i_i8[0] = sub_20C8B80(v150, **(_QWORD **)(a1 + 552));
  v46 = *(__int64 (**)())(*(_QWORD *)v28 + 1160LL);
  if ( v46 != sub_1D45FE0 )
  {
    v103 = ((__int64 (__fastcall *)(_BYTE *))v46)(v28);
    if ( v126 )
    {
      v104 = v130.m128i_i8[0];
      if ( v103 )
        v104 = 0;
      v130.m128i_i8[0] = v104;
    }
  }
  v47 = *(__int64 **)(a1 + 552);
  v48 = *(_DWORD *)(a1 + 536);
  v157 = 0u;
  v158.m128i_i64[1] = 0xFFFFFFFF00000020LL;
  v165 = v47;
  v169 = v171;
  v170 = 0x2000000000LL;
  v173 = 0x2000000000LL;
  v176 = 0x2000000000LL;
  v178 = v180;
  v179 = 0x400000000LL;
  v49 = *(__m128i **)a1;
  v132 = (__int64 **)v174;
  v172 = (__int64 **)v174;
  v158.m128i_i64[0] = 0;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  v164 = 0;
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v131 = v177;
  v175 = v177;
  v155.m128i_i64[0] = 0;
  v155.m128i_i32[2] = v48;
  if ( v49 )
  {
    if ( &v155 != &v49[3] )
    {
      v50 = v49[3].m128i_i64[0];
      v155.m128i_i64[0] = v50;
      if ( v50 )
      {
        sub_1623A60((__int64)&v155, v50, 2);
        if ( v166 )
          sub_161E7C0((__int64)&v166, v166);
        v166 = v155.m128i_i64[0];
        if ( v155.m128i_i64[0] )
          sub_1623A60((__int64)&v166, v155.m128i_i64[0], 2);
      }
    }
  }
  v167 = v155.m128i_i32[2];
  v51 = sub_2051C20((__int64 *)a1, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
  v52 = v150;
  v145 = v53;
  v144 = v51;
  v157.m128i_i64[0] = (__int64)v51;
  v54 = v150 & 0xFFFFFFFFFFFFFFF8LL;
  v55 = (_QWORD *)((v150 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  v157.m128i_i32[2] = v53;
  v158.m128i_i64[0] = v124;
  v125 = (v150 >> 2) & 1;
  if ( ((v150 >> 2) & 1) != 0 )
  {
    v56 = sub_1560260(v55, 0, 12);
    if ( !v56 )
    {
      v57 = *(_QWORD *)(v54 - 24);
      if ( !*(_BYTE *)(v57 + 16) )
      {
        v151 = *(_QWORD *)(v57 + 112);
        v56 = sub_1560260(&v151, 0, 12);
      }
    }
    v158.m128i_i8[8] = v158.m128i_i8[8] & 0xF7 | (8 * (v56 & 1));
    if ( (unsigned __int8)sub_1560260(v55, -1, 29)
      || (v58 = *(_QWORD *)(v54 - 24), !*(_BYTE *)(v58 + 16))
      && (v151 = *(_QWORD *)(v58 + 112), (unsigned __int8)sub_1560260(&v151, -1, 29)) )
    {
      v158.m128i_i8[8] |= 0x10u;
      v158.m128i_i8[8] = (4 * (*(_DWORD *)(v128 + 8) >> 8 != 0)) | v158.m128i_i8[8] & 0xFB;
      v158.m128i_i8[8] = (32 * (*(_QWORD *)(v54 + 8) != 0)) | v158.m128i_i8[8] & 0xDF;
LABEL_58:
      v60 = sub_1560260(v55, 0, 40);
      if ( !v60 )
      {
        v61 = *(_QWORD *)(v54 - 24);
        if ( !*(_BYTE *)(v61 + 16) )
        {
          v151 = *(_QWORD *)(v61 + 112);
          v60 = sub_1560260(&v151, 0, 40);
        }
      }
      v158.m128i_i8[8] = v158.m128i_i8[8] & 0xFE | v60 & 1;
      v62 = sub_1560260(v55, 0, 58);
      if ( !v62 )
      {
        v63 = *(_QWORD *)(v54 - 24);
        if ( !*(_BYTE *)(v63 + 16) )
        {
          v151 = *(_QWORD *)(v63 + 112);
          v62 = sub_1560260(&v151, 0, 58);
        }
      }
      v135 = a3;
      v158.m128i_i8[8] = v158.m128i_i8[8] & 0xFD | (2 * (v62 & 1));
      v160 = a3;
      v136 = a4;
      LODWORD(v161) = a4;
      v64 = (*(unsigned __int16 *)(v54 + 18) >> 2) & 0x3FFFDFFF;
      goto LABEL_65;
    }
    goto LABEL_55;
  }
  v79 = sub_1560260(v55, 0, 12);
  if ( !v79 )
  {
    v80 = *(_QWORD *)(v54 - 72);
    if ( !*(_BYTE *)(v80 + 16) )
    {
      v151 = *(_QWORD *)(v80 + 112);
      v79 = sub_1560260(&v151, 0, 12);
    }
  }
  v158.m128i_i8[8] = v158.m128i_i8[8] & 0xF7 | (8 * (v79 & 1));
  v81 = sub_1560260(v55, -1, 29);
  if ( v81 )
    goto LABEL_109;
  v82 = *(_QWORD *)(v54 - 72);
  if ( !*(_BYTE *)(v82 + 16) )
  {
    v151 = *(_QWORD *)(v82 + 112);
    v83 = sub_1560260(&v151, -1, 29);
    v81 = 0;
    if ( v83 )
    {
      v84 = v158.m128i_i8[8] | 0x10;
      goto LABEL_110;
    }
  }
  if ( v54 )
  {
LABEL_109:
    v84 = (16 * (v81 & 1)) | v158.m128i_i8[8] & 0xEF;
LABEL_110:
    v158.m128i_i8[8] = v84;
    v158.m128i_i8[8] = (4 * (*(_DWORD *)(v128 + 8) >> 8 != 0)) | v84 & 0xFB;
    v158.m128i_i8[8] = (32 * (*(_QWORD *)(v54 + 8) != 0)) | v158.m128i_i8[8] & 0xDF;
    goto LABEL_111;
  }
LABEL_55:
  v59 = *(_QWORD *)(v54 + 32);
  if ( v59 == *(_QWORD *)(v54 + 40) + 40LL || !v59 )
    BUG();
  v158.m128i_i8[8] = (16 * (*(_BYTE *)(v59 - 8) == 31)) | v158.m128i_i8[8] & 0xEF;
  v158.m128i_i8[8] = (4 * (*(_DWORD *)(v128 + 8) >> 8 != 0)) | v158.m128i_i8[8] & 0xFB;
  v158.m128i_i8[8] = (32 * (*(_QWORD *)(v54 + 8) != 0)) | v158.m128i_i8[8] & 0xDF;
  if ( v125 )
    goto LABEL_58;
LABEL_111:
  v85 = sub_1560260(v55, 0, 40);
  if ( !v85 )
  {
    v86 = *(_QWORD *)(v54 - 72);
    if ( !*(_BYTE *)(v86 + 16) )
    {
      v151 = *(_QWORD *)(v86 + 112);
      v85 = sub_1560260(&v151, 0, 40);
    }
  }
  v158.m128i_i8[8] = v158.m128i_i8[8] & 0xFE | v85 & 1;
  v87 = sub_1560260(v55, 0, 58);
  if ( !v87 )
  {
    v88 = *(_QWORD *)(v54 - 72);
    if ( !*(_BYTE *)(v88 + 16) )
    {
      v151 = *(_QWORD *)(v88 + 112);
      v87 = sub_1560260(&v151, 0, 58);
    }
  }
  v133 = a3;
  v158.m128i_i8[8] = v158.m128i_i8[8] & 0xFD | (2 * (v87 & 1));
  v160 = a3;
  v134 = a4;
  LODWORD(v161) = a4;
  v64 = (*(unsigned __int16 *)(v54 + 18) >> 2) & 0x3FFFDFFF;
LABEL_65:
  LODWORD(v159) = v64;
  v65 = v162;
  v66 = v164;
  v158.m128i_i32[3] = *(_DWORD *)(v128 + 12) - 1;
  v67 = v152;
  v152 = 0;
  v162 = v67;
  v68 = v153;
  v153 = 0;
  v163 = v68;
  v69 = v154;
  v154 = 0;
  v164 = v69;
  if ( v65 )
    j_j___libc_free_0(v65, (char *)v66 - (char *)v65);
  v168 = v52;
  v158.m128i_i8[9] = v130.m128i_i8[0];
  v70 = v150 & 0xFFFFFFFFFFFFFFF8LL;
  v71 = (_QWORD *)((v150 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( (v150 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560260(v71, -1, 8) )
    {
      v72 = *(_QWORD *)(v70 - 24);
      if ( !*(_BYTE *)(v72 + 16) )
      {
LABEL_70:
        v151 = *(_QWORD *)(v72 + 112);
        v73 = sub_1560260(&v151, -1, 8);
        goto LABEL_71;
      }
      goto LABEL_101;
    }
LABEL_98:
    v73 = 1;
    goto LABEL_71;
  }
  if ( (unsigned __int8)sub_1560260(v71, -1, 8) )
    goto LABEL_98;
  v72 = *(_QWORD *)(v70 - 72);
  if ( !*(_BYTE *)(v72 + 16) )
    goto LABEL_70;
LABEL_101:
  v73 = 0;
LABEL_71:
  v158.m128i_i8[8] = v158.m128i_i8[8] & 0xBF | ((v73 & 1) << 6);
  if ( v155.m128i_i64[0] )
    sub_161E7C0((__int64)&v155, v155.m128i_i64[0]);
  sub_2061DC0(&v155, a1, &v157, a6, a7, a8, a9);
  if ( v155.m128i_i64[0] )
  {
    v74 = v150 & 0xFFFFFFFFFFFFFFF8LL;
    v75 = sub_2055040(
            a1,
            *(_QWORD *)(a1 + 552),
            v150 & 0xFFFFFFFFFFFFFFF8LL,
            v155.m128i_i64[0],
            v155.m128i_u64[1],
            *(double *)a7.m128i_i64,
            *(double *)a8.m128i_i64,
            a9);
    v151 = v74;
    v143 = v76;
    v155.m128i_i64[0] = (__int64)v75;
    v142 = v75;
    v155.m128i_i32[2] = v76;
    v130 = _mm_loadu_si128(&v155);
    v77 = sub_205F5C0(a1 + 8, (__int64 *)&v151);
    a7 = _mm_load_si128(&v130);
    v141 = a7;
    v77[1] = a7.m128i_i64[0];
    *((_DWORD *)v77 + 4) = v141.m128i_i32[2];
  }
  if ( v126 )
  {
    v78 = *(__int64 (**)())(*(_QWORD *)v28 + 1160LL);
    if ( v78 != sub_1D45FE0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(_BYTE *))v78)(v28) )
      {
        v107 = (unsigned __int64)&v178[16 * (unsigned int)v179 - 16];
        v108 = *(_QWORD *)v107;
        v109 = *(unsigned int *)(v107 + 8);
        v110 = sub_1FE5EB0(*(_QWORD *)(a1 + 712), v150 & 0xFFFFFFFFFFFFFFF8LL);
        v111 = v165;
        v112 = v110;
        v129 = HIDWORD(v110);
        v130 = _mm_loadu_si128(&v156);
        *(_QWORD *)&v114 = sub_1D2A660(
                             v165,
                             v110,
                             *(unsigned __int8 *)(*(_QWORD *)(v108 + 40) + 16 * v109),
                             *(_QWORD *)(*(_QWORD *)(v108 + 40) + 16 * v109 + 8),
                             v113,
                             0);
        v115 = sub_1D3A900(
                 v111,
                 0x2Eu,
                 (__int64)&v166,
                 1u,
                 0,
                 0,
                 (__m128)a7,
                 *(double *)a8.m128i_i64,
                 a9,
                 v130.m128i_u64[0],
                 (__int16 *)v130.m128i_i64[1],
                 v114,
                 v108,
                 v109);
        v117 = v116;
        if ( (_BYTE)v129 )
          sub_1FE5190(*(_QWORD *)(a1 + 712), *(_QWORD *)(*(_QWORD *)(a1 + 712) + 784LL), v126, v112);
        v118 = *(_QWORD *)(a1 + 552);
        if ( v115 )
        {
          nullsub_686();
          v140 = v117;
          v139 = v115;
          *(_QWORD *)(v118 + 176) = v115;
          *(_DWORD *)(v118 + 184) = v140;
          sub_1D23870();
        }
        else
        {
          v138 = v117;
          v137 = 0;
          *(_QWORD *)(v118 + 176) = 0;
          *(_DWORD *)(v118 + 184) = v138;
        }
      }
    }
  }
  if ( v178 != v180 )
    _libc_free((unsigned __int64)v178);
  if ( v175 != v131 )
    _libc_free((unsigned __int64)v175);
  if ( v172 != v132 )
    _libc_free((unsigned __int64)v172);
  if ( v169 != v171 )
    _libc_free((unsigned __int64)v169);
  if ( v166 )
    sub_161E7C0((__int64)&v166, v166);
  if ( v162 )
    j_j___libc_free_0(v162, (char *)v164 - (char *)v162);
  if ( v152 )
    j_j___libc_free_0(v152, (char *)v154 - (char *)v152);
}
