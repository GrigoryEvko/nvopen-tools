// Function: sub_638AC0
// Address: 0x638ac0
//
__int64 __fastcall sub_638AC0(__int64 a1, _QWORD *a2, unsigned int a3, unsigned int a4, _DWORD *a5, const __m128i *a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  bool v8; // sf
  char v9; // al
  bool v10; // r15
  __m128i *v11; // r14
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdi
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int16 v19; // ax
  int v20; // r15d
  __int64 v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rdx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 j; // r8
  __int64 v35; // r10
  char v36; // dl
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rax
  __int8 v41; // al
  bool v42; // al
  __int64 i; // rax
  __int64 v44; // rcx
  char v45; // dl
  bool v46; // al
  __int64 v47; // r15
  __int64 v48; // rax
  __int64 v49; // r15
  __int64 v50; // rax
  char v51; // dl
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  int v56; // eax
  __int64 v57; // r10
  __int8 v58; // cl
  __int64 *v59; // rax
  __int64 result; // rax
  char v61; // al
  __int64 k; // r15
  __int64 v63; // rdx
  char v64; // al
  __int64 v65; // rax
  __int64 v66; // r10
  __int64 *v67; // rax
  int v68; // eax
  __int64 *v69; // r10
  unsigned int v70; // r15d
  __int64 v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // rdi
  unsigned int v74; // ecx
  __int64 v75; // rdi
  __int64 v76; // r13
  char v77; // al
  bool v78; // zf
  __int64 v79; // rax
  int v80; // eax
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 *v85; // rax
  __int64 v86; // r10
  _BOOL4 v87; // r13d
  __int64 v88; // rdi
  char v89; // al
  int v90; // eax
  int v91; // ecx
  char v92; // al
  __int64 v93; // r10
  int v94; // eax
  __int64 v95; // rdi
  int v96; // eax
  __int64 v97; // r8
  int v98; // eax
  __int64 v99; // r10
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // r8
  __int64 v103; // rax
  int v104; // eax
  __int64 v105; // rax
  __int64 v106; // rdi
  int v107; // eax
  __int64 v108; // rax
  char v109; // dl
  int v110; // eax
  int v111; // eax
  __int64 v112; // rax
  __int64 v113; // rdi
  char m; // al
  __int64 v115; // rsi
  __int64 v116; // rax
  __int64 v117; // rax
  __int64 *v118; // r10
  int v119; // eax
  int v120; // eax
  __int64 v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // r10
  __int64 v124; // rax
  __int64 v125; // rax
  __int64 v126; // rax
  __int64 v127; // rax
  __int64 v128; // rax
  __int64 v129; // rdi
  int v130; // eax
  __int64 v131; // r8
  __int64 v132; // rdx
  __int64 v133; // rax
  __int64 v134; // r13
  __int64 v135; // rax
  __int64 v136; // rax
  __int64 v137; // rdx
  __int64 v138; // rcx
  __int64 v139; // r8
  __int64 v140; // rax
  __int64 v141; // [rsp-10h] [rbp-1A0h]
  __int64 v142; // [rsp-8h] [rbp-198h]
  int v143; // [rsp+Ch] [rbp-184h]
  char v144; // [rsp+10h] [rbp-180h]
  __int64 v145; // [rsp+10h] [rbp-180h]
  __int64 *v146; // [rsp+10h] [rbp-180h]
  __int64 v147; // [rsp+10h] [rbp-180h]
  int v148; // [rsp+18h] [rbp-178h]
  __int64 *v149; // [rsp+18h] [rbp-178h]
  __int64 *v150; // [rsp+18h] [rbp-178h]
  __int64 *v151; // [rsp+18h] [rbp-178h]
  __int64 v152; // [rsp+18h] [rbp-178h]
  __int64 *v153; // [rsp+18h] [rbp-178h]
  __int64 v154; // [rsp+18h] [rbp-178h]
  __int64 v155; // [rsp+20h] [rbp-170h]
  __int64 v156; // [rsp+20h] [rbp-170h]
  __int64 v157; // [rsp+20h] [rbp-170h]
  __int64 *v158; // [rsp+20h] [rbp-170h]
  __m128i *v159; // [rsp+20h] [rbp-170h]
  __int64 v160; // [rsp+20h] [rbp-170h]
  __int64 v161; // [rsp+20h] [rbp-170h]
  unsigned __int8 v162; // [rsp+2Ah] [rbp-166h]
  char v163; // [rsp+2Bh] [rbp-165h]
  __int64 v165; // [rsp+30h] [rbp-160h]
  __int64 v166; // [rsp+38h] [rbp-158h]
  __int64 v167; // [rsp+40h] [rbp-150h]
  unsigned int v168; // [rsp+48h] [rbp-148h]
  unsigned int v169; // [rsp+4Ch] [rbp-144h]
  int v170; // [rsp+60h] [rbp-130h]
  int v171; // [rsp+64h] [rbp-12Ch]
  unsigned __int16 v172; // [rsp+68h] [rbp-128h]
  __int64 v174; // [rsp+70h] [rbp-120h]
  int v175; // [rsp+70h] [rbp-120h]
  __int64 v176; // [rsp+70h] [rbp-120h]
  __int64 v180; // [rsp+88h] [rbp-108h]
  __int64 v181; // [rsp+88h] [rbp-108h]
  __int64 v182; // [rsp+88h] [rbp-108h]
  __int64 v183; // [rsp+88h] [rbp-108h]
  __int64 v184; // [rsp+88h] [rbp-108h]
  __int64 v185; // [rsp+88h] [rbp-108h]
  __int64 v186; // [rsp+88h] [rbp-108h]
  __int64 v187; // [rsp+88h] [rbp-108h]
  __int64 v188; // [rsp+88h] [rbp-108h]
  __int64 v189; // [rsp+88h] [rbp-108h]
  __int64 v190; // [rsp+88h] [rbp-108h]
  int v191; // [rsp+94h] [rbp-FCh] BYREF
  __int64 v192; // [rsp+98h] [rbp-F8h] BYREF
  __int64 v193; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v194; // [rsp+A8h] [rbp-E8h] BYREF
  __int64 *v195; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 *v196; // [rsp+B8h] [rbp-D8h] BYREF
  __int64 v197; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v198; // [rsp+C8h] [rbp-C8h]

  v6 = a1;
  v7 = *(_QWORD *)a1;
  v192 = 0;
  *(_BYTE *)(a1 + 127) |= 4u;
  v8 = *(char *)(a1 + 125) < 0;
  *(_QWORD *)(a1 + 152) = a1;
  if ( v8 )
  {
    a1 = 78;
    v11 = 0;
    v10 = 0;
    sub_6851C0(78, a2);
    *(_BYTE *)(v6 + 176) &= ~2u;
    v169 = 0;
    v171 = 1;
    goto LABEL_15;
  }
  v9 = *(_BYTE *)(v7 + 80);
  switch ( v9 )
  {
    case 7:
      v11 = *(__m128i **)(v7 + 88);
      v10 = v11[8].m128i_i8[8] <= 2u;
      v169 = v10;
      *(_BYTE *)(a1 + 176) = (2 * v10) | *(_BYTE *)(a1 + 176) & 0xFD;
      break;
    case 9:
      v11 = *(__m128i **)(v7 + 88);
      v10 = 1;
      *(_BYTE *)(a1 + 176) |= 2u;
      v169 = 1;
      break;
    case 21:
      v10 = 1;
      v169 = 1;
      v11 = *(__m128i **)(*(_QWORD *)(v7 + 88) + 192LL);
      *(_BYTE *)(a1 + 176) |= 2u;
      break;
    default:
      a1 = 145;
      sub_6854C0(145, a2, v7);
      v171 = 1;
      v10 = dword_4F04C58 == -1;
      v11 = 0;
      v169 = v10;
      *(_BYTE *)(v6 + 176) = (2 * v10) | *(_BYTE *)(v6 + 176) & 0xFD;
      goto LABEL_15;
  }
  v12 = v11[7].m128i_i64[1];
  if ( unk_4D047EC )
  {
    a1 = v11[7].m128i_i64[1];
    if ( (unsigned int)sub_8D4070(a1) )
      *(_BYTE *)(v6 + 178) |= 4u;
  }
  if ( (*(_BYTE *)(v6 + 178) & 4) != 0 && (!dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0x9FC3u) )
    goto LABEL_11;
  if ( *(_BYTE *)(v7 + 80) == 7 && a3 && dword_4F04C58 != -1 )
  {
    a1 = 145;
    sub_6854C0(145, a2, v7);
    v171 = 1;
    goto LABEL_70;
  }
  v41 = v11[11].m128i_i8[1];
  if ( !v41 || (*(_BYTE *)(v6 + 127) & 0x10) != 0 && (*(_BYTE *)(v6 + 130) & 4) != 0 )
  {
    a1 = v12;
    if ( !(unsigned int)sub_8D25A0(v12) )
    {
      if ( !(unsigned int)sub_8D3410(v12) || (a1 = sub_8D4050(v12), (unsigned int)sub_8D23B0(a1)) )
      {
        a1 = v12;
        if ( !(unsigned int)sub_8D32E0(v12) )
        {
          a1 = v12;
          if ( !(unsigned int)sub_8DBE70(v12) )
          {
            if ( (unsigned int)sub_8D23B0(v12) )
            {
              a1 = (unsigned int)sub_67F240(v12);
              sub_685A50(a1, a2, v12, 8);
              *a5 = 1;
              goto LABEL_12;
            }
LABEL_11:
            a1 = 145;
            sub_6854C0(145, a2, v7);
LABEL_12:
            v171 = 1;
            if ( HIDWORD(qword_4F077B4) )
            {
              if ( v169 )
              {
                v171 = v169;
                if ( (v11[10].m128i_i8[9] & 8) == 0 )
                {
                  if ( v11[9].m128i_i8[0] )
                  {
                    a1 = 1559;
                    sub_6851C0(1559, a2);
                  }
                }
              }
            }
            goto LABEL_15;
          }
        }
      }
    }
    v171 = 0;
  }
  else if ( v41 != 1 || (v11[10].m128i_i8[12] & 8) == 0 || (v171 = 1, *(_BYTE *)(v11[11].m128i_i64[1] + 173)) )
  {
    a1 = 148;
    sub_6854C0(148, a2, v7);
    v171 = 1;
  }
LABEL_70:
  if ( HIDWORD(qword_4F077B4) && v169 && (v11[10].m128i_i8[9] & 8) == 0 && v11[9].m128i_i8[0] )
  {
    a1 = 1559;
    sub_6851C0(1559, a2);
  }
  if ( v12 )
  {
    v14 = *(_QWORD *)(v7 + 64);
    if ( (*(_BYTE *)(v7 + 81) & 0x10) != 0 )
      goto LABEL_74;
LABEL_16:
    v170 = 0;
    if ( v14 )
    {
      if ( (v15 = *(_BYTE *)(v7 + 80), v15 != 7) && v15 != 9
        || (v79 = *(_QWORD *)(v7 + 88)) == 0
        || (*(_BYTE *)(v79 + 170) & 0x10) == 0
        || (v170 = 0, !**(_QWORD **)(v79 + 216)) )
      {
        sub_864360(v14, 0);
        v170 = 1;
      }
    }
    v168 = dword_4D048B8;
    if ( dword_4D048B8 )
    {
      if ( v11 && v10 )
      {
        if ( (v11[5].m128i_i8[9] & 1) != 0 )
        {
          sub_733780(0, 0, 0, 1, 0);
          v168 = 0;
          v165 = unk_4F06BC0;
        }
        else
        {
          v168 = 0;
          v165 = 0;
        }
        v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      }
      else
      {
        v168 = 0;
        v165 = 0;
        v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      }
    }
    else
    {
      v165 = 0;
      v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    }
    goto LABEL_22;
  }
LABEL_15:
  v13 = sub_72C930(a1);
  v14 = *(_QWORD *)(v7 + 64);
  v12 = v13;
  if ( (*(_BYTE *)(v7 + 81) & 0x10) == 0 )
    goto LABEL_16;
LABEL_74:
  v170 = sub_8D23B0(v14);
  if ( v170 )
  {
LABEL_157:
    v170 = 0;
    v168 = 0;
    v165 = 0;
    v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    goto LABEL_22;
  }
  if ( *(_BYTE *)(v7 + 80) == 9 )
  {
    v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v109 = *(_BYTE *)(v16 + 4);
    if ( ((v109 - 15) & 0xFD) == 0 || v109 == 2 )
    {
      v168 = 0;
      v165 = 0;
      goto LABEL_22;
    }
  }
  if ( unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0 )
  {
    if ( dword_4F04C44 == -1 )
    {
      sub_8646E0(*(_QWORD *)(v7 + 64), 1);
      v168 = 1;
      v165 = 0;
      v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      goto LABEL_22;
    }
    goto LABEL_157;
  }
  v168 = 0;
  v165 = 0;
  v16 = 776LL * dword_4F04C64 + qword_4F04C68[0];
LABEL_22:
  v17 = *(_QWORD *)(v16 + 624);
  *(_QWORD *)(v16 + 624) = v6;
  v18 = *(_QWORD *)(v6 + 328);
  *(_BYTE *)(v6 + 178) |= 0xAu;
  v167 = v17;
  if ( v18 )
  {
    v172 = 27;
    if ( !a4 )
    {
      v19 = 35;
      if ( *(_BYTE *)(v18 + 8) == 1 )
        v19 = 73;
      v172 = v19;
    }
    v195 = *(__int64 **)sub_6E1A20(v18);
  }
  else
  {
    v172 = word_4F06418[0];
    v195 = *(__int64 **)&dword_4F063F8;
  }
  if ( dword_4F077C4 != 2 )
  {
    v17 = v169;
    if ( v169 )
      goto LABEL_30;
    if ( !unk_4D0421C )
    {
      v98 = sub_8D3B80(v12);
      if ( v172 != 73 || !v98 )
      {
LABEL_31:
        if ( *(char *)(v6 + 124) >= 0 )
          goto LABEL_32;
        goto LABEL_51;
      }
LABEL_30:
      *(_DWORD *)(v6 + 176) |= 0x400004u;
      goto LABEL_31;
    }
    if ( *(char *)(v6 + 124) >= 0 )
    {
      v166 = 0;
      v20 = 0;
      goto LABEL_34;
    }
LABEL_51:
    v36 = *(_BYTE *)(v12 + 140);
    if ( v36 == 12 )
    {
      v37 = v12;
      do
      {
        v37 = *(_QWORD *)(v37 + 160);
        v36 = *(_BYTE *)(v37 + 140);
      }
      while ( v36 == 12 );
    }
    if ( v36 )
    {
      if ( v172 != 73 || dword_4D04428 )
      {
        v17 = a4;
        sub_6BDE10(v6, a4);
        v12 = *(_QWORD *)(v6 + 288);
        if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v12) )
          sub_8AE000(v12);
        if ( (unsigned int)sub_8D23B0(v12) )
        {
          v17 = (__int64)a2;
          v73 = (unsigned int)sub_67F240(v12);
          sub_685A50(v73, a2, v12, 8);
          *a5 = 1;
          v12 = sub_72C930(v73);
        }
      }
      else
      {
        v38 = 2716;
        v17 = (__int64)dword_4F07508;
        if ( dword_4F077C4 == 2 )
          v38 = 1588;
        sub_6851C0(v38, dword_4F07508);
        v39 = sub_72C930(v38);
        v12 = v39;
        if ( v11 )
          v11[7].m128i_i64[1] = v39;
        v40 = sub_72C930(v38);
        *(_WORD *)(v6 + 124) &= 0xFC7Fu;
        *(_QWORD *)(v6 + 272) = v40;
        *(_QWORD *)(v6 + 280) = v40;
        *(_QWORD *)(v6 + 288) = v40;
        *(_QWORD *)(v6 + 304) = 0;
      }
    }
LABEL_32:
    if ( dword_4F077C4 != 2 )
    {
      v166 = 0;
      v20 = 0;
      goto LABEL_34;
    }
    goto LABEL_83;
  }
  v42 = 0;
  if ( v11 )
    v42 = (v11[10].m128i_i8[12] & 8) != 0;
  v8 = *(char *)(v6 + 124) < 0;
  *(_BYTE *)(v6 + 176) = *(_BYTE *)(v6 + 176) & 0xFB | (4 * v42);
  if ( v8 )
    goto LABEL_51;
LABEL_83:
  v166 = 0;
  v20 = sub_8D3AD0(v12);
  if ( v20 )
  {
    for ( i = v12; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v20 = 0;
    v166 = *(_QWORD *)(*(_QWORD *)i + 96LL);
    if ( word_4F06418[0] == 73
      && (*(_BYTE *)(v166 + 178) & 0x40) == 0
      && (*(_BYTE *)(v6 + 127) & 8) == 0
      && !dword_4D04428 )
    {
      v17 = v12;
      v20 = 1;
      sub_685380(285, v12);
      v166 = 0;
      v12 = sub_72C930(285);
    }
  }
LABEL_34:
  *(_QWORD *)(v6 + 288) = v12;
  if ( a4 )
  {
    v175 = sub_8DD3B0(v12);
    v191 = 0;
    v193 = 0;
    sub_6E2250(&v197, &v194, 4, 1, v6, v6 + 136);
    sub_6BE930(v12, 0, 0, &v193, &v191);
    sub_6E2C70(v194, 1, v6, v6 + 136);
    if ( v191 )
    {
      *(_BYTE *)(v6 + 179) |= 0x20u;
    }
    else if ( v166 && *(_QWORD *)(v166 + 8) )
    {
      if ( v193 )
        sub_6E1C20(v193, 1, v6 + 328);
      v196 = v195;
      if ( !v175 )
      {
        v17 = v12;
        v27 = v12;
        sub_6C64D0(v12, v12, (unsigned int)&v196, 1, 0, 0, v6 + 136);
LABEL_103:
        if ( a6 )
          a6[4].m128i_i64[1] = unk_4F061D8;
        goto LABEL_120;
      }
LABEL_102:
      v17 = 0;
      v27 = v6 + 136;
      sub_6C56C0(v6 + 136, 0);
      goto LABEL_103;
    }
    if ( v193 )
      sub_6E1C20(v193, 1, v6 + 328);
    if ( !v175 )
    {
      v27 = v6;
      v17 = a3;
      ++*(_BYTE *)(qword_4F061C8 + 36LL);
      sub_632300((__int64 *)v6, a3, 1u, (__int64)a2);
      v99 = *(_QWORD *)(v6 + 144);
      v174 = *(_QWORD *)(v6 + 136);
      v20 = (*(_BYTE *)(v6 + 177) & 2) != 0;
      if ( word_4F06418[0] == 28 && a6 )
      {
        v17 = (__int64)a6;
        a6[4].m128i_i64[1] = qword_4F063F0;
      }
      v160 = v99;
      --*(_BYTE *)(qword_4F061C8 + 36LL);
      sub_690B80();
      v35 = v160;
      if ( v174 && *(_BYTE *)(v174 + 173) == 10 && (*(_BYTE *)(v6 + 179) & 0x20) != 0 )
      {
        v17 = v174;
        *(_BYTE *)(v174 + 169) = *(_BYTE *)(v174 + 169) & 0x9F | 0x40;
      }
      if ( v166 )
      {
        if ( v160 )
        {
          if ( !*(_QWORD *)(v160 + 16) )
          {
            v17 = v12;
            v27 = v12;
            v100 = sub_87CF10(v12, v12, a2);
            v35 = v160;
            if ( v100 )
            {
              *(_QWORD *)(v160 + 16) = v100;
              if ( (*(_BYTE *)(v6 + 178) & 0x20) == 0 )
                *(_BYTE *)(v100 + 193) |= 0x40u;
            }
          }
        }
      }
      goto LABEL_121;
    }
    v196 = v195;
    goto LABEL_102;
  }
  if ( v172 == 73 )
  {
    v44 = *(_QWORD *)v6;
    v45 = *(_BYTE *)(*(_QWORD *)v6 + 80LL);
    v46 = (*(_BYTE *)(v6 + 127) & 8) != 0;
    if ( v45 == 9 || v45 == 7 )
    {
      v47 = *(_QWORD *)(v44 + 88);
    }
    else
    {
      v47 = 0;
      if ( v45 == 21 )
        v47 = *(_QWORD *)(*(_QWORD *)(v44 + 88) + 192LL);
    }
    v74 = dword_4D04428;
    if ( (*(_BYTE *)(v6 + 127) & 8) != 0 )
    {
      if ( !dword_4D04428 )
      {
        if ( dword_4F077BC )
        {
          sub_684B30(2068, &dword_4F063F8);
          v74 = dword_4D04428;
          v46 = 0;
        }
        else
        {
          if ( !(unsigned int)sub_8D97B0(v12) )
            sub_6851C0(702, &dword_4F063F8);
          v74 = dword_4D04428;
          v46 = 0;
        }
      }
      if ( v47 )
        *(_BYTE *)(v47 + 174) = (v46 << 6) | *(_BYTE *)(v47 + 174) & 0xBF;
      v75 = *(_QWORD *)(v6 + 288);
      *(_BYTE *)(v6 + 176) = *(_BYTE *)(v6 + 176) & 0xFE | v46;
    }
    else
    {
      *(_BYTE *)(v6 + 178) &= ~1u;
      v75 = v12;
    }
    if ( v74 )
    {
      if ( dword_4D04964 )
        *(_BYTE *)(v6 + 176) |= 0x80u;
      else
        *(_BYTE *)(v6 + 177) |= 1u;
    }
    if ( dword_4F077C4 != 2
      || (dword_4F04C64 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) == 0)
      && (v104 = sub_8D3BB0(v75), v75 = *(_QWORD *)(v6 + 288), v104) )
    {
      *(_BYTE *)(v6 + 179) |= 1u;
    }
    sub_637180(v75, 0, (__m128i *)(v6 + 136), (_QWORD *)v6, 1u, 0, a2);
    v17 = (__int64)a6;
    v27 = v142;
    if ( a6 )
      a6[4].m128i_i64[1] = unk_4F061D8;
    goto LABEL_120;
  }
  if ( !(unsigned int)sub_8D3B80(v12) )
  {
    v17 = 0;
    v48 = sub_6BB940(v6, 0, 0);
    v49 = v48;
    if ( a6 )
    {
      v17 = (__int64)a6;
      a6[4].m128i_i64[1] = *(_QWORD *)sub_6E1A60(v48);
    }
    v50 = *(_QWORD *)v6;
    if ( !*(_QWORD *)v6 )
      goto LABEL_119;
    v51 = *(_BYTE *)(v50 + 80);
    if ( v51 == 9 || v51 == 7 )
    {
      v101 = *(_QWORD *)(v50 + 88);
    }
    else
    {
      if ( v51 != 21 )
      {
LABEL_119:
        v27 = v49;
        sub_6E1990(v49);
LABEL_120:
        v35 = *(_QWORD *)(v6 + 144);
        v20 = (*(_BYTE *)(v6 + 177) & 2) != 0;
        v174 = *(_QWORD *)(v6 + 136);
        goto LABEL_121;
      }
      v101 = *(_QWORD *)(*(_QWORD *)(v50 + 88) + 192LL);
    }
    if ( v101 )
    {
      v17 = *(_QWORD *)(v6 + 288);
      sub_694AA0(v49, v17, 1, 1, v6 + 136);
      if ( *(_QWORD *)(v6 + 136) )
      {
        v102 = sub_6E1A20(v49);
        if ( (*(_BYTE *)(v6 + 176) & 6) == 6 )
        {
          v103 = *(_QWORD *)(v6 + 136);
          if ( *(_BYTE *)(v103 + 173) == 6 )
          {
            v17 = *(_QWORD *)(v103 + 136);
            sub_630E60(*(_QWORD *)(v103 + 128), v17, *(_QWORD *)(v6 + 288), 0, v102);
          }
        }
      }
    }
    goto LABEL_119;
  }
  v21 = v12;
  if ( !(unsigned int)sub_8D3A70(v12) )
  {
    v25 = HIDWORD(qword_4F077B4);
    if ( HIDWORD(qword_4F077B4) )
    {
      if ( v169 )
      {
        if ( dword_4F077C4 != 2 )
          goto LABEL_41;
        if ( (*(_BYTE *)(v6 + 131) & 0x10) == 0 )
        {
          v21 = sub_8D4130(v12);
          if ( !(unsigned int)sub_8D3A70(v21) )
            goto LABEL_41;
          v21 = sub_8D4130(v12);
          if ( (unsigned int)sub_8D3BB0(v21) )
            goto LABEL_41;
        }
      }
    }
    goto LABEL_174;
  }
  if ( dword_4F077C4 != 2 )
  {
    v22 = v169;
    if ( v169 )
    {
      v25 = HIDWORD(qword_4F077B4);
      if ( HIDWORD(qword_4F077B4) )
      {
LABEL_41:
        v26 = sub_724DC0(v21, v17, v25, v22, v23, v24);
        v17 = v6;
        v197 = v26;
        sub_6D6AC0(v12, v6, v26);
        v27 = (__int64)&v197;
        v174 = sub_724E50(&v197, v6, v28, v29, v30);
        if ( (v171 & 1) != 0 || !v11 )
          goto LABEL_46;
        v27 = v11[7].m128i_i64[1];
        for ( j = v27; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v155 = j;
        if ( !(unsigned int)sub_8D23B0(v27) )
          goto LABEL_46;
        v130 = sub_8D3410(*(_QWORD *)(v174 + 128));
        v131 = v155;
        if ( v130 )
        {
          v132 = *(_QWORD *)(*(_QWORD *)(v174 + 128) + 176LL);
          if ( *(_BYTE *)(v155 + 140) != 12 )
            goto LABEL_367;
        }
        else
        {
          if ( *(_BYTE *)(v155 + 140) != 12 )
          {
            v20 = 1;
            v17 = sub_7259C0(8);
            sub_73C230(v155, v17);
            v133 = v17;
            *(_QWORD *)(v17 + 176) = 1;
LABEL_369:
            v27 = v133;
            v161 = v133;
            sub_8D6090(v133);
            v11[7].m128i_i64[1] = v161;
LABEL_46:
            if ( a6 )
              a6[4].m128i_i64[1] = unk_4F061D8;
            v35 = 0;
            goto LABEL_121;
          }
          v132 = 1;
          v20 = 1;
        }
        do
          v131 = *(_QWORD *)(v131 + 160);
        while ( *(_BYTE *)(v131 + 140) == 12 );
LABEL_367:
        v147 = v132;
        v154 = v131;
        v17 = sub_7259C0(8);
        sub_73C230(v154, v17);
        v133 = v17;
        *(_QWORD *)(v17 + 176) = v147;
        if ( !v147 )
          *(_BYTE *)(v17 + 169) |= 0x20u;
        goto LABEL_369;
      }
LABEL_174:
      for ( k = *(_QWORD *)(v6 + 288); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      v63 = *(_QWORD *)v6;
      v64 = *(_BYTE *)(*(_QWORD *)v6 + 80LL);
      if ( v64 == 9 || v64 == 7 )
      {
        v176 = *(_QWORD *)(v63 + 88);
      }
      else
      {
        v176 = 0;
        if ( v64 == 21 )
          v176 = *(_QWORD *)(*(_QWORD *)(v63 + 88) + 192LL);
      }
      *(_BYTE *)(v6 + 178) &= ~1u;
      v144 = *(_BYTE *)(k + 140);
      if ( v144 != 8 )
        goto LABEL_181;
      v148 = sub_8D3880(k);
      if ( v148 )
      {
        v143 = 0;
        v148 = 0;
        v163 = 1;
        v162 = 3;
      }
      else
      {
        if ( (*(_BYTE *)(v6 + 131) & 0x10) != 0 || dword_4F077C4 != 2 )
        {
LABEL_181:
          v143 = 0;
          v148 = 0;
          v163 = 0;
          v162 = 3;
          goto LABEL_182;
        }
        if ( !dword_4F077BC || (unsigned int)sub_8D23E0(k) )
          goto LABEL_344;
        v113 = *(_QWORD *)(k + 160);
        for ( m = *(_BYTE *)(v113 + 140); m == 12; m = *(_BYTE *)(v113 + 140) )
          v113 = *(_QWORD *)(v113 + 160);
        if ( (unsigned __int8)(m - 9) > 2u || (v143 = sub_8D3BB0(v113)) != 0 )
        {
LABEL_344:
          sub_6851C0(520, &dword_4F063F8);
          v162 = 8;
          v163 = 0;
          v143 = 1;
        }
        else
        {
          v163 = 0;
          v162 = 3;
          v148 = 1;
        }
      }
LABEL_182:
      v17 = 0;
      v65 = sub_6BB940(v6, 0, 0);
      v66 = v65;
      if ( a6 )
      {
        v157 = v65;
        v67 = (__int64 *)sub_6E1A60(v65);
        v66 = v157;
        a6[4].m128i_i64[1] = *v67;
      }
      v158 = (__int64 *)v66;
      v68 = sub_6E1A80(v66);
      v69 = v158;
      if ( !v68 )
      {
        v159 = (__m128i *)(v6 + 136);
        if ( v144 == 8 && (*(_BYTE *)(v6 + 131) & 0x10) != 0 )
        {
          v17 = (__int64)v69;
          v151 = v69;
          sub_692C90(v6, v69);
          sub_6E1990(v151);
          goto LABEL_198;
        }
        if ( v163 )
        {
          v145 = (__int64)v69;
          v110 = sub_8D3880(*(_QWORD *)(v6 + 288));
          v69 = (__int64 *)v145;
          if ( v110 )
          {
            v17 = v6 + 288;
            v111 = sub_6320D0(v145, (__int64 *)(v6 + 288), (__int64)v159, (__int64)v159);
            v69 = (__int64 *)v145;
            if ( v111 )
            {
              sub_6E1990(v145);
              goto LABEL_198;
            }
          }
        }
        if ( v148 )
        {
          v115 = *(_QWORD *)(k + 160);
          *(_BYTE *)(v6 + 177) |= 0x80u;
          v116 = *(_QWORD *)(v6 + 288);
          v196 = v69;
          v146 = v69;
          v152 = v116;
          sub_634B10((__int64 *)&v196, v115, 0, v159, (__int64)a2, &v197);
          if ( (*(_WORD *)(k + 168) & 0x180) != 0 )
          {
            v128 = sub_724D50(10);
            v129 = v197;
            *(_QWORD *)(v6 + 136) = v128;
            sub_72A690(v129, v128, 0, 0);
            v117 = *(_QWORD *)(v6 + 136);
            v118 = v146;
          }
          else
          {
            v117 = sub_62FF50(v197, k);
            v118 = v146;
            *(_QWORD *)(v6 + 136) = v117;
          }
          v17 = v152;
          v153 = v118;
          *(_QWORD *)(v117 + 128) = v17;
          *(_QWORD *)(*(_QWORD *)(v6 + 136) + 64LL) = *(_QWORD *)sub_6E1A20(v118);
          *(_QWORD *)(*(_QWORD *)(v6 + 136) + 112LL) = *(_QWORD *)sub_6E1A60(v153);
          sub_6E1990(v153);
LABEL_198:
          if ( !*(_QWORD *)(v6 + 144) )
          {
            v17 = 0;
            sub_630880(v159->m128i_i64, 0);
          }
LABEL_291:
          v27 = *(_QWORD *)(v176 + 120);
          if ( (unsigned int)sub_8D23E0(v27) )
          {
            v27 = *(_QWORD *)(v6 + 288);
            v107 = sub_8D3410(v27);
            v32 = *(_QWORD *)(v6 + 288);
            if ( v107 )
              goto LABEL_297;
            v31 = *(unsigned __int8 *)(v32 + 140);
            if ( (_BYTE)v31 == 12 )
            {
              v108 = *(_QWORD *)(v6 + 288);
              do
              {
                v108 = *(_QWORD *)(v108 + 160);
                v31 = *(unsigned __int8 *)(v108 + 140);
              }
              while ( (_BYTE)v31 == 12 );
            }
            if ( !(_BYTE)v31 )
            {
LABEL_297:
              v17 = *(_QWORD *)v6;
              v27 = v176;
              sub_6301D0(v176, *(_QWORD *)v6, (__int64)a2, a3, v32);
              *(_QWORD *)(v6 + 288) = *(_QWORD *)(v176 + 120);
            }
          }
          goto LABEL_120;
        }
        if ( !v143 && dword_4F077C4 != 1 )
        {
          if ( dword_4F077C4 == 2 )
          {
            v162 = 8;
            v70 = 8;
          }
          else if ( dword_4F077C0 | dword_4D04964 )
          {
            v70 = byte_4F07472[0];
            v162 = byte_4F07472[0];
          }
          else
          {
            v162 = 5;
            v70 = 5;
          }
          v149 = v69;
          v71 = sub_6E1A20(v69);
          v17 = 520;
          sub_684AA0(v70, 520, v71);
          v69 = v149;
        }
        v196 = v69;
        if ( v162 != 8 )
        {
          v150 = v69;
          if ( (unsigned int)sub_8D3410(*(_QWORD *)(v6 + 288)) )
          {
            *(_BYTE *)(v6 + 178) |= 8u;
            v72 = (_QWORD *)sub_6E1A20(v196);
            v17 = v6 + 288;
            sub_635980(&v196, (__int64 *)(v6 + 288), v159, v72, v159->m128i_i64);
          }
          else
          {
            v127 = sub_6E1A20(v196);
            v17 = *(_QWORD *)(v6 + 288);
            sub_6333F0((__int64 *)&v196, v17, v159, v127, v159->m128i_i64);
          }
          sub_6E1990(v150);
          goto LABEL_198;
        }
      }
      sub_6E1990(v69);
      v105 = sub_72C9A0();
      v106 = *(_QWORD *)(v6 + 288);
      *(_BYTE *)(v6 + 177) |= 2u;
      *(_QWORD *)(v6 + 136) = v105;
      if ( (unsigned int)sub_8D23E0(v106) )
        *(_QWORD *)(v6 + 288) = sub_72C930(v106);
      goto LABEL_291;
    }
  }
  v27 = v6;
  sub_6D73D0(v6);
  v35 = *(_QWORD *)(v6 + 144);
  v20 = (*(_BYTE *)(v6 + 177) & 2) != 0;
  v174 = *(_QWORD *)(v6 + 136);
  if ( a6 )
    a6[4].m128i_i64[1] = unk_4F061D8;
LABEL_121:
  if ( *(_QWORD *)(v6 + 328) )
  {
    v27 = v6 + 328;
    v156 = v35;
    sub_6E1BF0(v6 + 328);
    v35 = v156;
  }
  if ( !v171 )
  {
    v194 = 0;
    v52 = v11[10].m128i_i8[14] & 0xCF | (32 * (a4 & 1)) | 0x10;
    v11[10].m128i_i8[14] = v11[10].m128i_i8[14] & 0xCF | (32 * (a4 & 1)) | 0x10;
    if ( (*(_BYTE *)(v6 + 176) & 4) != 0
      && v174
      && *(_BYTE *)(v174 + 173) == 6
      && *(_BYTE *)(v174 + 176) == 1
      && *(_BYTE *)(*(_QWORD *)(v174 + 184) + 136LL) > 2u )
    {
      v17 = (__int64)&v195;
      v27 = 28;
      sub_6851C0(28, &v195);
      goto LABEL_129;
    }
    if ( v20 )
    {
LABEL_129:
      v20 = 1;
      v197 = sub_724DC0(v27, v17, v31, v52, v32, v33);
      sub_72C970(v197);
      v174 = sub_724E50(&v197, v17, v53, v54, v55);
LABEL_130:
      if ( v11[8].m128i_i8[8] <= 2u )
      {
        if ( !(unsigned int)sub_8D23E0(v11[7].m128i_i64[1]) )
          goto LABEL_132;
        v95 = *(_QWORD *)(v6 + 288);
        if ( !(unsigned int)sub_8D3410(v95) )
          goto LABEL_132;
        v57 = 0;
LABEL_255:
        if ( (*(_BYTE *)(v6 + 177) & 2) != 0 )
        {
          v187 = v57;
          v112 = sub_72C930(v95);
          v57 = v187;
          v97 = v112;
        }
        else
        {
          v97 = *(_QWORD *)(v6 + 288);
        }
        v186 = v57;
        sub_6301D0((__int64)v11, *(_QWORD *)v6, (__int64)a2, a3, v97);
        v57 = v186;
        *(_QWORD *)(v6 + 288) = v11[7].m128i_i64[1];
LABEL_258:
        if ( v57 )
          goto LABEL_142;
LABEL_132:
        if ( v11[8].m128i_i8[8] <= 2u && (v11[5].m128i_i8[9] & 1) != 0 )
        {
          if ( (*(_BYTE *)(v174 - 8) & 1) != 0 )
          {
            v11[11].m128i_i64[1] = v174;
            v11[11].m128i_i8[1] = 1;
          }
          else if ( *(_BYTE *)(v174 + 173) == 10 || (unsigned int)sub_72AA80(v174) )
          {
            v192 = sub_7333B0(v11, 0, 1, v174, 0);
          }
          else
          {
            sub_7296C0(&v197);
            v11[11].m128i_i64[1] = sub_7401F0(v174);
            sub_729730((unsigned int)v197);
            v11[11].m128i_i8[1] = 1;
          }
        }
        else
        {
          v11[11].m128i_i8[1] = 1;
          v11[11].m128i_i64[1] = v174;
        }
        goto LABEL_143;
      }
      v35 = sub_725A70(2);
      *(_QWORD *)(v35 + 56) = v174;
      v61 = ((v172 == 73) << 6) | *(_BYTE *)(v35 + 50) & 0xBF;
      *(_BYTE *)(v35 + 50) = v61;
      v174 = 0;
      if ( (*(_BYTE *)(v6 + 177) & 0x10) != 0 )
        *(_BYTE *)(v35 + 50) = v61 | 0x80;
      goto LABEL_141;
    }
    if ( v35 )
    {
      if ( word_4D04898 )
      {
        if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
        {
          v181 = v35;
          v80 = sub_867AA0();
          v35 = v181;
          if ( !v80 )
          {
            v85 = (__int64 *)sub_724DC0(v27, v17, v81, v82, v83, v84);
            v86 = v181;
            v87 = 0;
            v196 = v85;
            if ( *(_BYTE *)(v181 + 48) == 5 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) & 0x20) == 0 )
            {
              v136 = *(_QWORD *)(v181 + 56);
              if ( v136 )
                v87 = (*(_BYTE *)(v136 + 193) & 4) != 0;
            }
            if ( (v11[5].m128i_i8[9] & 1) == 0 || v11[8].m128i_i8[8] <= 2u || (v11[10].m128i_i8[12] & 8) != 0 )
              goto LABEL_350;
            v88 = v11[7].m128i_i64[1];
            if ( (*(_BYTE *)(v88 + 140) & 0xFB) == 8 )
            {
              v89 = sub_8D4C10(v88, dword_4F077C4 != 2);
              v86 = v181;
              if ( (v89 & 1) != 0 )
              {
                v119 = sub_8D2930(v11[7].m128i_i64[1]);
                v86 = v181;
                if ( v119 )
                  goto LABEL_350;
              }
              v88 = v11[7].m128i_i64[1];
            }
            v182 = v86;
            v90 = sub_8D32E0(v88);
            v86 = v182;
            v91 = v90;
            if ( !v90 )
            {
              v197 = 0;
              v92 = 1;
              v198 = 0;
              if ( *(_QWORD *)(v182 + 8) )
              {
LABEL_250:
                if ( *(_BYTE *)(v86 + 48) == 2 && v92 )
                {
                  v183 = v86;
                  v20 = 0;
                  sub_724E30(&v196);
                  v93 = v183;
                  goto LABEL_253;
                }
LABEL_353:
                v188 = v86;
                v120 = sub_7A1C60(
                         v86,
                         (unsigned int)&v195,
                         *(_QWORD *)(v6 + 288),
                         v91,
                         (_DWORD)v196,
                         (unsigned int)&v197,
                         0);
                v123 = v188;
                v20 = v120;
                if ( v120 )
                {
                  v124 = v196[18];
                  v20 = *((_BYTE *)v196 + 173) == 0;
                  if ( v124 && *(_BYTE *)(v124 + 24) == 31 )
                  {
                    *(_QWORD *)(v124 + 36) = v195;
                    v121 = unk_4F061D8;
                    *(_QWORD *)(v124 + 44) = unk_4F061D8;
                  }
                  if ( v169 && ((v125 = *(_QWORD *)(v188 + 16)) == 0 || (*(_BYTE *)(v125 + 194) & 8) != 0) )
                  {
                    v126 = sub_724E50(&v196, &v195, v121, v122, v141);
                    v93 = 0;
                    v174 = v126;
                  }
                  else
                  {
                    v190 = sub_725A70(2);
                    v140 = sub_724E50(&v196, &v195, v137, v138, v139);
                    sub_72F900(v190, v140);
                    v93 = v190;
                  }
                }
                else
                {
                  if ( (*(_BYTE *)(v6 + 176) & 4) != 0
                    || v87
                    || (v11[10].m128i_i8[12] & 0x10) != 0 && *(_BYTE *)(v188 + 48) > 2u )
                  {
                    v20 = 1;
                    v134 = sub_67D9D0(28, &v195);
                    sub_67E370(v134, &v197);
                    sub_685910(v134);
                    v135 = sub_72C9A0();
                    v123 = 0;
                    v174 = v135;
                  }
                  v189 = v123;
                  sub_724E30(&v196);
                  v93 = v189;
                }
LABEL_253:
                v184 = v93;
                sub_67E3D0(&v197);
                v94 = sub_8D23E0(v11[7].m128i_i64[1]);
                v57 = v184;
                if ( !v94 )
                  goto LABEL_258;
LABEL_254:
                v95 = *(_QWORD *)(v6 + 288);
                v185 = v57;
                v96 = sub_8D3410(v95);
                v57 = v185;
                if ( !v96 )
                  goto LABEL_258;
                goto LABEL_255;
              }
LABEL_382:
              *(_QWORD *)(v86 + 8) = v11;
              goto LABEL_250;
            }
LABEL_350:
            if ( (*(_BYTE *)(v6 + 10) & 0x20) != 0 )
              v11[10].m128i_i8[12] |= 0x10u;
            v197 = 0;
            v91 = 1;
            v198 = 0;
            if ( *(_QWORD *)(v86 + 8) )
              goto LABEL_353;
            v92 = 0;
            v91 = 1;
            goto LABEL_382;
          }
        }
      }
    }
    else
    {
      if ( !v166 )
        goto LABEL_130;
      v76 = sub_87CF10(v12, v12, a2);
      if ( !v76 )
        goto LABEL_130;
      v35 = sub_725A70(2);
      *(_QWORD *)(v35 + 56) = v174;
      v77 = ((v172 == 73) << 6) | *(_BYTE *)(v35 + 50) & 0xBF;
      *(_BYTE *)(v35 + 50) = v77;
      v78 = (*(_BYTE *)(v6 + 177) & 0x10) == 0;
      *(_QWORD *)(v35 + 16) = v76;
      if ( !v78 )
        v77 |= 0x80u;
      *(_BYTE *)(v35 + 50) = v77;
      if ( (*(_BYTE *)(v6 + 178) & 0x20) == 0 )
        *(_BYTE *)(v76 + 193) |= 0x40u;
      v174 = 0;
    }
LABEL_141:
    v180 = v35;
    v56 = sub_8D23E0(v11[7].m128i_i64[1]);
    v57 = v180;
    if ( !v56 )
    {
LABEL_142:
      sub_630370((__int64)v11, v57, &v192, (__int64)a2, (__int64)a6, &v194);
LABEL_143:
      v27 = v6;
      sub_649FB0(v6);
      v31 = v194;
      if ( v194 )
      {
        if ( v192 )
        {
          v58 = *(_BYTE *)(v192 + 16);
          v59 = (__int64 *)(v192 + 24);
        }
        else
        {
          v58 = v11[11].m128i_i8[1];
          v59 = &v11[11].m128i_i64[1];
        }
        if ( v58 == 2 )
          *(_QWORD *)(v194 + 72) = *v59;
      }
      if ( a6 )
        v11[12] = _mm_loadu_si128(a6 + 4);
      goto LABEL_150;
    }
    goto LABEL_254;
  }
LABEL_150:
  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_QWORD *)(result + 624) = v167;
  if ( (*(_BYTE *)(v7 + 81) & 0x10) != 0 )
  {
    if ( v168 )
      return sub_866010(v27, v168, v31, v167, v32);
  }
  else
  {
    if ( v165 )
      result = sub_630710(v165, v192, v20 | (unsigned int)v171);
    if ( v170 )
      return sub_8645D0();
  }
  return result;
}
