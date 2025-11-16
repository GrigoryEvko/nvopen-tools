// Function: sub_65F400
// Address: 0x65f400
//
__int16 __fastcall sub_65F400(__int64 a1, __int64 a2, const __m128i *a3)
{
  _QWORD *v3; // r14
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rsi
  __m128i *v10; // r15
  __int64 v11; // rax
  char v12; // r13
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rax
  char j; // dl
  _BOOL4 v17; // ecx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  int v21; // r12d
  char v22; // al
  __int16 result; // ax
  char v24; // al
  char v25; // al
  char v26; // al
  __int64 v27; // rdi
  unsigned __int64 v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rdx
  __int64 m; // rax
  __int64 v34; // rdi
  bool v35; // r12
  __int64 v36; // rdx
  char v37; // al
  __int64 v38; // rdi
  char *v39; // rdx
  __int64 v40; // rdi
  char v41; // al
  int v42; // eax
  char v43; // al
  char v44; // al
  __int64 v45; // r13
  __int64 v46; // rdi
  __int64 v47; // rdi
  char v48; // al
  __int64 v49; // r13
  __int64 v50; // rdi
  char v51; // al
  __int64 v52; // r15
  __m128i v53; // xmm6
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rax
  int v58; // r13d
  char v59; // al
  __int64 v60; // rax
  char v61; // al
  int v62; // eax
  unsigned __int64 v63; // rdx
  __int64 v64; // r12
  unsigned int v65; // eax
  __int64 v66; // r11
  char v67; // dl
  char v68; // al
  __int64 i; // rdi
  int v70; // eax
  __int64 v71; // r11
  unsigned __int64 v72; // rax
  char v73; // dl
  __int64 v74; // rdi
  int v75; // eax
  __int64 v76; // r11
  __int64 v77; // rax
  char *v78; // rdx
  __int64 v79; // r12
  __int64 k; // r12
  int v81; // eax
  char v82; // al
  __int64 v83; // rax
  _BOOL4 v84; // [rsp+8h] [rbp-A8h]
  __int64 v85; // [rsp+8h] [rbp-A8h]
  __int64 v86; // [rsp+8h] [rbp-A8h]
  __int64 v87; // [rsp+8h] [rbp-A8h]
  __int64 v88; // [rsp+8h] [rbp-A8h]
  __int64 v89; // [rsp+8h] [rbp-A8h]
  __int64 v90; // [rsp+8h] [rbp-A8h]
  __int64 v91; // [rsp+8h] [rbp-A8h]
  __int64 v92; // [rsp+8h] [rbp-A8h]
  int v93; // [rsp+20h] [rbp-90h]
  __int64 v94; // [rsp+20h] [rbp-90h]
  __int64 v95; // [rsp+20h] [rbp-90h]
  char v96; // [rsp+28h] [rbp-88h]
  unsigned int v97; // [rsp+38h] [rbp-78h]
  unsigned int *v98; // [rsp+38h] [rbp-78h]
  unsigned int v99; // [rsp+38h] [rbp-78h]
  __int64 v100; // [rsp+38h] [rbp-78h]
  __int64 v101; // [rsp+38h] [rbp-78h]
  int v102; // [rsp+40h] [rbp-70h]
  int v103; // [rsp+44h] [rbp-6Ch]
  char v104; // [rsp+44h] [rbp-6Ch]
  __int64 v105; // [rsp+48h] [rbp-68h]
  unsigned int v106; // [rsp+58h] [rbp-58h]
  char v107; // [rsp+67h] [rbp-49h] BYREF
  int v108; // [rsp+68h] [rbp-48h] BYREF
  unsigned int v109; // [rsp+6Ch] [rbp-44h] BYREF
  __int64 v110; // [rsp+70h] [rbp-40h] BYREF
  _BYTE v111[56]; // [rsp+78h] [rbp-38h] BYREF

  v3 = (_QWORD *)a1;
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_QWORD *)(a1 + 288);
  v108 = 0;
  v109 = 0;
  v96 = v6;
  v105 = v6 & 1;
  if ( (unsigned int)sub_8D2600(v7) )
    sub_6426B0(v7, a1 + 48);
  sub_657FD0(a1, a2, 1);
  v8 = *(_QWORD *)(a2 + 24);
  if ( v8 && (*(_BYTE *)(v8 + 81) & 0x10) != 0 )
  {
    v9 = (unsigned int)dword_4F04C5C;
    if ( dword_4F04C34 == dword_4F04C5C )
    {
      v59 = *(_BYTE *)(a1 + 269);
      if ( v59 == 3 || v59 == 5 )
      {
        v9 = (__int64)&a3->m128i_i64[1];
        a1 = 149;
        sub_6851C0(149, &a3->m128i_u64[1]);
        *((_BYTE *)v3 + 269) = 0;
      }
    }
    v102 = 1;
    goto LABEL_7;
  }
  v24 = *(_BYTE *)(a1 + 269);
  v9 = (unsigned int)dword_4F04C34;
  if ( dword_4F04C5C == dword_4F04C34 && (v24 == 3 || !*(_QWORD *)(a1 + 240) && v24 == 5) )
  {
    v9 = (__int64)&a3->m128i_i64[1];
    a1 = 149;
    sub_6851C0(149, &a3->m128i_u64[1]);
    *((_BYTE *)v3 + 269) = 0;
  }
  else if ( v24 )
  {
LABEL_84:
    v102 = 0;
    goto LABEL_7;
  }
  v25 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4);
  if ( ((v25 - 15) & 0xFD) != 0 && v25 != 2 && !v3[45] )
  {
    if ( (*((_BYTE *)v3 + 126) & 2) != 0 )
      *((_BYTE *)v3 + 269) = 1;
    goto LABEL_84;
  }
  if ( (*((_BYTE *)v3 + 10) & 0x40) != 0
    || (v9 = v3[23]) != 0 && (a1 = 90, sub_736C60(90, v9))
    || (v9 = v3[25]) != 0 && (a1 = 90, sub_736C60(90, v9)) )
  {
    *((_BYTE *)v3 + 269) = 2;
    v102 = 0;
    if ( (*((_BYTE *)v3 + 130) & 0x20) != 0 )
      goto LABEL_8;
LABEL_59:
    if ( !v105 )
    {
      if ( word_4F06418[0] != 56 && (dword_4F077C4 != 2 || word_4F06418[0] != 73) )
      {
        v103 = 0;
        goto LABEL_11;
      }
      a3[4].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
    }
    v10 = (__m128i *)v3[45];
    v103 = 1;
    *((_BYTE *)v3 + 127) |= 4u;
    if ( v10 )
      goto LABEL_12;
    goto LABEL_61;
  }
  *((_BYTE *)v3 + 269) = 3;
  v102 = 0;
LABEL_7:
  if ( (*((_BYTE *)v3 + 130) & 0x20) == 0 )
    goto LABEL_59;
LABEL_8:
  if ( word_4F06418[0] != 55 )
  {
    sub_7B80F0();
    a1 = 53;
    ++*(_BYTE *)(qword_4F061C8 + 63LL);
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    sub_6851D0(53);
    --*(_BYTE *)(qword_4F061C8 + 63LL);
    sub_7B8160();
  }
  v103 = 0;
LABEL_11:
  v10 = (__m128i *)v3[45];
  if ( v10 )
  {
LABEL_12:
    v11 = v10->m128i_i64[1];
    v12 = 0;
    v84 = 0;
    v97 = 0;
    *v3 = v11;
    v13 = *(_QWORD *)(a2 + 8);
    v93 = 0;
    *(_QWORD *)(v11 + 48) = v13;
    v10[1].m128i_i64[0] = v3[36];
    v10[2].m128i_i64[0] = v3[3];
    v10[2].m128i_i8[8] = *((_BYTE *)v3 + 269);
    v10[4] = _mm_loadu_si128(a3 + 2);
    v10[5] = _mm_loadu_si128(a3 + 3);
    v10[6] = _mm_loadu_si128(a3 + 1);
    v14 = 0;
    goto LABEL_13;
  }
LABEL_61:
  if ( v102 )
  {
    if ( (*((_BYTE *)v3 + 10) & 8) != 0 )
    {
      v47 = v3[36];
      if ( (*(_BYTE *)(v47 + 140) & 0xFB) == 8 )
      {
        if ( (sub_8D4C10(v47, dword_4F077C4 != 2) & 1) != 0 )
          goto LABEL_221;
        v47 = v3[36];
      }
      v3[36] = sub_73C570(v47, 1, -1);
    }
LABEL_221:
    v48 = *((_BYTE *)v3 + 269);
    v49 = *(_QWORD *)(a2 + 24);
    if ( v48 && *((_BYTE *)v3 + 268) )
    {
      if ( dword_4F077BC && v48 == 1 )
      {
        *((_BYTE *)v3 + 269) = 0;
        v50 = 5;
      }
      else
      {
        v50 = 8;
      }
      sub_684AA0(v50, 80, (char *)v3 + 260);
    }
    v51 = *(_BYTE *)(v49 + 80);
    v98 = (unsigned int *)(a2 + 8);
    if ( v51 == 9 )
    {
      *v3 = v49;
      v66 = *(_QWORD *)(v49 + 88);
      v3[37] = *(_QWORD *)(v66 + 120);
      if ( dword_4D04820 && (*(_BYTE *)(v66 + 172) & 8) != 0 && !v103 )
      {
        v95 = 1;
        v67 = 0;
      }
      else
      {
        v95 = 3;
        v67 = 1;
      }
      v68 = v67 | *((_BYTE *)v3 + 122) & 0xFE;
      *((_BYTE *)v3 + 122) = v68;
      if ( (v68 & 1) != 0 && (*(_BYTE *)(v49 + 81) & 2) != 0 )
      {
        sub_685920(v98, v49, 8);
        goto LABEL_233;
      }
      v85 = v66;
      if ( !(unsigned int)sub_85ED80(v49, qword_4F04C68[0] + 776LL * dword_4F04C64) )
        goto LABEL_310;
      for ( i = v3[36]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v70 = sub_8D3EA0(i);
      v71 = v85;
      if ( v70 || (v81 = sub_64E420(v49, v3[36], v98), v71 = v85, !v81) )
      {
        if ( *(_BYTE *)(v71 + 136) == 1 )
          *(_BYTE *)(v71 + 136) = 0;
        v72 = v3[1];
        v73 = *(_BYTE *)(v71 + 172);
        if ( (v72 & 0x80000) != 0 && (v73 & 8) == 0 )
        {
          if ( (*((_BYTE *)v3 + 122) & 1) != 0 )
          {
            v73 |= 8u;
            *(_BYTE *)(v71 + 172) = v73;
            v72 = v3[1];
          }
          else
          {
            v92 = v71;
            sub_6854C0(2384, v3 + 4, v49);
            v71 = v92;
            v72 = v3[1] & 0xFFFFFFFFFFF7FFFFLL;
            v3[1] = v72;
            v73 = *(_BYTE *)(v92 + 172);
          }
        }
        if ( (v73 & 0x10) != 0 && (v72 & 0x200000) == 0 )
        {
          v90 = v71;
          sub_6854F0(8, 3116, v3 + 4, v71 + 64);
          v72 = v3[1];
          v71 = v90;
        }
        if ( ((*(_BYTE *)(v71 + 176) & 8) != 0) != ((v72 & 0x400000) != 0) )
        {
          v86 = v71;
          sub_6854F0(8, (unsigned int)((*(_BYTE *)(v71 + 176) & 8) != 0) + 2502, v98, v71 + 64);
          v71 = v86;
        }
        *(_BYTE *)(v71 + 88) |= 4u;
        if ( *(_QWORD *)(v49 + 96) )
        {
          v87 = v71;
          sub_648C10(v49, (__int64)v98);
          v71 = v87;
          *(_WORD *)(v87 + 170) |= 0x180u;
        }
        if ( v103 )
          goto LABEL_292;
        if ( (*(_BYTE *)(v71 + 176) & 1) == 0 )
        {
          v74 = *(_QWORD *)(v71 + 120);
          if ( (*(_BYTE *)(v74 + 140) & 0xFB) == 8
            && (v91 = v71,
                v82 = sub_8D4C10(v74, dword_4F077C4 != 2),
                v71 = v91,
                v74 = *(_QWORD *)(v91 + 120),
                (v82 & 1) != 0) )
          {
            v75 = sub_8D5940(v74, 1, 0);
            v71 = v91;
          }
          else
          {
            v88 = v71;
            v75 = sub_8D5940(v74, 0, 1);
            v71 = v88;
          }
          if ( v75 )
LABEL_292:
            v95 |= 0x800uLL;
        }
        v89 = v71;
        sub_8756F0(v95, v49, v98, v3[44]);
        v76 = v89;
        if ( dword_4F04C64 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0 )
        {
          if ( dword_4F077C4 == 2 && (*(_BYTE *)(v89 - 8) & 1) != 0 && (*(_BYTE *)(a2 + 18) & 0x40) == 0 )
          {
            v83 = sub_7CAFF0(a2, v89, qword_4F04C68);
            v76 = v89;
            v10 = (__m128i *)v83;
          }
          if ( (v95 & 2) != 0 && (*(_BYTE *)(v76 + 176) & 1) == 0 )
            v10[2].m128i_i8[1] |= 0x10u;
        }
        v77 = v3[44];
        if ( v77 && *(_BYTE *)(v77 + 16) == 53 )
        {
          v100 = v76;
          sub_86A3D0(v76, v3[35], v10, (unsigned __int8)((*((_BYTE *)v3 + 126) & 4) != 0) << 6, a3);
          v76 = v100;
        }
        v101 = v76;
        sub_644920(v3, *((_BYTE *)v3 + 122) & 1);
        sub_729470(v101, a3);
        goto LABEL_234;
      }
    }
    else
    {
      if ( v51 != 8 )
      {
        if ( (unsigned int)sub_85ED80(v49, qword_4F04C68[0] + 776LL * dword_4F04C64) )
        {
          if ( *(_BYTE *)(v49 + 80) == 16 || (*(_BYTE *)(v49 + 84) & 2) != 0 )
            sub_6851C0(298, v98);
          else
            sub_6854C0(147, v98, v49);
          goto LABEL_233;
        }
LABEL_310:
        sub_6854E0(551, v49);
        goto LABEL_233;
      }
      sub_6851C0(246, v98);
    }
LABEL_233:
    v52 = *(_QWORD *)(v49 + 64);
    v94 = *(_QWORD *)a2;
    sub_8756F0(1, v49, v98, v3[44]);
    *(__m128i *)a2 = _mm_loadu_si128(xmmword_4F06660);
    *(__m128i *)(a2 + 16) = _mm_loadu_si128(&xmmword_4F06660[1]);
    *(__m128i *)(a2 + 32) = _mm_loadu_si128(&xmmword_4F06660[2]);
    v53 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v54 = *(_QWORD *)dword_4F07508;
    *(_BYTE *)(a2 + 17) |= 0x20u;
    *(__m128i *)(a2 + 48) = v53;
    *(_QWORD *)(a2 + 8) = v54;
    v49 = sub_885AD0(9, a2, 0, 1);
    *(_QWORD *)v49 = v94;
    v99 = dword_4F04C34;
    v55 = sub_72C930(9);
    v57 = sub_735FB0(v55, 2, v99, v56);
    *(_QWORD *)(v49 + 88) = v57;
    sub_877D80(v57, v49);
    sub_877E20(v49, 0, v52);
LABEL_234:
    v9 = 0;
    sub_854980(v49, 0);
    *v3 = v49;
    v14 = *(_QWORD *)(v49 + 88);
    a1 = (__int64)v3;
    v109 = 0;
    v3[36] = *(_QWORD *)(v14 + 120);
    v58 = *((_BYTE *)v3 + 122) & 1;
    v97 = v58;
    sub_648B20(v3);
    v93 = v58;
    v84 = 0;
    goto LABEL_169;
  }
  v26 = *((_BYTE *)v3 + 125);
  if ( unk_4D047EC && (v26 & 0x40) == 0 && (unsigned __int8)(*((_BYTE *)v3 + 269) - 1) <= 1u )
  {
    if ( (unsigned int)sub_8D4070(v3[36]) )
    {
      sub_6851C0(893, a2 + 8);
      v26 = *((_BYTE *)v3 + 125);
    }
    else
    {
      if ( *((_BYTE *)v3 + 269) == 1 && (unsigned int)sub_8DD010(v3[36]) )
        sub_6851C0(892, a2 + 8);
      v26 = *((_BYTE *)v3 + 125);
    }
  }
  if ( v26 < 0 )
    goto LABEL_163;
  if ( v103 )
  {
    if ( dword_4F04C5C != dword_4F04C34 && *((_BYTE *)v3 + 269) == 1 )
    {
      v27 = 2442 - ((unsigned int)(*((_DWORD *)v3 + 64) == 0) - 1);
      sub_6851C0(v27, &dword_4F063F8);
      *((_BYTE *)v3 + 269) = 2;
      v3[36] = sub_72C930(v27);
    }
    goto LABEL_70;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( (*((_BYTE *)v3 + 130) & 0x20) != 0 )
      goto LABEL_70;
    if ( *((_BYTE *)v3 + 269) != 1 )
    {
      v40 = v3[36];
      if ( (*(_BYTE *)(v40 + 140) & 0xFB) == 8 && (v41 = sub_8D4C10(v40, 0), v40 = v3[36], (v41 & 1) != 0) )
        v42 = sub_8D5940(v40, 1, 0);
      else
        v42 = sub_8D5940(v40, 0, 1);
      if ( !v42 )
        goto LABEL_163;
LABEL_70:
      v28 = 2051;
LABEL_164:
      v9 = (__int64)v3;
      sub_6582F0((__m128i *)a2, (__int64)v3, v28, (int *)&v109, &v110, (__int64)a3);
      a1 = *v3;
      v14 = *(_QWORD *)(*v3 + 88LL);
      sub_86F690(*v3);
      v97 = 1;
      v84 = 0;
      v93 = 1;
      v3[36] = *(_QWORD *)(v14 + 120);
      *((_BYTE *)v3 + 269) = *(_BYTE *)(v14 + 136);
      *(_BYTE *)(v14 + 137) = *((_BYTE *)v3 + 268);
      goto LABEL_165;
    }
    goto LABEL_312;
  }
  v61 = *((_BYTE *)v3 + 269);
  if ( dword_4F04C5C )
  {
    if ( v61 != 1 )
    {
LABEL_163:
      v28 = 3;
      goto LABEL_164;
    }
LABEL_312:
    v84 = 0;
    v63 = 1;
    goto LABEL_249;
  }
  v62 = v61 & 0xFD;
  v84 = v62 == 0;
  v63 = (_BYTE)v62 == 0 ? 515LL : 1LL;
LABEL_249:
  v9 = (__int64)v3;
  sub_6582F0((__m128i *)a2, (__int64)v3, v63, (int *)&v109, &v110, (__int64)a3);
  a1 = *v3;
  v14 = *(_QWORD *)(*v3 + 88LL);
  sub_86F690(*v3);
  v97 = 0;
  v3[36] = *(_QWORD *)(v14 + 120);
  *((_BYTE *)v3 + 269) = *(_BYTE *)(v14 + 136);
  v93 = v84;
LABEL_165:
  if ( *((char *)v3 + 125) < 0 )
  {
    v60 = *v3;
    v13 = *(unsigned __int8 *)(*v3 + 80LL);
    *(_BYTE *)(*v3 + 81LL) |= 1u;
    if ( (_BYTE)v13 == 7 || (_BYTE)v13 == 9 )
      *(_BYTE *)(*(_QWORD *)(v60 + 88) + 169LL) |= 0x10u;
  }
  if ( (*((_BYTE *)v3 + 131) & 0x10) != 0 )
  {
    v43 = *(_BYTE *)(v14 + 156);
    *(_BYTE *)(v14 + 170) |= 2u;
    *(_QWORD *)(v14 + 128) = 0;
    if ( (v43 & 1) != 0 )
    {
      v78 = "__constant__";
      if ( (v43 & 4) == 0 )
      {
        v78 = "__managed__";
        if ( (*(_BYTE *)(v14 + 157) & 1) == 0 )
        {
          v78 = "__shared__";
          if ( (v43 & 2) == 0 )
            v78 = "__device__";
        }
      }
      v9 = a2 + 8;
      sub_6851A0(3578, a2 + 8, v78);
    }
    a1 = (__int64)v3;
    sub_659DF0(v3);
  }
LABEL_169:
  v44 = *((_BYTE *)v3 + 125);
  if ( (v44 & 2) != 0 )
  {
    *(_BYTE *)(v14 + 175) |= 2u;
  }
  else
  {
    v13 = v3[15] & 0x10010000000LL;
    if ( v13 == 0x10000000000LL )
    {
      *(_BYTE *)(v14 + 175) |= 1u;
    }
    else if ( (v44 & 4) != 0 )
    {
      *(_BYTE *)(v14 + 175) |= 4u;
    }
  }
  if ( (v3[1] & 2) != 0 )
  {
    if ( dword_4D04820 || dword_4F077BC && (a1 = (__int64)(v3 + 11), (unsigned int)sub_657F30((unsigned int *)v3 + 22)) )
    {
      v9 = v97;
      a1 = v14;
      sub_658080((_BYTE *)v14, v97);
    }
  }
  if ( v93 )
  {
    v12 = 1;
    if ( dword_4F077C4 != 2 )
      goto LABEL_13;
    v45 = v3[36];
    a1 = v45;
    if ( (unsigned int)sub_8D23B0(v45) )
    {
      a1 = v45;
      sub_8AE000(v45);
    }
  }
  else
  {
    v84 = 0;
    v97 = 0;
  }
  if ( dword_4F077C4 == 2 )
  {
    a1 = v3[36];
    v12 = 1;
    if ( (unsigned int)sub_8D5830(a1) )
    {
      v9 = 322;
      a1 = 8;
      sub_5EB950(8u, 322, v3[36], a2 + 8);
    }
  }
  else
  {
    v12 = 1;
  }
LABEL_13:
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  if ( v103 )
  {
    v104 = v96 & (v102 ^ 1);
    if ( v104 )
    {
      a1 = dword_4F077BC;
      if ( !dword_4F077BC || qword_4F077A8 > 0x76BFu || v3[37] )
      {
        v104 = 0;
      }
      else if ( (*(_BYTE *)(*v3 + 81LL) & 0x20) == 0 )
      {
        *(_BYTE *)(*v3 + 83LL) |= 0x40u;
      }
    }
    if ( !v105 && word_4F06418[0] == 56 )
      sub_7B8B50(a1, v9, v13, dword_4F07508);
    else
      *((_BYTE *)v3 + 127) |= 8u;
    if ( *(_BYTE *)(*v3 + 80LL) == 7
      && *((char *)v3 + 125) >= 0
      && dword_4F04C5C == dword_4F04C34
      && *(_BYTE *)(v14 + 136) == 1 )
    {
      *(_BYTE *)(v14 + 136) = 0;
    }
    sub_638AC0((__int64)v3, (_QWORD *)(a2 + 8), v109, v105, &v108, a3);
    v9 = *v3;
    if ( *(_BYTE *)(*v3 + 80LL) == 7 && (*(_BYTE *)(*(_QWORD *)(v9 + 88) + 156LL) & 2) != 0 )
    {
      v9 += 48;
      sub_686610(3510, v9, "__shared__", byte_3F871B3);
    }
    if ( v104 && (*(_BYTE *)(*v3 + 81LL) & 0x20) == 0 )
      *(_BYTE *)(*v3 + 83LL) &= ~0x40u;
    if ( v12 )
    {
      v9 = dword_4F077BC;
      if ( dword_4F077BC && (v96 & 1) != 0 && word_4F06418[0] == 142 )
      {
        v9 = (__int64)v3;
        sub_650EA0(v14, (__int64)v3);
      }
      if ( *(_BYTE *)(*v3 + 80LL) == 7 )
        sub_8756B0(*v3);
      v15 = v3[36];
      for ( j = *(_BYTE *)(v15 + 140); j == 12; j = *(_BYTE *)(v15 + 140) )
        v15 = *(_QWORD *)(v15 + 160);
      if ( j )
      {
        v3[36] = *(_QWORD *)(v14 + 120);
        if ( (*(_BYTE *)(v14 + 140) & 1) == 0 )
          goto LABEL_34;
        goto LABEL_118;
      }
      goto LABEL_33;
    }
LABEL_78:
    *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 8);
    sub_6522D0(v3);
    if ( dword_4F077C4 != 2 )
      goto LABEL_48;
LABEL_79:
    sub_65C470((__int64)v3, v9, v18, v19, v20);
    if ( v97 )
      goto LABEL_41;
LABEL_80:
    if ( !v12 )
      goto LABEL_48;
    goto LABEL_43;
  }
  if ( *((char *)v3 + 125) < 0 || (*((_BYTE *)v3 + 130) & 0x20) != 0 )
  {
LABEL_77:
    if ( v12 )
      goto LABEL_33;
    goto LABEL_78;
  }
  v34 = *v3;
  if ( !v97 || (*(_BYTE *)(a2 + 17) & 0x20) != 0 || *(_BYTE *)(v14 + 177) )
  {
    if ( *(_BYTE *)(v34 + 80) == 7 && (*((_BYTE *)v3 + 269) == 1 || v84) )
      sub_8756B0(v34);
    goto LABEL_77;
  }
  v35 = (*(_BYTE *)(v34 + 83) & 0x40) != 0;
  if ( dword_4F077BC )
    *(_BYTE *)(v34 + 83) |= 0x40u;
  if ( (*((_BYTE *)v3 + 10) & 0x20) != 0 )
    *(_BYTE *)(v14 + 172) |= 0x10u;
  v9 = a2 + 8;
  v36 = (unsigned int)sub_63BB10(*v3, a2 + 8);
  v3[36] = *(_QWORD *)(v14 + 120);
  if ( dword_4F077BC )
  {
    v9 = (unsigned __int8)v35 << 6;
    *(_BYTE *)(*v3 + 83LL) = (v35 << 6) | *(_BYTE *)(*v3 + 83LL) & 0xBF;
  }
  if ( (_DWORD)v36 )
  {
    sub_649FB0((__int64)v3, v9);
    v38 = *v3;
    if ( *(_BYTE *)(*v3 + 80LL) != 7 )
      goto LABEL_33;
    if ( *(_BYTE *)(v14 + 136) <= 2u )
    {
LABEL_141:
      sub_8756B0(v38);
      goto LABEL_33;
    }
    for ( k = *(_QWORD *)(v14 + 120); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
      ;
    if ( (unsigned int)sub_8D3410(k) )
    {
      for ( k = sub_8D40F0(k); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
    }
    if ( (unsigned __int8)(*(_BYTE *)(k + 140) - 9) <= 2u && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)k + 96LL) + 16LL) )
      goto LABEL_33;
LABEL_140:
    v38 = *v3;
    goto LABEL_141;
  }
  v37 = *(_BYTE *)(*v3 + 80LL);
  if ( v37 == 7 )
  {
    v36 = *((_BYTE *)v3 + 268) == 2;
  }
  else if ( v37 != 9 )
  {
    goto LABEL_33;
  }
  v9 = v3[36];
  sub_640330(*v3, v9, v36, 0);
  if ( *(_BYTE *)(*v3 + 80LL) == 7
    && ((*(_BYTE *)(v14 + 89) & 1) == 0 || *(_BYTE *)(v14 + 136) == 2 || (unsigned int)sub_8D2660(v3[36])) )
  {
    goto LABEL_140;
  }
LABEL_33:
  if ( (*(_BYTE *)(v14 + 140) & 1) == 0 )
    goto LABEL_34;
LABEL_118:
  v9 = 0;
  sub_72F9F0(v14, 0, &v107, v111);
  if ( v107 == 2
    && (!dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0x9F5Fu || (*(_BYTE *)(v14 + 89) & 1) == 0) )
  {
    v9 = (__int64)(v3 + 6);
    sub_6851C0(1418, v3 + 6);
  }
LABEL_34:
  if ( *(_QWORD *)(v14 + 104) )
  {
    v17 = 0;
    if ( (*((_BYTE *)v3 + 122) & 1) == 0 )
      v17 = (*(_BYTE *)(*v3 + 81LL) & 2) != 0;
    v9 = 7;
    sub_656C00((__int64)v3, 7, v14, v17, *((_BYTE *)v3 + 122) & 1);
  }
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 8);
  if ( (*(_BYTE *)(a2 + 17) & 0x20) == 0 )
  {
    v29 = *(_QWORD *)(v14 + 120);
    if ( (unsigned int)sub_8D23B0(v29) )
    {
      if ( !unk_4D041A8
        || (v29 = *(_QWORD *)(v14 + 120), (unsigned __int8)(*(_BYTE *)(v29 + 140) - 9) > 2u)
        || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
      {
        if ( v97 || dword_4F077C4 == 2 && (v29 = v3[36], (unsigned int)sub_8D2600(v29)) )
        {
LABEL_239:
          if ( !v108 )
          {
            v79 = *(_QWORD *)(v14 + 120);
            v9 = a2 + 8;
            v29 = (unsigned int)sub_67F240(v79);
            sub_685A50(v29, a2 + 8, v79, 8);
          }
          goto LABEL_241;
        }
        if ( v84 )
        {
          v29 = v3[36];
          if ( !(unsigned int)sub_8D2600(v29) )
          {
            if ( dword_4D04964 && *((_BYTE *)v3 + 269) == 2 && !v108 )
            {
              v64 = *(_QWORD *)(v14 + 120);
              v106 = unk_4F07471;
              v65 = sub_67F240(v64);
              v9 = a2 + 8;
              sub_685A50(v65, a2 + 8, v64, v106);
            }
            goto LABEL_96;
          }
          goto LABEL_239;
        }
LABEL_96:
        if ( dword_4D0488C
          || word_4D04898
          && (v9 = (unsigned int)qword_4F077B4, (_DWORD)qword_4F077B4)
          && qword_4F077A0 > 0x765Bu
          && (unsigned int)sub_729F80(dword_4F063F8) )
        {
          if ( unk_4F04C50
            && (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 193LL) & 2) != 0
            && (*((_BYTE *)v3 + 130) & 0x20) == 0 )
          {
            v9 = a2 + 8;
            sub_646FB0(v14, a2 + 8);
          }
        }
        if ( (*(_BYTE *)(a2 + 17) & 0x20) == 0
          && (*(_BYTE *)(v14 + 170) & 0x60) == 0
          && *(_BYTE *)(v14 + 177) != 5
          && (unk_4F04C50 && (v30 = *(_QWORD *)(unk_4F04C50 + 32LL)) != 0 && (*(_BYTE *)(v30 + 198) & 0x10) != 0
           || (*(_BYTE *)(v14 + 156) & 1) != 0) )
        {
          if ( v93 )
          {
            v31 = *(_QWORD *)(v14 + 120);
            if ( v31 )
            {
              while ( *(_BYTE *)(v31 + 140) == 12 )
                v31 = *(_QWORD *)(v31 + 160);
              if ( !(unsigned int)sub_8D23B0(v31) )
              {
                v32 = *(_QWORD *)(v14 + 8);
                if ( v32 )
                {
                  for ( m = *(_QWORD *)(v14 + 120); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                    ;
                  if ( !*(_QWORD *)(m + 128) )
                  {
                    v9 = a2 + 8;
                    sub_6851A0(3668, a2 + 8, v32);
                  }
                }
              }
            }
          }
        }
        goto LABEL_39;
      }
    }
    else
    {
      v29 = *(_QWORD *)(v14 + 120);
    }
    if ( (unsigned int)sub_8D2BE0(v29) && ((*(_BYTE *)(v14 + 136) - 3) & 0xFD) != 0 )
    {
      v9 = a2 + 8;
      v29 = 3414;
      sub_685360(3414, a2 + 8);
LABEL_241:
      *(_QWORD *)(v14 + 120) = sub_72C930(v29);
      goto LABEL_96;
    }
    goto LABEL_96;
  }
LABEL_39:
  sub_6522D0(v3);
  if ( dword_4F077C4 == 2 )
    goto LABEL_79;
  if ( !v97 )
    goto LABEL_80;
LABEL_41:
  if ( !v12 )
    goto LABEL_48;
  v9 = 7;
  sub_737310(v14, 7);
LABEL_43:
  v21 = sub_825FC0(*(_QWORD *)(v14 + 40));
  if ( (*(_WORD *)(v14 + 156) & 0x101) == 0x101 )
  {
    if ( (*(_BYTE *)(v14 + 156) & 6) != 0 )
    {
      v9 = a2 + 8;
      sub_6851C0(3568, a2 + 8);
    }
    v46 = *(_QWORD *)(v14 + 120);
    if ( v46 )
    {
      if ( (*(_BYTE *)(v46 + 140) & 0xFB) == 8 )
      {
        v9 = dword_4F077C4 != 2;
        if ( (sub_8D4C10(v46, v9) & 1) != 0 )
        {
          v9 = a2 + 8;
          sub_6851C0(3566, a2 + 8);
        }
        v46 = *(_QWORD *)(v14 + 120);
      }
      if ( (unsigned int)sub_8D2FB0(v46) )
      {
        v9 = a2 + 8;
        sub_6851C0(3567, a2 + 8);
      }
    }
  }
  if ( v21 )
  {
    v22 = *(_BYTE *)(v14 + 156);
    if ( (v22 & 1) != 0 )
    {
      v39 = "__constant__";
      if ( (v22 & 4) == 0 )
      {
        v39 = "__managed__";
        if ( (*(_BYTE *)(v14 + 157) & 1) == 0 )
        {
          v39 = "__shared__";
          if ( (v22 & 2) == 0 )
            v39 = "__device__";
        }
      }
      v9 = a2 + 8;
      sub_6851A0(3579, a2 + 8, v39);
    }
    if ( (unsigned int)sub_8D2FF0(*(_QWORD *)(v14 + 120), v9) || (unsigned int)sub_8D3030(*(_QWORD *)(v14 + 120)) )
      sub_6851C0(3580, a2 + 8);
  }
LABEL_48:
  result = *((_WORD *)v3 + 65) & 0x1020;
  if ( result == 4096 )
    return sub_6570B0((__int64)v3);
  return result;
}
