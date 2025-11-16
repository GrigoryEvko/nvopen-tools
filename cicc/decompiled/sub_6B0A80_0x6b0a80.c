// Function: sub_6B0A80
// Address: 0x6b0a80
//
__int64 __fastcall sub_6B0A80(__int64 a1, __int64 *a2, int a3, __int64 a4, __m128i *a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 v7; // r12
  __m128i *v8; // rbx
  _BOOL4 v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  char v14; // cl
  char v15; // dl
  __int64 j; // r8
  unsigned int *v17; // rdx
  __int8 v18; // al
  char v19; // al
  __int64 v20; // rdi
  char v21; // dl
  __m128i v22; // xmm1
  __m128i v23; // xmm2
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  __m128i v26; // xmm5
  __m128i v27; // xmm6
  __m128i v28; // xmm7
  __m128i v29; // xmm0
  char v30; // dl
  __int64 v31; // rax
  char v32; // al
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v37; // rax
  char i; // dl
  __int32 v39; // edx
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 k; // r15
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 m; // rdx
  __int64 v47; // rdi
  int v48; // eax
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  unsigned int *n; // rdx
  char v53; // al
  __int64 v54; // rax
  __int64 v55; // r15
  __m128i v56; // xmm1
  __m128i v57; // xmm2
  __m128i v58; // xmm3
  __m128i v59; // xmm7
  __m128i v60; // xmm4
  __m128i v61; // xmm5
  __m128i v62; // xmm6
  __int8 v63; // al
  __m128i v64; // xmm7
  __int8 v65; // al
  char v66; // al
  __m128i v67; // xmm2
  __m128i v68; // xmm3
  __m128i v69; // xmm4
  __m128i v70; // xmm5
  __m128i v71; // xmm6
  __m128i v72; // xmm7
  __m128i v73; // xmm1
  __m128i v74; // xmm2
  __m128i v75; // xmm3
  __m128i v76; // xmm4
  __m128i v77; // xmm5
  __m128i v78; // xmm6
  __int64 v79; // rax
  bool v80; // cf
  __int64 v81; // r14
  __m128i v82; // xmm4
  __m128i v83; // xmm5
  __m128i v84; // xmm6
  __m128i v85; // xmm3
  __m128i v86; // xmm7
  __m128i v87; // xmm4
  __m128i v88; // xmm5
  __m128i v89; // xmm6
  __m128i v90; // xmm3
  __m128i v91; // xmm4
  __m128i v92; // xmm5
  __m128i v93; // xmm6
  int v94; // eax
  __int64 v95; // [rsp-10h] [rbp-350h]
  __int64 v96; // [rsp-8h] [rbp-348h]
  __int64 v97; // [rsp+0h] [rbp-340h]
  __int64 v98; // [rsp+8h] [rbp-338h]
  __int64 v99; // [rsp+10h] [rbp-330h]
  __int64 v100; // [rsp+10h] [rbp-330h]
  unsigned int *v101; // [rsp+10h] [rbp-330h]
  int v102; // [rsp+10h] [rbp-330h]
  int v103; // [rsp+18h] [rbp-328h]
  __int64 v104; // [rsp+18h] [rbp-328h]
  __int64 v105; // [rsp+18h] [rbp-328h]
  _BOOL4 v106; // [rsp+20h] [rbp-320h]
  __int64 v107; // [rsp+20h] [rbp-320h]
  __int64 v109; // [rsp+28h] [rbp-318h]
  _BOOL4 v110; // [rsp+3Ch] [rbp-304h] BYREF
  unsigned int v111; // [rsp+40h] [rbp-300h] BYREF
  unsigned int v112; // [rsp+44h] [rbp-2FCh] BYREF
  __int64 v113; // [rsp+48h] [rbp-2F8h] BYREF
  _BYTE v114[352]; // [rsp+50h] [rbp-2F0h] BYREF
  __m128i v115; // [rsp+1B0h] [rbp-190h] BYREF
  __m128i v116; // [rsp+1C0h] [rbp-180h] BYREF
  __m128i v117; // [rsp+1D0h] [rbp-170h] BYREF
  __m128i v118; // [rsp+1E0h] [rbp-160h] BYREF
  __m256i v119; // [rsp+1F0h] [rbp-150h] BYREF
  __m128i v120; // [rsp+210h] [rbp-130h] BYREF
  __m128i v121; // [rsp+220h] [rbp-120h] BYREF
  __m128i v122; // [rsp+230h] [rbp-110h] BYREF
  __m128i v123; // [rsp+240h] [rbp-100h] BYREF
  __m128i v124; // [rsp+250h] [rbp-F0h] BYREF
  __m128i v125; // [rsp+260h] [rbp-E0h] BYREF
  __m128i v126; // [rsp+270h] [rbp-D0h] BYREF
  __m128i v127; // [rsp+280h] [rbp-C0h] BYREF
  __m128i v128; // [rsp+290h] [rbp-B0h] BYREF
  __m128i v129; // [rsp+2A0h] [rbp-A0h] BYREF
  __m128i v130; // [rsp+2B0h] [rbp-90h] BYREF
  __m128i v131; // [rsp+2C0h] [rbp-80h] BYREF
  __m128i v132; // [rsp+2D0h] [rbp-70h] BYREF
  __m128i v133; // [rsp+2E0h] [rbp-60h] BYREF
  __m128i v134; // [rsp+2F0h] [rbp-50h] BYREF
  __m128i v135[4]; // [rsp+300h] [rbp-40h] BYREF

  v6 = a2;
  v7 = a4;
  v111 = 0;
  if ( !a2 )
  {
    v8 = (__m128i *)a1;
    v113 = *(_QWORD *)&dword_4F063F8;
    v112 = dword_4F06650[0];
    v110 = word_4F06418[0] == 148;
    v10 = qword_4D03C50;
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
    {
      sub_7B8B50(a1, 0, qword_4D03C50, a4);
      sub_69ED20((__int64)&v115, 0, 16, 0);
      v9 = v110;
      goto LABEL_5;
    }
LABEL_45:
    v32 = *(_BYTE *)(v10 + 16);
    if ( v32 )
    {
      if ( v32 == 1 )
      {
        if ( (unsigned int)sub_6E5430(a1, a2, v10, a4, a5, a6) )
        {
          a2 = &v113;
          a1 = 60;
          sub_6851C0(0x3Cu, &v113);
        }
      }
      else if ( v32 == 2 )
      {
        if ( (unsigned int)sub_6E5430(a1, a2, v10, a4, a5, a6) )
        {
          a2 = &v113;
          a1 = 529;
          sub_6851C0(0x211u, &v113);
        }
      }
      else if ( (unsigned int)sub_6E5430(a1, a2, v10, a4, a5, a6) )
      {
        a2 = &v113;
        a1 = 57;
        sub_6851C0(0x39u, &v113);
      }
    }
    else if ( (unsigned int)sub_6E5430(a1, a2, v10, a4, a5, a6) )
    {
      a2 = &v113;
      a1 = 58;
      sub_6851C0(0x3Au, &v113);
    }
    if ( !v6 )
    {
      sub_7B8B50(a1, a2, v33, v34);
      sub_69ED20((__int64)&v115, 0, 16, 0);
    }
    goto LABEL_52;
  }
  v106 = *((_WORD *)a2 + 4) == 148;
  if ( a3 )
  {
    v8 = (__m128i *)v114;
    a2 = 0;
    a1 = (__int64)v6;
    sub_6F8AB0((_DWORD)v6, 0, (unsigned int)v114, 0, (unsigned int)&v113, (unsigned int)&v112, 0);
    v21 = *(_BYTE *)(v7 + 16);
    v22 = _mm_loadu_si128((const __m128i *)(v7 + 16));
    v23 = _mm_loadu_si128((const __m128i *)(v7 + 32));
    v24 = _mm_loadu_si128((const __m128i *)(v7 + 48));
    v115 = _mm_loadu_si128((const __m128i *)v7);
    v25 = _mm_loadu_si128((const __m128i *)(v7 + 64));
    v9 = v106;
    v116 = v22;
    v26 = _mm_loadu_si128((const __m128i *)(v7 + 80));
    a4 = v96;
    v117 = v23;
    v27 = _mm_loadu_si128((const __m128i *)(v7 + 96));
    v28 = _mm_loadu_si128((const __m128i *)(v7 + 112));
    v118 = v24;
    v29 = _mm_loadu_si128((const __m128i *)(v7 + 128));
    *(__m128i *)v119.m256i_i8 = v25;
    *(__m128i *)&v119.m256i_u64[2] = v26;
    v120 = v27;
    v121 = v28;
    v122 = v29;
    if ( v21 == 2 )
    {
      v67 = _mm_loadu_si128((const __m128i *)(v7 + 160));
      v68 = _mm_loadu_si128((const __m128i *)(v7 + 176));
      v69 = _mm_loadu_si128((const __m128i *)(v7 + 192));
      v70 = _mm_loadu_si128((const __m128i *)(v7 + 208));
      v123 = _mm_loadu_si128((const __m128i *)(v7 + 144));
      v71 = _mm_loadu_si128((const __m128i *)(v7 + 224));
      v72 = _mm_loadu_si128((const __m128i *)(v7 + 240));
      v124 = v67;
      v73 = _mm_loadu_si128((const __m128i *)(v7 + 256));
      v74 = _mm_loadu_si128((const __m128i *)(v7 + 272));
      v125 = v68;
      v75 = _mm_loadu_si128((const __m128i *)(v7 + 288));
      v126 = v69;
      v76 = _mm_loadu_si128((const __m128i *)(v7 + 304));
      v127 = v70;
      v77 = _mm_loadu_si128((const __m128i *)(v7 + 320));
      v128 = v71;
      v78 = _mm_loadu_si128((const __m128i *)(v7 + 336));
      v129 = v72;
      v130 = v73;
      v131 = v74;
      v132 = v75;
      v133 = v76;
      v134 = v77;
      v135[0] = v78;
    }
    else if ( v21 == 5 || v21 == 1 )
    {
      v123.m128i_i64[0] = *(_QWORD *)(v7 + 144);
    }
  }
  else
  {
    v8 = (__m128i *)v114;
    a1 = (__int64)a2;
    a2 = (__int64 *)v114;
    sub_6F8AB0(a1, (unsigned int)v114, (unsigned int)&v115, 0, (unsigned int)&v113, (unsigned int)&v112, 0);
    v9 = v106;
  }
  v110 = v9;
  v10 = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
    goto LABEL_45;
LABEL_5:
  if ( v9 )
    goto LABEL_31;
  if ( dword_4F04C44 == -1 )
  {
    v11 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v11 + 6) & 6) == 0 && *(_BYTE *)(v11 + 4) != 12 )
      goto LABEL_9;
  }
  if ( (unsigned int)sub_8DBE70(v8->m128i_i64[0]) || (unsigned int)sub_8DBE70(v115.m128i_i64[0]) )
  {
    if ( (unsigned int)sub_8D3A70(v8->m128i_i64[0]) || (unsigned int)sub_8DD3B0(v8->m128i_i64[0]) )
    {
      if ( (unsigned int)sub_8D3D40(v115.m128i_i64[0]) || (unsigned int)sub_8D3D10(v115.m128i_i64[0]) )
      {
        sub_7038B0(96, (_DWORD)v8, (unsigned int)&v115, v7, (unsigned int)&v113, v112, 0);
LABEL_67:
        v111 = 1;
        if ( (*(_BYTE *)(v7 + 18) & 1) != 0 )
          goto LABEL_54;
LABEL_68:
        v39 = v8[4].m128i_i32[1];
        *(_WORD *)(v7 + 72) = v8[4].m128i_i16[4];
        *(_DWORD *)(v7 + 68) = v39;
        *(_QWORD *)dword_4F07508 = *(_QWORD *)(v7 + 68);
        v40 = *(__int64 *)((char *)&v119.m256i_i64[1] + 4);
        *(_QWORD *)(v7 + 76) = *(__int64 *)((char *)&v119.m256i_i64[1] + 4);
        unk_4F061D8 = v40;
        sub_6E3280(v7, &v113);
        goto LABEL_55;
      }
      v37 = v115.m128i_i64[0];
      for ( i = *(_BYTE *)(v115.m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v37 + 140) )
        v37 = *(_QWORD *)(v37 + 160);
      if ( i )
        sub_6E68E0(380, &v115);
    }
    else
    {
      v30 = *(_BYTE *)(v8->m128i_i64[0] + 140);
      if ( v30 == 12 )
      {
        v31 = v8->m128i_i64[0];
        do
        {
          v31 = *(_QWORD *)(v31 + 160);
          v30 = *(_BYTE *)(v31 + 140);
        }
        while ( v30 == 12 );
      }
      if ( v30 )
        sub_6E6930(153, v8, v8->m128i_i64[0]);
    }
    sub_6E6260(v7);
    goto LABEL_67;
  }
  if ( !v110 )
  {
LABEL_9:
    v12 = 4;
LABEL_10:
    sub_6F69D0(v8, v12);
    if ( !v8[1].m128i_i8[0] )
      goto LABEL_71;
    v13 = v8->m128i_i64[0];
    v14 = *(_BYTE *)(v8->m128i_i64[0] + 140);
    v107 = v8->m128i_i64[0];
    if ( v14 == 12 )
    {
      do
      {
        v13 = *(_QWORD *)(v13 + 160);
        v15 = *(_BYTE *)(v13 + 140);
      }
      while ( v15 == 12 );
    }
    else
    {
      v15 = *(_BYTE *)(v8->m128i_i64[0] + 140);
    }
    if ( !v15 )
    {
LABEL_71:
      sub_6E6870(v8);
      j = 0;
      v107 = 0;
      v103 = 1;
      goto LABEL_72;
    }
    v103 = v110;
    if ( v110 )
    {
      if ( !(unsigned int)sub_6FB4D0(v8, 3164) )
      {
        v107 = 0;
        j = 0;
        v103 = 1;
        goto LABEL_72;
      }
      v107 = sub_8D46C0(v8->m128i_i64[0]);
      j = v107;
      v19 = *(_BYTE *)(v107 + 140);
      if ( v19 == 12 )
      {
        do
        {
          j = *(_QWORD *)(j + 160);
          v19 = *(_BYTE *)(j + 140);
        }
        while ( v19 == 12 );
      }
      else
      {
        j = v107;
      }
    }
    else
    {
      for ( j = v8->m128i_i64[0]; v14 == 12; v14 = *(_BYTE *)(j + 140) )
        j = *(_QWORD *)(j + 160);
      v17 = &dword_4F077C4;
      v18 = v8[1].m128i_i8[1];
      if ( dword_4F077C4 == 2 )
      {
        if ( unk_4F07778 > 201102 || (v17 = &dword_4F07774, dword_4F07774) )
        {
          v17 = &dword_4F077BC;
          v12 = dword_4F077BC;
          if ( !dword_4F077BC
            || (v17 = (unsigned int *)(unsigned int)qword_4F077B4, (_DWORD)qword_4F077B4)
            || (v17 = (unsigned int *)&qword_4F077A8, qword_4F077A8 > 0x1869Fu) )
          {
            if ( v18 == 2 )
            {
              if ( (unsigned __int8)(v14 - 9) > 2u )
              {
                if ( (unsigned __int8)(*(_BYTE *)(j + 140) - 9) > 2u )
                {
                  v20 = 153;
LABEL_23:
                  v99 = j;
                  sub_6E6930(v20, v8, v8->m128i_i64[0]);
                  j = v99;
                  v103 = 1;
                }
LABEL_72:
                v41 = 0;
                v100 = j;
                sub_6F69D0(&v115, 0);
                for ( k = v115.m128i_i64[0]; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                  ;
                if ( !(unsigned int)sub_8D3D10(k) )
                {
                  while ( 1 )
                  {
                    v66 = *(_BYTE *)(k + 140);
                    if ( v66 != 12 )
                      break;
                    k = *(_QWORD *)(k + 160);
                  }
                  if ( v66 )
                    sub_6E68E0(380, &v115);
                  goto LABEL_52;
                }
                if ( !v103 )
                {
                  v43 = sub_8D4890(k);
                  v44 = v43;
                  if ( v100 == v43
                    || v100
                    && v43
                    && dword_4F07588
                    && (v45 = *(_QWORD *)(v100 + 32), *(_QWORD *)(v44 + 32) == v45)
                    && v45 )
                  {
                    v98 = 0;
LABEL_83:
                    for ( m = sub_8D4870(k); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                      ;
                    v47 = m;
                    v101 = (unsigned int *)m;
                    v48 = sub_8D2310(m);
                    if ( !v48 )
                    {
                      n = &dword_4D04410;
                      if ( dword_4D04410 )
                      {
                        n = (unsigned int *)v110;
                        if ( !v110 )
                        {
                          v48 = 1;
                          if ( v8[1].m128i_i8[1] == 1 )
                            v48 = sub_6ED0A0(v8) != 0;
                        }
                        v103 = v48;
                        v102 = 1;
                        goto LABEL_98;
                      }
                      if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
                      {
                        if ( v110 || v8[1].m128i_i8[1] == 1 && !(unsigned int)sub_6ED0A0(v8) )
                          goto LABEL_131;
LABEL_138:
                        if ( (unsigned int)sub_6ED0A0(v8) )
                        {
                          v103 = 1;
LABEL_131:
                          v102 = 1;
                          goto LABEL_98;
                        }
LABEL_154:
                        v102 = 0;
                        goto LABEL_98;
                      }
                      goto LABEL_90;
                    }
                    n = v101;
                    v53 = *(_BYTE *)(*((_QWORD *)v101 + 21) + 19LL) >> 6;
                    if ( v53 != 2 )
                    {
                      if ( v53 != 1 )
                        goto LABEL_88;
                      if ( !v110 )
                      {
                        if ( v8[1].m128i_i8[1] == 2 || (v47 = (__int64)v8, v94 = sub_6ED0A0(v8), n = v101, v94) )
                        {
                          v51 = dword_4D041BC;
                          if ( !dword_4D041BC || (*(_BYTE *)(*((_QWORD *)n + 21) + 18LL) & 0x7F) != 1 )
                          {
                            if ( (unsigned int)sub_6E5430(v47, v41, n, v49, v50, dword_4D041BC) )
                              sub_685360(0x996u, &v113, k);
                            goto LABEL_52;
                          }
                        }
LABEL_88:
                        v49 = dword_4D04410;
                        if ( dword_4D04410 )
                          goto LABEL_154;
                        if ( dword_4F077C4 != 2 )
                          goto LABEL_90;
                        goto LABEL_152;
                      }
                      if ( dword_4D04410 )
                        goto LABEL_154;
                      if ( dword_4F077C4 != 2 )
                        goto LABEL_131;
LABEL_152:
                      if ( unk_4F07778 > 201102 )
                        goto LABEL_154;
                      n = (unsigned int *)dword_4F07774;
                      if ( dword_4F07774 )
                        goto LABEL_154;
LABEL_90:
                      if ( v110 || v8[1].m128i_i8[1] == 1 && !(unsigned int)sub_6ED0A0(v8) )
                        goto LABEL_131;
LABEL_92:
                      if ( !v8[1].m128i_i8[0] )
                        goto LABEL_131;
                      v54 = v8->m128i_i64[0];
                      for ( n = (unsigned int *)*(unsigned __int8 *)(v8->m128i_i64[0] + 140);
                            (_BYTE)n == 12;
                            n = (unsigned int *)*(unsigned __int8 *)(v54 + 140) )
                      {
                        v54 = *(_QWORD *)(v54 + 160);
                      }
                      if ( !(_BYTE)n )
                        goto LABEL_131;
                      if ( qword_4D0495C )
                      {
                        sub_6FFCF0(v8, &v110);
                        v102 = 1;
LABEL_98:
                        if ( v98 )
                        {
                          sub_6F7270((_DWORD)v8, v98, 0, 1, 0, 1, 0, 1);
                          v51 = v95;
                        }
                        v55 = sub_73CAD0(v107, k, n, v49, v50, v51);
                        if ( (unsigned int)sub_8D2310(v55) )
                        {
                          v56 = _mm_loadu_si128(&v117);
                          v57 = _mm_loadu_si128(&v118);
                          v58 = _mm_loadu_si128((const __m128i *)&v119);
                          *(__m128i *)v7 = _mm_loadu_si128(&v115);
                          v59 = _mm_loadu_si128(&v116);
                          v60 = _mm_loadu_si128((const __m128i *)&v119.m256i_u64[2]);
                          v61 = _mm_loadu_si128(&v120);
                          v62 = _mm_loadu_si128(&v121);
                          *(__m128i *)(v7 + 32) = v56;
                          *(__m128i *)(v7 + 16) = v59;
                          v63 = v116.m128i_i8[0];
                          v64 = _mm_loadu_si128(&v122);
                          *(__m128i *)(v7 + 48) = v57;
                          *(__m128i *)(v7 + 64) = v58;
                          *(__m128i *)(v7 + 80) = v60;
                          *(__m128i *)(v7 + 96) = v61;
                          *(__m128i *)(v7 + 112) = v62;
                          *(__m128i *)(v7 + 128) = v64;
                          if ( v63 == 2 )
                          {
                            v82 = _mm_loadu_si128(&v124);
                            v83 = _mm_loadu_si128(&v125);
                            v84 = _mm_loadu_si128(&v126);
                            *(__m128i *)(v7 + 144) = _mm_loadu_si128(&v123);
                            v85 = _mm_loadu_si128(&v127);
                            v86 = _mm_loadu_si128(&v131);
                            *(__m128i *)(v7 + 160) = v82;
                            v87 = _mm_loadu_si128(&v128);
                            *(__m128i *)(v7 + 176) = v83;
                            v88 = _mm_loadu_si128(&v129);
                            *(__m128i *)(v7 + 192) = v84;
                            v89 = _mm_loadu_si128(&v130);
                            *(__m128i *)(v7 + 208) = v85;
                            v90 = _mm_loadu_si128(&v132);
                            *(__m128i *)(v7 + 224) = v87;
                            v91 = _mm_loadu_si128(&v133);
                            *(__m128i *)(v7 + 240) = v88;
                            v92 = _mm_loadu_si128(&v134);
                            *(__m128i *)(v7 + 256) = v89;
                            v93 = _mm_loadu_si128(v135);
                            *(__m128i *)(v7 + 272) = v86;
                            *(__m128i *)(v7 + 288) = v90;
                            *(__m128i *)(v7 + 304) = v91;
                            *(__m128i *)(v7 + 320) = v92;
                            *(__m128i *)(v7 + 336) = v93;
                          }
                          else if ( v63 == 5 || v63 == 1 )
                          {
                            *(_QWORD *)(v7 + 144) = v123.m128i_i64[0];
                          }
                          *(_QWORD *)(v7 + 88) = 0;
                          *a5 = _mm_loadu_si128(v8);
                          a5[1] = _mm_loadu_si128(v8 + 1);
                          a5[2] = _mm_loadu_si128(v8 + 2);
                          a5[3] = _mm_loadu_si128(v8 + 3);
                          a5[4] = _mm_loadu_si128(v8 + 4);
                          a5[5] = _mm_loadu_si128(v8 + 5);
                          a5[6] = _mm_loadu_si128(v8 + 6);
                          a5[7] = _mm_loadu_si128(v8 + 7);
                          a5[8] = _mm_loadu_si128(v8 + 8);
                          v65 = v8[1].m128i_i8[0];
                          if ( v65 == 2 )
                          {
                            a5[9] = _mm_loadu_si128(v8 + 9);
                            a5[10] = _mm_loadu_si128(v8 + 10);
                            a5[11] = _mm_loadu_si128(v8 + 11);
                            a5[12] = _mm_loadu_si128(v8 + 12);
                            a5[13] = _mm_loadu_si128(v8 + 13);
                            a5[14] = _mm_loadu_si128(v8 + 14);
                            a5[15] = _mm_loadu_si128(v8 + 15);
                            a5[16] = _mm_loadu_si128(v8 + 16);
                            a5[17] = _mm_loadu_si128(v8 + 17);
                            a5[18] = _mm_loadu_si128(v8 + 18);
                            a5[19] = _mm_loadu_si128(v8 + 19);
                            a5[20] = _mm_loadu_si128(v8 + 20);
                            a5[21] = _mm_loadu_si128(v8 + 21);
                          }
                          else if ( v65 == 5 || v65 == 1 )
                          {
                            a5[9].m128i_i64[0] = v8[9].m128i_i64[0];
                          }
                          sub_82F1E0(a5, v110, v7);
                        }
                        else
                        {
                          v109 = sub_6F6F40(v8, 0);
                          v79 = sub_6F6F40(&v115, 0);
                          v80 = !v110;
                          *(_QWORD *)(v109 + 16) = v79;
                          v81 = sub_73DBF0(96 - ((unsigned int)v80 - 1), v55, v109);
                          if ( unk_4D04810 )
                            *(_BYTE *)(v81 + 60) |= 1u;
                          sub_6E70E0(v81, v7);
                          if ( v102 )
                          {
                            *(_BYTE *)(v81 + 25) |= 1u;
                            sub_6E6A20(v7);
                            *(_QWORD *)(v7 + 88) = v8[5].m128i_i64[1];
                            if ( v103 )
                              sub_6ED1A0(v7);
                          }
                          else
                          {
                            if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v55) )
                              sub_8AE000(v55);
                            if ( (unsigned int)sub_8D23B0(v55) )
                            {
                              sub_6E5F60(&v113, v55, 8);
                              sub_6E6840(v7);
                            }
                          }
                        }
                        goto LABEL_53;
                      }
                      goto LABEL_138;
                    }
                    if ( !v110 )
                    {
                      if ( v8[1].m128i_i8[1] != 1 )
                      {
                        v50 = dword_4D04410;
                        if ( dword_4D04410 )
                          goto LABEL_154;
                        if ( dword_4F077C4 != 2 )
                          goto LABEL_92;
                        goto LABEL_152;
                      }
                      v47 = (__int64)v8;
                      if ( (unsigned int)sub_6ED0A0(v8) )
                        goto LABEL_88;
                    }
                    if ( (unsigned int)sub_6E5430(v47, v41, n, v49, v50, v51) )
                      sub_685360(0x997u, &v113, k);
                    goto LABEL_52;
                  }
                  v41 = v44;
                  v97 = v44;
                  v98 = sub_8D5CE0(v100, v44);
                  if ( v98 )
                    goto LABEL_83;
                  sub_6E5ED0(521, &v113, v100, v97);
                }
LABEL_52:
                sub_6E6260(v7);
                sub_6E6450(v8);
                sub_6E6450(&v115);
                goto LABEL_53;
              }
              v12 = 1;
              v105 = j;
              sub_6F9770(v8, 1);
              v18 = v8[1].m128i_i8[1];
              j = v105;
            }
          }
        }
      }
      if ( v18 == 1 )
      {
        v104 = j;
        sub_6ECC10(v8, v12, v17);
        j = v104;
      }
      v19 = *(_BYTE *)(j + 140);
    }
    v103 = 0;
    if ( (unsigned __int8)(v19 - 9) <= 2u )
      goto LABEL_72;
    v20 = !v110 ? 153 : 131;
    goto LABEL_23;
  }
LABEL_31:
  if ( (unsigned int)sub_68FE10(v8, 0, 1) || (unsigned int)sub_68FE10(&v115, 0, 1) )
    sub_84EC30(40, 0, 0, 1, 0, (_DWORD)v8, (__int64)&v115, (__int64)&v113, v112, 0, 0, v7, 0, 0, (__int64)&v111);
  v12 = v111;
  if ( !v111 )
  {
    if ( v110 )
      goto LABEL_10;
    goto LABEL_9;
  }
LABEL_53:
  if ( (*(_BYTE *)(v7 + 18) & 1) == 0 )
    goto LABEL_68;
LABEL_54:
  *(_DWORD *)(v7 + 68) = v119.m256i_i32[1];
  *(_WORD *)(v7 + 72) = v119.m256i_i16[4];
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v7 + 68);
  v35 = *(__int64 *)((char *)&v119.m256i_i64[1] + 4);
  *(_QWORD *)(v7 + 76) = *(__int64 *)((char *)&v119.m256i_i64[1] + 4);
  unk_4F061D8 = v35;
  sub_6E3280(v7, 0);
LABEL_55:
  sub_6E3BA0(v7, &v113, v112, 0);
  return sub_6E26D0(2, v7);
}
