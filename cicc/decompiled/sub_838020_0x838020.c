// Function: sub_838020
// Address: 0x838020
//
__int64 __fastcall sub_838020(__int64 a1, __int64 a2, __m128i *a3, __int64 a4, int a5, int a6, __m128i *a7)
{
  __m128i *v8; // r14
  int v9; // r13d
  __int64 v10; // r12
  const __m128i *i; // r10
  int v12; // r11d
  __int64 result; // rax
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // eax
  int v19; // eax
  char v20; // al
  __int64 j; // rdi
  __int8 v22; // cl
  const __m128i *k; // rsi
  unsigned __int8 v24; // cl
  char v25; // al
  int v26; // eax
  int v27; // eax
  int v28; // eax
  int v29; // eax
  int v30; // eax
  _BOOL4 v31; // eax
  const __m128i *v32; // rdi
  int v33; // eax
  __int64 v34; // r8
  __int64 v35; // r9
  _BOOL4 v36; // eax
  __int32 v37; // eax
  int v38; // eax
  __int64 v39; // r12
  bool v40; // r11
  int v41; // eax
  int v42; // eax
  _QWORD *v43; // rax
  bool v44; // zf
  int v45; // eax
  int v46; // eax
  __int64 v47; // rax
  __m128i v48; // xmm1
  int v49; // eax
  int v50; // eax
  int v51; // eax
  _BOOL4 v52; // eax
  int v53; // eax
  __int64 v54; // rax
  __int32 v55; // eax
  _BOOL4 v56; // eax
  __int64 v57; // rax
  int v58; // eax
  int v59; // eax
  _BOOL4 v60; // eax
  _BOOL4 v61; // eax
  int v62; // eax
  __int8 v63; // al
  _BOOL4 v64; // eax
  __int64 v65; // r14
  __int64 v66; // rax
  __int64 v67; // rax
  __m128i v68; // xmm3
  int v69; // r8d
  int v70; // eax
  __int64 v71; // rsi
  int v72; // eax
  __int32 v73; // eax
  int v74; // eax
  __m128i v75; // xmm4
  __int64 v76; // rdx
  int v77; // eax
  __int64 v78; // rcx
  int *m128i_i32; // rdi
  unsigned __int8 v80; // [rsp+8h] [rbp-278h]
  const __m128i *v81; // [rsp+8h] [rbp-278h]
  __int64 v82; // [rsp+8h] [rbp-278h]
  int v83; // [rsp+14h] [rbp-26Ch]
  __int64 m128i_i64; // [rsp+18h] [rbp-268h]
  const __m128i *v85; // [rsp+20h] [rbp-260h]
  const __m128i *v86; // [rsp+20h] [rbp-260h]
  const __m128i *v87; // [rsp+20h] [rbp-260h]
  const __m128i *v88; // [rsp+20h] [rbp-260h]
  _BOOL4 v89; // [rsp+28h] [rbp-258h]
  const __m128i *v90; // [rsp+28h] [rbp-258h]
  int v91; // [rsp+30h] [rbp-250h]
  int v92; // [rsp+30h] [rbp-250h]
  const __m128i *v93; // [rsp+30h] [rbp-250h]
  unsigned int v94; // [rsp+30h] [rbp-250h]
  const __m128i *v95; // [rsp+30h] [rbp-250h]
  int v96; // [rsp+38h] [rbp-248h]
  int v97; // [rsp+38h] [rbp-248h]
  unsigned int v98; // [rsp+38h] [rbp-248h]
  unsigned int v99; // [rsp+38h] [rbp-248h]
  int v100; // [rsp+38h] [rbp-248h]
  int v101; // [rsp+40h] [rbp-240h]
  const __m128i *v102; // [rsp+40h] [rbp-240h]
  int v103; // [rsp+40h] [rbp-240h]
  int v104; // [rsp+48h] [rbp-238h]
  const __m128i *v105; // [rsp+48h] [rbp-238h]
  const __m128i *v106; // [rsp+48h] [rbp-238h]
  int v107; // [rsp+48h] [rbp-238h]
  const __m128i *v108; // [rsp+48h] [rbp-238h]
  unsigned int v109; // [rsp+48h] [rbp-238h]
  const __m128i *v110; // [rsp+48h] [rbp-238h]
  int v111; // [rsp+48h] [rbp-238h]
  const __m128i *v112; // [rsp+48h] [rbp-238h]
  bool v113; // [rsp+52h] [rbp-22Eh]
  bool v114; // [rsp+53h] [rbp-22Dh]
  int v115; // [rsp+54h] [rbp-22Ch]
  int v116; // [rsp+58h] [rbp-228h]
  const __m128i *v117; // [rsp+58h] [rbp-228h]
  const __m128i *v118; // [rsp+58h] [rbp-228h]
  __m128i *v119; // [rsp+60h] [rbp-220h]
  const __m128i *v121; // [rsp+68h] [rbp-218h]
  int v122; // [rsp+68h] [rbp-218h]
  int v123; // [rsp+68h] [rbp-218h]
  int v124; // [rsp+68h] [rbp-218h]
  _BOOL4 v125; // [rsp+70h] [rbp-210h]
  unsigned int v126; // [rsp+74h] [rbp-20Ch]
  int v127; // [rsp+78h] [rbp-208h]
  const __m128i *v129; // [rsp+80h] [rbp-200h]
  const __m128i *v130; // [rsp+80h] [rbp-200h]
  const __m128i *v131; // [rsp+80h] [rbp-200h]
  const __m128i *v132; // [rsp+80h] [rbp-200h]
  int v133; // [rsp+88h] [rbp-1F8h]
  const __m128i *v134; // [rsp+88h] [rbp-1F8h]
  const __m128i *v135; // [rsp+88h] [rbp-1F8h]
  int v136; // [rsp+98h] [rbp-1E8h] BYREF
  int v137; // [rsp+9Ch] [rbp-1E4h] BYREF
  __m128i v138; // [rsp+A0h] [rbp-1E0h] BYREF
  __int64 v139; // [rsp+B0h] [rbp-1D0h]
  __m128i v140[3]; // [rsp+C0h] [rbp-1C0h] BYREF
  _DWORD v141[100]; // [rsp+F0h] [rbp-190h] BYREF

  v8 = (__m128i *)a1;
  v9 = a6;
  v10 = a2;
  if ( a6 )
    v9 = 2048;
  v133 = sub_8D32E0(a3);
  v126 = sub_8D3110(a3);
  sub_82D850((__int64)a7);
  a7[2].m128i_i64[0] = (__int64)a3;
  if ( a4 )
  {
    a7[1].m128i_i32[2] = *(_DWORD *)(a4 + 36);
    if ( (*(_BYTE *)(a4 + 34) & 0x20) != 0 )
      v9 |= 0x40u;
  }
  if ( a2 )
  {
    v116 = sub_8D2600(a2);
    if ( !v116 )
    {
      v115 = sub_8D3410(a2);
      if ( !v115 )
      {
        v125 = 0;
        v127 = sub_8D2310(a2);
        v119 = 0;
        if ( v127 )
          goto LABEL_30;
        v8 = 0;
        goto LABEL_12;
      }
      v125 = 0;
      v115 = 0;
      v119 = 0;
      goto LABEL_10;
    }
LABEL_22:
    a7->m128i_i32[2] = 7;
    result = 0;
    goto LABEL_23;
  }
  v10 = *(_QWORD *)a1;
  if ( (unsigned int)sub_8D2600(*(_QWORD *)a1) )
    goto LABEL_22;
  v116 = (*(_BYTE *)(a1 + 19) & 0x10) != 0;
  v115 = sub_6EB660(a1);
  v14 = *(_BYTE *)(a1 + 17);
  if ( v14 != 1 )
    goto LABEL_26;
  v125 = 1;
  if ( sub_6ED0A0(a1) )
  {
    v14 = *(_BYTE *)(a1 + 17);
LABEL_26:
    v125 = v14 == 3;
  }
  if ( (unsigned int)sub_8D3410(v10) )
  {
    v119 = (__m128i *)a1;
LABEL_10:
    if ( !v133
      || !(unsigned int)sub_831CF0(
                          (__int64)v119,
                          v10,
                          (__int64)a3,
                          0,
                          0,
                          &v137,
                          &v138,
                          v140[0].m128i_i32,
                          &v136,
                          v141,
                          0) )
    {
      v8 = 0;
      v127 = 1;
      v10 = sub_8D67C0(v10);
      goto LABEL_12;
    }
    if ( !v119 )
    {
      v127 = sub_8D2310(v10);
      if ( !v127 )
      {
        v8 = 0;
        goto LABEL_32;
      }
LABEL_120:
      if ( (unsigned int)sub_831CF0(0, v10, (__int64)a3, 0, 0, &v137, &v138, v140[0].m128i_i32, &v136, v141, 0) )
      {
        v127 = 0;
        v8 = v119;
        goto LABEL_32;
      }
LABEL_31:
      v8 = 0;
      v127 = 1;
      v10 = sub_6EE750(v10, (__int64)v119);
      if ( v133 )
        goto LABEL_32;
      goto LABEL_13;
    }
    v10 = v119->m128i_i64[0];
    v8 = v119;
  }
  v119 = v8;
  v127 = 0;
  if ( v8[1].m128i_i8[1] == 3 && v8[1].m128i_i8[0] != 3 )
  {
LABEL_30:
    if ( !v133 )
      goto LABEL_31;
    goto LABEL_120;
  }
LABEL_12:
  if ( v133 )
  {
LABEL_32:
    v15 = sub_8D46C0(a3);
    LODWORD(v16) = 0;
    i = (const __m128i *)v15;
    if ( (*(_BYTE *)(v15 + 140) & 0xFB) == 8 )
    {
      v108 = (const __m128i *)v15;
      v29 = sub_8D4C10(v15, dword_4F077C4 != 2);
      i = v108;
      LODWORD(v16) = v29;
    }
    LODWORD(v17) = 0;
    if ( (*(_BYTE *)(v10 + 140) & 0xFB) == 8 )
    {
      v102 = i;
      v107 = v16;
      v28 = sub_8D4C10(v10, dword_4F077C4 != 2);
      i = v102;
      LODWORD(v16) = v107;
      LODWORD(v17) = v28;
      if ( !v126 )
      {
LABEL_36:
        v101 = unk_4D04950 | qword_4D0495C | HIDWORD(qword_4D0495C);
        if ( !v101 )
        {
          if ( (dword_4F077C4 != 2 || unk_4F07778 <= 202001) && !dword_4F077BC
            || (v92 = v17,
                v97 = v16,
                v105 = i,
                v18 = sub_8D23E0(i),
                i = v105,
                LODWORD(v16) = v97,
                LODWORD(v17) = v92,
                !v18) )
          {
            if ( (v16 & 1) == 0 )
              goto LABEL_47;
            if ( (v16 & 2) != 0 )
            {
              if ( dword_4F077BC || (v101 = dword_4D04964) == 0 )
              {
LABEL_47:
                if ( (_DWORD)v16 == (_DWORD)v17 )
                {
                  v96 = 0;
                  v12 = 0;
                  v104 = 0;
                  v91 = 0;
                }
                else if ( ((unsigned int)v17 & ~(_DWORD)v16) != 0 )
                {
                  if ( !qword_4D0495C || (v106 = i, v19 = sub_8D3A70(i), i = v106, (v12 = v19) != 0) )
                  {
                    v90 = i;
                    v38 = sub_8DF8D0(i, v10);
                    i = v90;
                    v96 = 0;
                    v104 = 0;
                    v91 = 1;
                    v12 = v38 != 0;
                  }
                  else
                  {
                    v104 = 0;
                    v91 = 0;
                    v96 = 1;
                  }
                }
                else
                {
                  v96 = 0;
                  v12 = 0;
                  v104 = 1;
                  v91 = 0;
                }
                goto LABEL_53;
              }
              if ( dword_4F077C4 == 2 )
              {
                v101 = 0;
                if ( unk_4F07778 <= 201102 )
                  v101 = dword_4F07774 == 0;
                goto LABEL_47;
              }
            }
          }
        }
        v101 = 1;
        goto LABEL_47;
      }
    }
    else if ( !v126 )
    {
      goto LABEL_36;
    }
    v93 = i;
    v98 = v17;
    v109 = v16;
    v30 = sub_8D4D20(a3);
    LODWORD(v16) = v109;
    LODWORD(v17) = v98;
    v101 = v30;
    i = v93;
    if ( !v30 )
      goto LABEL_47;
    if ( !v8 )
      goto LABEL_47;
    if ( v8[1].m128i_i8[1] != 1 )
      goto LABEL_47;
    v31 = sub_6ED0A0((__int64)v8);
    LODWORD(v16) = v109;
    LODWORD(v17) = v98;
    i = v93;
    if ( v31 )
      goto LABEL_47;
    v32 = v93;
    v94 = v98;
    v99 = v109;
    v110 = i;
    v33 = sub_8DF8D0(v32, v10);
    i = v110;
    v16 = v99;
    v17 = v94;
    if ( !v33 || dword_4F077BC && qword_4F077A8 <= 0x9E33u )
      goto LABEL_47;
    v95 = v110;
    v100 = v17;
    v111 = v16;
    v36 = sub_6ECD10((__int64)v8, v10, v16, v17, v34, v35);
    LODWORD(v16) = v111;
    LODWORD(v17) = v100;
    i = v95;
    if ( v36 )
      goto LABEL_47;
    a7->m128i_i32[2] = 7;
    result = (unsigned __int8)v127;
LABEL_23:
    a7[1].m128i_i8[1] = result;
    return result;
  }
LABEL_13:
  while ( *(_BYTE *)(v10 + 140) == 12 )
    v10 = *(_QWORD *)(v10 + 160);
  for ( i = a3; i[8].m128i_i8[12] == 12; i = (const __m128i *)i[10].m128i_i64[0] )
    ;
  v12 = qword_4D0495C | HIDWORD(qword_4D0495C);
  if ( qword_4D0495C )
  {
    v96 = 0;
    v12 = 0;
    v104 = 0;
    v91 = 0;
    v101 = 1;
  }
  else if ( v8
         && v8[1].m128i_i8[1] == 1
         && (v103 = qword_4D0495C | HIDWORD(qword_4D0495C),
             v112 = i,
             v56 = sub_6ED0A0((__int64)v8),
             i = v112,
             v12 = v103,
             (v91 = v56) == 0) )
  {
    v57 = sub_6ED2B0((__int64)v8);
    i = v112;
    v12 = v103;
    if ( v57 )
    {
      v8 = (__m128i *)v141;
      sub_6E6A50(v57, (__int64)v141);
      i = v112;
      v96 = 0;
      v104 = 0;
      v12 = v103;
    }
    else
    {
      v96 = 0;
      v104 = 0;
    }
    v101 = 1;
  }
  else
  {
    v96 = 0;
    v104 = 0;
    v91 = 0;
    v101 = 1;
  }
LABEL_53:
  v20 = *(_BYTE *)(v10 + 140);
  for ( j = v10; v20 == 12; v20 = *(_BYTE *)(j + 140) )
    j = *(_QWORD *)(j + 160);
  v22 = i[8].m128i_i8[12];
  for ( k = i; v22 == 12; v22 = k[8].m128i_i8[12] )
    k = (const __m128i *)k[10].m128i_i64[0];
  if ( !v22 || !v20 )
  {
    a7->m128i_i32[2] = 6;
    a7[1].m128i_i8[1] = v127;
    return (unsigned __int8)v127;
  }
  v24 = v22 - 9;
  v114 = v24 <= 2u;
  v89 = v24 <= 2u;
  v113 = (unsigned __int8)(v20 - 9) <= 2u;
  if ( v8 )
  {
    if ( v8[1].m128i_i8[1] == 2 && v8[1].m128i_i8[0] == 2 )
    {
      v83 = 1;
      m128i_i64 = (__int64)v8[9].m128i_i64;
      v25 = 1;
    }
    else
    {
      m128i_i64 = 0;
      v25 = 0;
      v83 = 0;
    }
  }
  else
  {
    m128i_i64 = 0;
    v25 = 0;
    v83 = 0;
  }
  a7[1].m128i_i8[0] = v25;
  if ( v12 )
  {
    if ( !a5 )
    {
      if ( v24 > 2u || !v113 )
        goto LABEL_146;
      goto LABEL_138;
    }
    goto LABEL_95;
  }
  if ( v133 )
  {
    if ( k == (const __m128i *)j )
      goto LABEL_71;
    v86 = i;
    v45 = sub_8DED30(j, k, 1048579);
    i = v86;
    if ( v45 )
      goto LABEL_71;
    goto LABEL_126;
  }
  if ( k != (const __m128i *)j )
  {
    v80 = v24;
    v85 = i;
    v26 = sub_8DED30(j, k, 3);
    i = v85;
    v24 = v80;
    if ( !v26 )
    {
      v27 = sub_8D2E30(v85);
      i = v85;
      if ( v27 )
      {
        v70 = sub_8D2E30(v10);
        i = v85;
        if ( v70 )
        {
          v82 = sub_8D46C0(v10);
          v71 = sub_8D46C0(v85);
          v72 = sub_8DF7B0(v82, v71, 0, v140, 0);
          i = v85;
          if ( v72 )
          {
            v73 = v140[0].m128i_i32[0];
            a7[5].m128i_i8[4] |= 2u;
            a7->m128i_i32[2] = 0;
            a7[5].m128i_i32[0] = v73;
            a7[1].m128i_i8[1] = v127;
            goto LABEL_107;
          }
        }
      }
      if ( i[8].m128i_i8[12] == 3 && i[10].m128i_i8[0] == 2 && *(_BYTE *)(v10 + 140) == 3 && *(_BYTE *)(v10 + 160) == 1 )
      {
LABEL_71:
        a7->m128i_i32[2] = 0;
        a7[1].m128i_i8[1] = v127;
        goto LABEL_107;
      }
LABEL_126:
      if ( !v8 || v8[1].m128i_i8[0] != 3 )
      {
LABEL_128:
        v87 = i;
        v46 = sub_8E1010(v10, v83, v116, v115, 1, m128i_i64, (__int64)i, 0, 0, 1, 1015, (__int64)&v138, 0);
        i = v87;
        v12 = 0;
        if ( !v46 )
          goto LABEL_192;
        if ( qword_4D0495C )
        {
          if ( !v83 )
            goto LABEL_132;
          v61 = sub_712570(m128i_i64);
          i = v87;
          if ( v61 )
          {
            v62 = sub_8D2E30(v87);
            i = v87;
            v12 = 0;
            if ( v62 || (v77 = sub_8D3D10(v87), i = v87, v12 = 0, v77) )
            {
              if ( (v8[1].m128i_i8[3] & 0x40) == 0 )
              {
LABEL_192:
                if ( v114 && v113 )
                {
                  v118 = i;
                  v54 = sub_8D5CE0(v10, i);
                  i = v118;
                  v12 = 0;
                  if ( v54 && ((*(_BYTE *)(v54 + 96) & 4) == 0 || qword_4D03C50 && *(char *)(qword_4D03C50 + 18LL) >= 0) )
                  {
                    a7[5].m128i_i8[4] |= 0x20u;
                    a7->m128i_i32[2] = 2;
                    a7[4].m128i_i64[1] = v54;
                    if ( v133 )
                    {
                      if ( v8 )
                      {
                        v89 = 1;
                        a7[4].m128i_i8[0] = (4 * (v8[1].m128i_i8[1] == 1)) | a7[4].m128i_i8[0] & 0xFB;
                        a7[1].m128i_i8[1] = v127;
                        goto LABEL_107;
                      }
LABEL_156:
                      v89 = 1;
                      a7[1].m128i_i8[1] = v127;
                      goto LABEL_107;
                    }
                    if ( !a5 && v8 )
                    {
                      if ( !(unsigned int)sub_83E620(v8, v118) )
                      {
                        v78 = 12;
                        a7->m128i_i32[2] = 7;
                        m128i_i32 = a7[3].m128i_i32;
                        while ( v78 )
                        {
                          *m128i_i32++ = v133;
                          --v78;
                        }
                        a7[1].m128i_i8[1] = v127;
                        return (unsigned __int8)v127;
                      }
                      v55 = a7->m128i_i32[2];
                      a7[4].m128i_i8[0] |= 0x80u;
                      v89 = 1;
                      i = v118;
                      result = (unsigned int)(v55 - 4);
                      goto LABEL_106;
                    }
LABEL_155:
                    a7[4].m128i_i8[0] |= 0x80u;
                    goto LABEL_156;
                  }
                  if ( !a5 )
                  {
LABEL_138:
                    if ( a4 == 0 && v133 != 0 && (v91 & 1) == 0 )
                    {
                      v135 = i;
                      v49 = sub_8DD3B0(v10);
                      i = v135;
                      if ( v49 || (v50 = sub_8DD3B0(v135), i = v135, v50) )
                      {
                        a7[5].m128i_i8[4] |= 0x20u;
                        a7->m128i_i32[2] = 2;
                        a7[4].m128i_i64[1] = 0;
                        if ( v8 )
                          a7[4].m128i_i8[0] = (4 * (v8[1].m128i_i8[1] == 1)) | a7[4].m128i_i8[0] & 0xFB;
                        v91 = 0;
                        a7[1].m128i_i8[1] = v127;
                        goto LABEL_107;
                      }
                    }
LABEL_146:
                    a7->m128i_i32[2] = 7;
                    a7[1].m128i_i8[1] = v127;
                    return (unsigned __int8)v127;
                  }
                }
                else if ( !a5 )
                {
                  goto LABEL_146;
                }
LABEL_95:
                if ( !v133 )
                {
                  v39 = 0;
                  if ( v126 )
                  {
                    v123 = v12;
                    v39 = 0;
                    v131 = i;
                    v58 = sub_82EAE0();
                    i = v131;
                    v12 = v123;
                    if ( v58 )
                    {
                      if ( !sub_6ED230(v119) )
                        goto LABEL_146;
                      i = v131;
                      v12 = v123;
                    }
                  }
                  goto LABEL_97;
                }
                if ( v126 )
                {
                  v124 = v12;
                  v132 = i;
                  v59 = sub_82EAE0();
                  i = v132;
                  v12 = v124;
                  if ( v59 )
                  {
                    v60 = sub_6ED230(v119);
                    i = v132;
                    v12 = v124;
                    if ( !v60 )
                      goto LABEL_146;
                  }
                }
                if ( !v113 )
                {
                  v39 = (__int64)a3;
                  goto LABEL_97;
                }
                v130 = i;
                v122 = v12;
                v51 = sub_8413E0((_DWORD)v119, (_DWORD)a3, v9, 0, (unsigned int)v140, (unsigned int)&v137, 0);
                i = v130;
                if ( !(v137 | v51) )
                {
                  v12 = v122;
                  v39 = (__int64)a3;
LABEL_97:
                  if ( v12 )
                    goto LABEL_146;
                  v40 = v101 != 0;
                  if ( v114
                    && v101
                    && (v121 = i,
                        v41 = sub_836C50(v119, 0, i, 0, 1u, 1u, v39, 0, v9, (__int64)v140, 0, (unsigned int *)&v137, 0),
                        i = v121,
                        v40 = v101 != 0,
                        v137 | v41) )
                  {
                    sub_827240(a7, v140, a3, v133, 0);
                    i = v121;
                  }
                  else
                  {
                    if ( !v113 )
                      goto LABEL_146;
                    if ( !v40 )
                      goto LABEL_146;
                    v129 = i;
                    v42 = sub_840360((_DWORD)v119, (_DWORD)i, 0, 0, 1, 1, v39, 0, v9, (__int64)v140, (__int64)&v137, 0);
                    if ( !(v137 | v42) )
                      goto LABEL_146;
                    sub_827240(a7, v140, a3, v133, 0);
                    i = v129;
                  }
                  goto LABEL_105;
                }
                sub_827240(a7, v140, a3, v133, 1);
                i = v130;
LABEL_105:
                result = (unsigned int)(a7->m128i_i32[2] - 4);
                goto LABEL_106;
              }
            }
          }
        }
        else if ( !v83 )
        {
          goto LABEL_132;
        }
        if ( *(_BYTE *)(m128i_i64 + 173) == 12 )
        {
          v117 = i;
          v52 = sub_712690(m128i_i64);
          i = v117;
          if ( v52 )
          {
            v53 = sub_8D2E30(v117);
            i = v117;
            v12 = 0;
            if ( v53 )
              goto LABEL_192;
          }
        }
LABEL_132:
        v47 = v139;
        v48 = _mm_loadu_si128(&v138);
        a7->m128i_i32[2] = 2;
        a7[5].m128i_i64[1] = v47;
        *(__m128i *)((char *)a7 + 72) = v48;
        if ( v133 )
          a7[5].m128i_i8[4] = a7[5].m128i_i8[4] & 0xF9 | (2 * a7[5].m128i_i8[4]) & 4;
        if ( (v138.m128i_i8[12] & 0x40) != 0 )
        {
          a7->m128i_i32[2] = 1;
          a7[1].m128i_i8[1] = v125;
          v127 = v125;
          goto LABEL_107;
        }
        if ( (v138.m128i_i8[12] & 0x22) != 2 )
        {
          if ( (v138.m128i_i8[13] & 8) == 0 )
          {
            if ( HIDWORD(qword_4D0495C) && v133 && !v138.m128i_i64[0] )
            {
              a7->m128i_i32[2] = 4;
              a7[1].m128i_i8[1] = v125;
              return v125;
            }
            if ( (v138.m128i_i8[13] & 2) != 0 )
              a7->m128i_i32[2] = 3;
            a7[1].m128i_i8[1] = v125;
            v127 = v125;
            goto LABEL_107;
          }
          if ( (v138.m128i_i8[12] & 0x10) != 0 )
          {
            a7[5].m128i_i8[5] &= ~8u;
            a7->m128i_i8[13] = 1;
            a7[1].m128i_i8[1] = v125;
            v127 = v125;
            goto LABEL_107;
          }
        }
        a7->m128i_i32[2] = 0;
        a7[1].m128i_i8[1] = v125;
        v127 = v125;
        goto LABEL_107;
      }
      v63 = v8[1].m128i_i8[1];
      if ( v101 )
      {
        if ( !v126 || v63 == 2 )
          goto LABEL_239;
      }
      else
      {
        if ( v63 != 3 )
          goto LABEL_128;
        if ( !v126 )
          goto LABEL_239;
      }
      v88 = i;
      v64 = sub_6ED0A0((__int64)v8);
      i = v88;
      if ( !v64 )
        goto LABEL_128;
      v63 = v8[1].m128i_i8[1];
LABEL_239:
      v81 = i;
      v67 = sub_82C9F0(
              v8[8].m128i_i64[1],
              (v8[1].m128i_i8[3] & 8) != 0,
              v8[6].m128i_i64[1],
              v63 == 3,
              (__int64)a3,
              0,
              0,
              &a7->m128i_i32[2],
              (__int64)&v138,
              0,
              v140,
              &v137);
      i = v81;
      if ( v67 )
      {
        v68 = _mm_loadu_si128(&v138);
        v69 = v137;
        a7[5].m128i_i64[1] = v139;
        *(__m128i *)((char *)a7 + 72) = v68;
        if ( v69 )
          a7[4].m128i_i8[0] |= 8u;
        if ( (unsigned __int8)(*(_BYTE *)(v67 + 80) - 10) <= 1u && (*(_BYTE *)(*(_QWORD *)(v67 + 88) + 195LL) & 1) != 0 )
          a7[6].m128i_i64[0] = v67;
        goto LABEL_105;
      }
      v74 = v137;
      if ( v140[0].m128i_i32[0] )
      {
        v76 = v139;
        *(__m128i *)((char *)a7 + 72) = _mm_loadu_si128(&v138);
        a7[5].m128i_i64[1] = v76;
        if ( !v74 )
          goto LABEL_105;
      }
      else
      {
        if ( !v137 )
          goto LABEL_128;
        v75 = _mm_loadu_si128(&v138);
        a7[5].m128i_i64[1] = v139;
        *(__m128i *)((char *)a7 + 72) = v75;
      }
      a7[4].m128i_i8[0] |= 8u;
      goto LABEL_105;
    }
  }
  a7->m128i_i32[2] = 0;
  if ( v24 > 2u )
  {
    v89 = 0;
    a7[1].m128i_i8[1] = v127;
    goto LABEL_107;
  }
  if ( !v8 || a5 )
    goto LABEL_155;
  v134 = i;
  if ( !(unsigned int)sub_83E620(v8, i) )
  {
    a7->m128i_i32[2] = 7;
    a7[3] = 0;
    a7[1].m128i_i8[1] = v127;
    a7[4] = 0;
    a7[5] = 0;
    return (unsigned __int8)v127;
  }
  v37 = a7->m128i_i32[2];
  a7[4].m128i_i8[0] |= 0x80u;
  v89 = 1;
  i = v134;
  result = (unsigned int)(v37 - 4);
LABEL_106:
  a7[1].m128i_i8[1] = v127;
  if ( (unsigned int)result <= 0x7FFFFFFB )
    return result;
LABEL_107:
  if ( v104 )
  {
    a7[5].m128i_i8[4] |= 2u;
  }
  else if ( v96 )
  {
    a7->m128i_i8[13] = 1;
  }
  result = v126;
  if ( v126 )
  {
    result = v127 ^ 1u;
    if ( ((unsigned int)result & v125) == 0 )
      goto LABEL_111;
    result = sub_8D2310(i);
    if ( (_DWORD)result )
    {
      result = sub_8D3190();
      if ( (_DWORD)result )
      {
        a7[1].m128i_i8[3] = 1;
        goto LABEL_111;
      }
    }
    if ( !v91 )
    {
      result = (__int64)&dword_4F077BC;
      if ( dword_4F077BC )
      {
        result = (__int64)&qword_4F077A8;
        if ( qword_4F077A8 <= 0x9E33u )
        {
          a7->m128i_i8[12] = 1;
          goto LABEL_111;
        }
      }
    }
  }
  else
  {
    if ( v101 )
      goto LABEL_111;
    if ( !v127 )
    {
      if ( !v8 )
      {
        if ( a7->m128i_i32[2] != 7 )
          return result;
        goto LABEL_165;
      }
      if ( v8[1].m128i_i8[1] != 2 )
      {
        result = sub_6ED0A0((__int64)v8);
        if ( !(_DWORD)result )
        {
          if ( a7->m128i_i32[2] != 7 )
            goto LABEL_113;
          goto LABEL_165;
        }
      }
    }
    result = (__int64)&dword_4D0435C;
    if ( dword_4D0435C && v89 )
    {
LABEL_111:
      if ( a7->m128i_i32[2] != 7 )
        goto LABEL_112;
      goto LABEL_165;
    }
  }
  a7->m128i_i32[2] = 7;
LABEL_165:
  a7[3] = 0;
  a7[4] = 0;
  a7[5] = 0;
LABEL_112:
  if ( !v8 )
    return result;
LABEL_113:
  if ( a7[6].m128i_i64[0] )
    return result;
  if ( v8[1].m128i_i8[0] != 2 || (result = sub_8D3D10(v8->m128i_i64[0]), !(_DWORD)result) )
  {
    result = sub_8D2E30(v8->m128i_i64[0]);
    if ( !(_DWORD)result )
      return result;
    v43 = sub_724DC0();
    v44 = v8[1].m128i_i8[0] == 2;
    v140[0].m128i_i64[0] = (__int64)v43;
    if ( v44 )
    {
      v65 = (__int64)v8[9].m128i_i64;
    }
    else
    {
      if ( v8[1].m128i_i16[0] != 513 )
        return (__int64)sub_724E30((__int64)v140);
      if ( (unsigned int)sub_696840((__int64)v8) )
        return (__int64)sub_724E30((__int64)v140);
      if ( !(unsigned int)sub_717520((_QWORD *)v8[9].m128i_i64[0], v140[0].m128i_i64[0], 0) )
        return (__int64)sub_724E30((__int64)v140);
      v65 = v140[0].m128i_i64[0];
      if ( !v140[0].m128i_i64[0] )
        return (__int64)sub_724E30((__int64)v140);
    }
    if ( *(_BYTE *)(v65 + 173) == 6
      && !*(_BYTE *)(v65 + 176)
      && !*(_QWORD *)(v65 + 192)
      && (*(_BYTE *)(v65 + 168) & 8) == 0 )
    {
      v66 = *(_QWORD *)(v65 + 184);
      if ( (*(_BYTE *)(v66 + 195) & 1) != 0 )
        a7[6].m128i_i64[0] = *(_QWORD *)v66;
    }
    return (__int64)sub_724E30((__int64)v140);
  }
  if ( v8[19].m128i_i8[13] == 7 && (v8[21].m128i_i8[0] & 2) != 0 )
  {
    result = v8[21].m128i_i64[1];
    if ( result )
    {
      if ( (*(_BYTE *)(result + 195) & 1) != 0 )
      {
        result = *(_QWORD *)result;
        a7[6].m128i_i64[0] = result;
      }
    }
  }
  return result;
}
