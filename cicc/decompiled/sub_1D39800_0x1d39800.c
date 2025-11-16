// Function: sub_1D39800
// Address: 0x1d39800
//
__int64 *__fastcall sub_1D39800(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        const void **a5,
        unsigned __int16 a6,
        double a7,
        double a8,
        __m128i a9,
        __int64 *a10,
        __int64 a11)
{
  __int64 *result; // rax
  __int64 *v12; // r12
  __int64 v13; // r9
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 *v16; // r14
  __int16 v17; // ax
  __int64 v18; // rbx
  __int64 v19; // rax
  char v20; // cl
  __int64 v21; // rsi
  unsigned __int8 v22; // cl
  int v23; // eax
  unsigned int *v24; // r15
  int v25; // eax
  __int64 v26; // rcx
  __int64 v27; // r8
  bool v28; // zf
  const void **v29; // rdx
  char v30; // al
  __int64 *v31; // r15
  __int16 v32; // ax
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned __int64 v36; // r11
  __int64 v37; // r10
  __int64 v38; // rax
  char v39; // r13
  const void **v40; // rsi
  char v41; // al
  __int64 v42; // rax
  __int64 *v43; // rax
  __int64 v44; // r9
  __int64 v45; // r13
  __int64 v46; // rax
  unsigned __int8 v47; // r12
  const void **v48; // r14
  char v49; // al
  unsigned __int8 v50; // al
  const void **v51; // rdx
  __int64 v52; // rax
  __int64 *v53; // rax
  _QWORD *v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rax
  __int128 v57; // rax
  int v58; // r9d
  __int64 v59; // r14
  unsigned __int64 v60; // r15
  int v61; // eax
  __int64 v62; // rax
  __int64 *v63; // rax
  _QWORD *v64; // r12
  __int64 v65; // rdi
  __int64 *v66; // r15
  __int16 v67; // ax
  __int64 v68; // rdi
  __int16 v69; // ax
  __int64 v70; // rdi
  __int16 v71; // ax
  __int16 v72; // ax
  signed __int64 v73; // rax
  __int64 v74; // rax
  char v75; // cl
  __int64 v76; // rbx
  unsigned __int8 v77; // cl
  int v78; // eax
  __int64 v79; // rax
  char v80; // cl
  __int64 v81; // rsi
  unsigned __int8 v82; // cl
  int v83; // eax
  __int64 v84; // rax
  char v85; // cl
  __int64 v86; // rsi
  unsigned __int8 v87; // cl
  int v88; // eax
  __int64 v89; // rax
  char v90; // cl
  __int64 v91; // rsi
  unsigned __int8 v92; // cl
  int v93; // eax
  int v94; // r9d
  _QWORD *v95; // r12
  __int64 v96; // rdx
  __int64 v97; // r13
  __int64 v98; // rax
  _QWORD *v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // r10
  unsigned __int64 v105; // r11
  unsigned int v106; // r13d
  unsigned int v107; // eax
  unsigned int v108; // edx
  unsigned int v109; // eax
  unsigned int v110; // edx
  __int16 v111; // ax
  __int16 v112; // ax
  __int64 v113; // rax
  const void **v114; // rdx
  __int64 v115; // rcx
  char v116; // al
  __int64 v117; // rbx
  unsigned __int8 v118; // al
  int v119; // eax
  __int64 v120; // rcx
  char v121; // al
  __int64 v122; // rbx
  unsigned __int8 v123; // al
  int v124; // eax
  __int64 v125; // rsi
  char v126; // r15
  __int64 v127; // rbx
  unsigned int v128; // eax
  __int64 v129; // rdx
  __int64 v130; // r9
  char v131; // r14
  __int64 v132; // rdx
  __int64 v133; // rcx
  unsigned int v134; // ebx
  __int64 v135; // r8
  __int64 v136; // r9
  unsigned int v137; // eax
  __int128 v138; // [rsp-10h] [rbp-1E0h]
  __int64 v139; // [rsp-10h] [rbp-1E0h]
  __int128 v140; // [rsp-10h] [rbp-1E0h]
  __int128 v141; // [rsp-10h] [rbp-1E0h]
  __int64 v142; // [rsp-8h] [rbp-1D8h]
  const void **v143; // [rsp+18h] [rbp-1B8h]
  __int64 v144; // [rsp+20h] [rbp-1B0h]
  __int64 v145; // [rsp+38h] [rbp-198h]
  __int64 v146; // [rsp+40h] [rbp-190h]
  unsigned __int64 v147; // [rsp+48h] [rbp-188h]
  __int64 *v148; // [rsp+50h] [rbp-180h]
  __int64 v149; // [rsp+58h] [rbp-178h]
  __int64 v150; // [rsp+60h] [rbp-170h]
  __int64 v151; // [rsp+60h] [rbp-170h]
  __int64 v152; // [rsp+60h] [rbp-170h]
  __int64 v153; // [rsp+60h] [rbp-170h]
  __int64 v154; // [rsp+60h] [rbp-170h]
  unsigned __int64 v155; // [rsp+68h] [rbp-168h]
  unsigned __int64 v156; // [rsp+68h] [rbp-168h]
  unsigned __int64 v157; // [rsp+68h] [rbp-168h]
  unsigned __int64 v158; // [rsp+68h] [rbp-168h]
  unsigned int v159; // [rsp+70h] [rbp-160h]
  char v160; // [rsp+7Dh] [rbp-153h]
  __int64 *v163; // [rsp+88h] [rbp-148h]
  __int64 *v165; // [rsp+98h] [rbp-138h]
  __int64 v166; // [rsp+C0h] [rbp-110h] BYREF
  const void **v167; // [rsp+C8h] [rbp-108h]
  __int64 v168; // [rsp+D0h] [rbp-100h] BYREF
  const void **v169; // [rsp+D8h] [rbp-F8h]
  char v170[8]; // [rsp+E0h] [rbp-F0h] BYREF
  const void **v171; // [rsp+E8h] [rbp-E8h]
  __int64 v172; // [rsp+F0h] [rbp-E0h] BYREF
  const void **v173; // [rsp+F8h] [rbp-D8h]
  _BYTE *v174; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v175; // [rsp+108h] [rbp-C8h]
  _BYTE v176[64]; // [rsp+110h] [rbp-C0h] BYREF
  unsigned __int64 v177; // [rsp+150h] [rbp-80h] BYREF
  __int64 v178; // [rsp+158h] [rbp-78h]
  _QWORD v179[14]; // [rsp+160h] [rbp-70h] BYREF

  v166 = a4;
  v167 = a5;
  if ( a2 > 0x102 )
    return 0;
  v12 = a1;
  if ( sub_1D18610((__int64)a1, a2, (__int64)a10) )
  {
    v177 = 0;
    LODWORD(v178) = 0;
    v64 = sub_1D2B300(a1, 0x30u, (__int64)&v177, v166, (__int64)v167, v13);
    if ( v177 )
      sub_161E7C0((__int64)&v177, v177);
    return v64;
  }
  if ( (_BYTE)v166 )
  {
    if ( (unsigned __int8)(v166 - 14) > 0x5Fu )
      return 0;
    v159 = word_42E7700[(unsigned __int8)(v166 - 14)];
  }
  else
  {
    if ( !(unsigned __int8)sub_1F58D20(&v166) )
      return 0;
    v159 = sub_1F58D30(&v166);
  }
  v163 = &a10[2 * a11];
  v14 = (16 * a11) >> 6;
  v15 = (16 * a11) >> 4;
  if ( v14 <= 0 )
  {
    v16 = a10;
LABEL_77:
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_82;
LABEL_155:
        v111 = *(_WORD *)(*v16 + 24);
        if ( v111 == 104 )
        {
          if ( !(unsigned __int8)sub_1D23510(*v16) )
            goto LABEL_81;
LABEL_82:
          if ( v14 <= 0 )
          {
            v24 = (unsigned int *)a10;
            goto LABEL_84;
          }
          goto LABEL_13;
        }
        if ( v111 == 7 || v111 == 48 )
          goto LABEL_82;
LABEL_81:
        if ( v163 != v16 )
          return 0;
        goto LABEL_82;
      }
      v72 = *(_WORD *)(*v16 + 24);
      if ( v72 == 104 )
      {
        if ( !(unsigned __int8)sub_1D23510(*v16) )
          goto LABEL_81;
      }
      else if ( v72 != 7 && v72 != 48 )
      {
        goto LABEL_81;
      }
      v16 += 2;
    }
    v112 = *(_WORD *)(*v16 + 24);
    if ( v112 == 104 )
    {
      if ( !(unsigned __int8)sub_1D23510(*v16) )
        goto LABEL_81;
    }
    else if ( v112 != 48 && v112 != 7 )
    {
      goto LABEL_81;
    }
    v16 += 2;
    goto LABEL_155;
  }
  v16 = a10;
  while ( 1 )
  {
    v17 = *(_WORD *)(*v16 + 24);
    if ( v17 == 104 )
    {
      if ( !(unsigned __int8)sub_1D23510(*v16) )
        goto LABEL_12;
    }
    else if ( v17 != 7 && v17 != 48 )
    {
      goto LABEL_12;
    }
    v65 = v16[2];
    v66 = v16 + 2;
    v67 = *(_WORD *)(v65 + 24);
    if ( v67 == 104 )
    {
      if ( !(unsigned __int8)sub_1D23510(v65) )
        goto LABEL_94;
    }
    else if ( v67 != 7 && v67 != 48 )
    {
LABEL_94:
      v16 = v66;
      goto LABEL_12;
    }
    v68 = v16[4];
    v66 = v16 + 4;
    v69 = *(_WORD *)(v68 + 24);
    if ( v69 == 104 )
    {
      if ( !(unsigned __int8)sub_1D23510(v68) )
        goto LABEL_94;
    }
    else if ( v69 != 7 && v69 != 48 )
    {
      v16 += 4;
      goto LABEL_12;
    }
    v70 = v16[6];
    v66 = v16 + 6;
    v71 = *(_WORD *)(v70 + 24);
    if ( v71 != 104 )
      break;
    if ( !(unsigned __int8)sub_1D23510(v70) )
      goto LABEL_94;
LABEL_75:
    v16 += 8;
    if ( &a10[8 * v14] == v16 )
    {
      v15 = ((char *)v163 - (char *)v16) >> 4;
      goto LABEL_77;
    }
  }
  if ( v71 == 7 || v71 == 48 )
    goto LABEL_75;
  v16 += 6;
LABEL_12:
  if ( v16 != v163 )
    return 0;
LABEL_13:
  v18 = (__int64)a10;
  while ( 2 )
  {
    v19 = *(_QWORD *)(*(_QWORD *)v18 + 40LL) + 16LL * *(unsigned int *)(v18 + 8);
    v20 = *(_BYTE *)v19;
    v21 = *(_QWORD *)(v19 + 8);
    LOBYTE(v174) = v20;
    v175 = v21;
    if ( v20 )
    {
      v22 = v20 - 14;
      if ( v22 <= 0x5Fu )
      {
        v23 = word_42E7700[v22];
LABEL_17:
        if ( v159 != v23 )
        {
          v24 = (unsigned int *)v18;
          goto LABEL_19;
        }
      }
    }
    else if ( (unsigned __int8)sub_1F58D20(&v174) )
    {
      LOBYTE(v177) = 0;
      v178 = v21;
      v23 = sub_1F58D30(&v177);
      goto LABEL_17;
    }
    v79 = *(_QWORD *)(*(_QWORD *)(v18 + 16) + 40LL) + 16LL * *(unsigned int *)(v18 + 24);
    v80 = *(_BYTE *)v79;
    v81 = *(_QWORD *)(v79 + 8);
    LOBYTE(v174) = v80;
    v175 = v81;
    if ( v80 )
    {
      v82 = v80 - 14;
      if ( v82 <= 0x5Fu )
      {
        v83 = word_42E7700[v82];
LABEL_105:
        if ( v159 != v83 )
        {
          v24 = (unsigned int *)(v18 + 16);
          goto LABEL_19;
        }
      }
    }
    else if ( (unsigned __int8)sub_1F58D20(&v174) )
    {
      LOBYTE(v177) = 0;
      v178 = v81;
      v83 = sub_1F58D30(&v177);
      goto LABEL_105;
    }
    v84 = *(_QWORD *)(*(_QWORD *)(v18 + 32) + 40LL) + 16LL * *(unsigned int *)(v18 + 40);
    v85 = *(_BYTE *)v84;
    v86 = *(_QWORD *)(v84 + 8);
    LOBYTE(v174) = v85;
    v175 = v86;
    if ( v85 )
    {
      v87 = v85 - 14;
      if ( v87 <= 0x5Fu )
      {
        v88 = word_42E7700[v87];
LABEL_110:
        if ( v159 != v88 )
        {
          v24 = (unsigned int *)(v18 + 32);
          goto LABEL_19;
        }
      }
    }
    else if ( (unsigned __int8)sub_1F58D20(&v174) )
    {
      LOBYTE(v177) = 0;
      v178 = v86;
      v88 = sub_1F58D30(&v177);
      goto LABEL_110;
    }
    v89 = *(_QWORD *)(*(_QWORD *)(v18 + 48) + 40LL) + 16LL * *(unsigned int *)(v18 + 56);
    v90 = *(_BYTE *)v89;
    v91 = *(_QWORD *)(v89 + 8);
    LOBYTE(v174) = v90;
    v175 = v91;
    if ( v90 )
    {
      v92 = v90 - 14;
      if ( v92 <= 0x5Fu )
      {
        v93 = word_42E7700[v92];
LABEL_115:
        if ( v159 != v93 )
        {
          v24 = (unsigned int *)(v18 + 48);
          goto LABEL_19;
        }
      }
    }
    else if ( (unsigned __int8)sub_1F58D20(&v174) )
    {
      LOBYTE(v177) = 0;
      v178 = v91;
      v93 = sub_1F58D30(&v177);
      goto LABEL_115;
    }
    v18 += 64;
    if ( --v14 )
      continue;
    break;
  }
  v24 = (unsigned int *)v18;
LABEL_84:
  v73 = (char *)v163 - (char *)v24;
  if ( (char *)v163 - (char *)v24 != 32 )
  {
    if ( v73 != 48 )
    {
      if ( v73 == 16 )
        goto LABEL_87;
      goto LABEL_20;
    }
    v115 = *(_QWORD *)(*(_QWORD *)v24 + 40LL) + 16LL * v24[2];
    v116 = *(_BYTE *)v115;
    v117 = *(_QWORD *)(v115 + 8);
    LOBYTE(v174) = v116;
    v175 = v117;
    if ( v116 )
    {
      v118 = v116 - 14;
      if ( v118 <= 0x5Fu )
      {
        v119 = word_42E7700[v118];
        goto LABEL_178;
      }
      goto LABEL_179;
    }
    if ( !(unsigned __int8)sub_1F58D20(&v174) )
      goto LABEL_179;
    LOBYTE(v177) = 0;
    v178 = v117;
    v119 = sub_1F58D30(&v177);
LABEL_178:
    if ( v159 == v119 )
    {
LABEL_179:
      v24 += 4;
      goto LABEL_180;
    }
LABEL_19:
    if ( v163 == (__int64 *)v24 )
      goto LABEL_20;
    return 0;
  }
LABEL_180:
  v120 = *(_QWORD *)(*(_QWORD *)v24 + 40LL) + 16LL * v24[2];
  v121 = *(_BYTE *)v120;
  v122 = *(_QWORD *)(v120 + 8);
  LOBYTE(v174) = v121;
  v175 = v122;
  if ( v121 )
  {
    v123 = v121 - 14;
    if ( v123 <= 0x5Fu )
    {
      v124 = word_42E7700[v123];
LABEL_183:
      if ( v159 != v124 )
        goto LABEL_19;
    }
  }
  else if ( (unsigned __int8)sub_1F58D20(&v174) )
  {
    LOBYTE(v177) = 0;
    v178 = v122;
    v124 = sub_1F58D30(&v177);
    goto LABEL_183;
  }
  v24 += 4;
LABEL_87:
  v74 = *(_QWORD *)(*(_QWORD *)v24 + 40LL) + 16LL * v24[2];
  v75 = *(_BYTE *)v74;
  v76 = *(_QWORD *)(v74 + 8);
  LOBYTE(v174) = v75;
  v175 = v76;
  if ( v75 )
  {
    v77 = v75 - 14;
    if ( v77 > 0x5Fu )
      goto LABEL_20;
    v78 = word_42E7700[v77];
  }
  else
  {
    if ( !(unsigned __int8)sub_1F58D20(&v174) )
      goto LABEL_20;
    LOBYTE(v177) = 0;
    v178 = v76;
    v78 = sub_1F58D30(&v177);
  }
  if ( v159 != v78 )
    goto LABEL_19;
LABEL_20:
  if ( a2 == 137 )
  {
    v160 = 2;
    v143 = 0;
  }
  else
  {
    LOBYTE(v113) = sub_1D15870((char *)&v166);
    v144 = v113;
    v160 = v113;
    v143 = v114;
  }
  LOBYTE(v25) = sub_1D15870((char *)&v166);
  v28 = *((_BYTE *)v12 + 658) == 0;
  LODWORD(v168) = v25;
  v169 = v29;
  if ( !v28 )
  {
    v30 = (_BYTE)v168 ? (unsigned __int8)(v168 - 2) <= 5u || (unsigned __int8)(v168 - 14) <= 0x47u : sub_1F58CF0(&v168);
    if ( v30 )
    {
      v125 = v12[2];
      sub_1F40D10(&v177, v125, v12[6], v168, v169);
      v126 = v178;
      v127 = v179[0];
      LOBYTE(v168) = v178;
      v169 = (const void **)v179[0];
      LOBYTE(v128) = sub_1D15870((char *)&v166);
      v27 = v128;
      v178 = v129;
      v131 = v128;
      v177 = v128;
      if ( v126 == (_BYTE)v128 )
      {
        if ( (_BYTE)v128 || v127 == v129 )
          goto LABEL_26;
      }
      else if ( v126 )
      {
        v134 = sub_1D13440(v126);
LABEL_189:
        if ( v131 )
          v137 = sub_1D13440(v131);
        else
          v137 = sub_1F58D40(&v177, v125, v132, v133, v135, v136);
        if ( v137 > v134 )
          return 0;
        goto LABEL_26;
      }
      v134 = sub_1F58D40(&v168, v125, v128, v26, v128, v130);
      goto LABEL_189;
    }
  }
LABEL_26:
  v174 = v176;
  v175 = 0x400000000LL;
  if ( !v159 )
    goto LABEL_59;
  v148 = v12;
  v149 = 0;
  do
  {
    v31 = a10;
    v177 = (unsigned __int64)v179;
    v178 = 0x400000000LL;
    if ( a10 == v163 )
    {
      v54 = v179;
      v55 = 0;
      goto LABEL_49;
    }
    do
    {
      while ( 1 )
      {
        v44 = v31[1];
        v45 = *v31;
        v46 = *(_QWORD *)(*v31 + 40) + 16LL * (unsigned int)v44;
        v47 = *(_BYTE *)v46;
        v48 = *(const void ***)(v46 + 8);
        LOBYTE(v172) = v47;
        v173 = v48;
        if ( v47 )
        {
          if ( (unsigned __int8)(v47 - 14) <= 0x5Fu )
          {
            switch ( v47 )
            {
              case 0x18u:
              case 0x19u:
              case 0x1Au:
              case 0x1Bu:
              case 0x1Cu:
              case 0x1Du:
              case 0x1Eu:
              case 0x1Fu:
              case 0x20u:
              case 0x3Eu:
              case 0x3Fu:
              case 0x40u:
              case 0x41u:
              case 0x42u:
              case 0x43u:
                v47 = 3;
                break;
              case 0x21u:
              case 0x22u:
              case 0x23u:
              case 0x24u:
              case 0x25u:
              case 0x26u:
              case 0x27u:
              case 0x28u:
              case 0x44u:
              case 0x45u:
              case 0x46u:
              case 0x47u:
              case 0x48u:
              case 0x49u:
                v47 = 4;
                break;
              case 0x29u:
              case 0x2Au:
              case 0x2Bu:
              case 0x2Cu:
              case 0x2Du:
              case 0x2Eu:
              case 0x2Fu:
              case 0x30u:
              case 0x4Au:
              case 0x4Bu:
              case 0x4Cu:
              case 0x4Du:
              case 0x4Eu:
              case 0x4Fu:
                v47 = 5;
                break;
              case 0x31u:
              case 0x32u:
              case 0x33u:
              case 0x34u:
              case 0x35u:
              case 0x36u:
              case 0x50u:
              case 0x51u:
              case 0x52u:
              case 0x53u:
              case 0x54u:
              case 0x55u:
                v47 = 6;
                break;
              case 0x37u:
                v47 = 7;
                break;
              case 0x56u:
              case 0x57u:
              case 0x58u:
              case 0x62u:
              case 0x63u:
              case 0x64u:
                v47 = 8;
                break;
              case 0x59u:
              case 0x5Au:
              case 0x5Bu:
              case 0x5Cu:
              case 0x5Du:
              case 0x65u:
              case 0x66u:
              case 0x67u:
              case 0x68u:
              case 0x69u:
                v47 = 9;
                break;
              case 0x5Eu:
              case 0x5Fu:
              case 0x60u:
              case 0x61u:
              case 0x6Au:
              case 0x6Bu:
              case 0x6Cu:
              case 0x6Du:
                v47 = 10;
                break;
              default:
                v47 = 2;
                break;
            }
            v48 = 0;
          }
        }
        else
        {
          v150 = v44;
          v49 = sub_1F58D20(&v172);
          v44 = v150;
          if ( v49 )
          {
            v50 = sub_1F596B0(&v172);
            v44 = v150;
            v47 = v50;
            v32 = *(_WORD *)(v45 + 24);
            v48 = v51;
            v33 = v47;
            if ( v32 != 104 )
              goto LABEL_44;
LABEL_32:
            v34 = *(_QWORD *)(v45 + 32);
            v35 = v149;
            v36 = *(_QWORD *)(v34 + v149 + 8);
            v37 = *(_QWORD *)(v34 + v149);
            v38 = *(_QWORD *)(v37 + 40) + 16LL * (unsigned int)v36;
            v39 = *(_BYTE *)v38;
            v40 = *(const void ***)(v38 + 8);
            v170[0] = v39;
            v171 = v40;
            if ( v39 )
            {
              v41 = (unsigned __int8)(v39 - 14) <= 0x47u || (unsigned __int8)(v39 - 2) <= 5u;
            }
            else
            {
              v146 = v37;
              v147 = v36;
              v41 = sub_1F58CF0(v170);
              v37 = v146;
              v36 = v147;
            }
            if ( v41 )
            {
              v172 = v33;
              v173 = v48;
              if ( v47 != v39 )
              {
                if ( v39 )
                {
                  v106 = sub_1D13440(v39);
                  goto LABEL_137;
                }
LABEL_151:
                v152 = v37;
                v157 = v36;
                v109 = sub_1F58D40(v170, v40, v35, v26, v27, v44);
                v104 = v152;
                v105 = v157;
                v106 = v109;
LABEL_137:
                if ( v47 )
                {
                  v107 = sub_1D13440(v47);
                }
                else
                {
                  v151 = v104;
                  v156 = v105;
                  v107 = sub_1F58D40(&v172, v40, v100, v101, v102, v103);
                  v37 = v151;
                  v36 = v156;
                }
                if ( v107 < v106 )
                {
                  *((_QWORD *)&v141 + 1) = v36;
                  *(_QWORD *)&v141 = v37;
                  v155 = v36;
                  v37 = sub_1D309E0(v148, 145, a3, (unsigned int)v33, v48, 0, a7, a8, *(double *)a9.m128i_i64, v141);
                  v27 = v142;
                  v36 = v108 | v155 & 0xFFFFFFFF00000000LL;
                }
                goto LABEL_37;
              }
              if ( !v47 && v48 != v40 )
                goto LABEL_151;
            }
LABEL_37:
            v42 = (unsigned int)v178;
            if ( (unsigned int)v178 >= HIDWORD(v178) )
            {
              v153 = v37;
              v158 = v36;
              sub_16CD150((__int64)&v177, v179, 0, 16, v27, v44);
              v42 = (unsigned int)v178;
              v37 = v153;
              v36 = v158;
            }
            v43 = (__int64 *)(v177 + 16 * v42);
            *v43 = v37;
            v43[1] = v36;
            LODWORD(v178) = v178 + 1;
            goto LABEL_40;
          }
        }
        v32 = *(_WORD *)(v45 + 24);
        v33 = v47;
        if ( v32 == 104 )
          goto LABEL_32;
LABEL_44:
        if ( v32 != 48 )
          break;
        v172 = 0;
        LODWORD(v173) = 0;
        v95 = sub_1D2B300(v148, 0x30u, (__int64)&v172, v33, (__int64)v48, v44);
        v97 = v96;
        if ( v172 )
          sub_161E7C0((__int64)&v172, v172);
        v98 = (unsigned int)v178;
        if ( (unsigned int)v178 >= HIDWORD(v178) )
        {
          sub_16CD150((__int64)&v177, v179, 0, 16, v27, v94);
          v98 = (unsigned int)v178;
        }
        v99 = (_QWORD *)(v177 + 16 * v98);
        *v99 = v95;
        v99[1] = v97;
        LODWORD(v178) = v178 + 1;
LABEL_40:
        v31 += 2;
        if ( v163 == v31 )
          goto LABEL_48;
      }
      v52 = (unsigned int)v178;
      if ( (unsigned int)v178 >= HIDWORD(v178) )
      {
        v154 = v44;
        sub_16CD150((__int64)&v177, v179, 0, 16, v27, v44);
        v52 = (unsigned int)v178;
        v44 = v154;
      }
      v53 = (__int64 *)(v177 + 16 * v52);
      v31 += 2;
      *v53 = v45;
      v53[1] = v44;
      LODWORD(v178) = v178 + 1;
    }
    while ( v163 != v31 );
LABEL_48:
    v54 = (_QWORD *)v177;
    v55 = (unsigned int)v178;
LABEL_49:
    v145 = v55;
    v56 = v144;
    *((_QWORD *)&v138 + 1) = v145;
    *(_QWORD *)&v138 = v54;
    LOBYTE(v56) = v160;
    v144 = v56;
    *(_QWORD *)&v57 = sub_1D359D0(v148, a2, a3, (unsigned int)v56, v143, a6, a7, a8, a9, v138);
    v26 = v139;
    v60 = *((_QWORD *)&v57 + 1);
    v59 = v57;
    if ( v160 != (_BYTE)v168 || !v160 && v169 != v143 )
    {
      v59 = sub_1D309E0(v148, 142, a3, (unsigned int)v168, v169, 0, a7, a8, *(double *)a9.m128i_i64, v57);
      v60 = v110 | v60 & 0xFFFFFFFF00000000LL;
    }
    v61 = *(unsigned __int16 *)(v59 + 24);
    if ( (_WORD)v61 != 48 && (unsigned int)(v61 - 10) > 1 )
    {
      result = 0;
      if ( (_QWORD *)v177 != v179 )
      {
        _libc_free(v177);
        result = 0;
      }
      goto LABEL_60;
    }
    v62 = (unsigned int)v175;
    if ( (unsigned int)v175 >= HIDWORD(v175) )
    {
      sub_16CD150((__int64)&v174, v176, 0, 16, v27, v58);
      v62 = (unsigned int)v175;
    }
    v63 = (__int64 *)&v174[16 * v62];
    *v63 = v59;
    v63[1] = v60;
    LODWORD(v175) = v175 + 1;
    if ( (_QWORD *)v177 != v179 )
      _libc_free(v177);
    v149 += 40;
  }
  while ( 40LL * v159 != v149 );
  v12 = v148;
LABEL_59:
  *((_QWORD *)&v140 + 1) = (unsigned int)v175;
  *(_QWORD *)&v140 = v174;
  result = sub_1D359D0(v12, 104, a3, v166, v167, 0, a7, a8, a9, v140);
LABEL_60:
  if ( v174 != v176 )
  {
    v165 = result;
    _libc_free((unsigned __int64)v174);
    return v165;
  }
  return result;
}
