// Function: sub_2CE9A90
// Address: 0x2ce9a90
//
unsigned __int64 __fastcall sub_2CE9A90(
        _QWORD *a1,
        __int64 a2,
        unsigned __int8 *a3,
        __int64 a4,
        _QWORD *a5,
        int a6,
        int a7)
{
  int v11; // r11d
  _QWORD *v12; // rax
  _QWORD *v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rdx
  char v17; // al
  unsigned __int64 v18; // r12
  unsigned __int8 v19; // al
  _BYTE *v20; // rax
  char v21; // r11
  __int64 v23; // r15
  __int64 *v24; // rax
  _QWORD *v25; // rax
  _QWORD *v26; // rbx
  char v27; // r11
  const char *v28; // rsi
  const char **v29; // r12
  unsigned __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // r15
  __int64 v37; // r14
  _QWORD *v38; // rax
  _QWORD *v39; // rdx
  _QWORD *v40; // rdi
  _QWORD *v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rax
  unsigned __int8 v44; // r11
  __int64 v45; // r10
  _QWORD *v46; // rax
  _QWORD *v47; // r8
  __int64 *v48; // rax
  __int64 v49; // rbx
  __int64 v50; // r15
  _QWORD *v51; // rax
  __int64 v52; // r13
  __int64 v53; // rax
  __int64 v54; // rsi
  _QWORD *v55; // rax
  _QWORD *v56; // rdx
  char v57; // di
  __int64 v58; // rax
  _QWORD *v59; // rax
  _QWORD *v60; // rdx
  char v61; // di
  unsigned __int64 v62; // r13
  __int64 v63; // r12
  __int64 v64; // rbx
  __int64 i; // rax
  int v66; // eax
  unsigned int v67; // edi
  __int64 v68; // rax
  __int64 v69; // rdi
  __int64 v70; // rdi
  __int64 v71; // rdx
  __int64 v72; // rsi
  int v73; // eax
  __int64 v74; // rax
  int v75; // r11d
  const char **v76; // rbx
  _BYTE *v77; // rdx
  _BYTE *v78; // rsi
  const char *v79; // rax
  __int64 v80; // rsi
  unsigned __int8 *v81; // rsi
  __int64 *v82; // r8
  __int64 *v83; // r15
  _QWORD *v84; // rax
  char v85; // r11
  _QWORD *v86; // rbx
  __int64 v87; // rax
  unsigned int v88; // ecx
  char v89; // al
  size_t v90; // rdx
  char v91; // r11
  char *v92; // r15
  size_t v93; // rax
  __int64 v94; // rax
  __int64 v95; // rcx
  size_t v96; // rdx
  char *v97; // r15
  size_t v98; // rax
  size_t v99; // rdx
  char *v100; // r15
  size_t v101; // rax
  __int64 v102; // rax
  __int64 v103; // rcx
  size_t v104; // rdx
  char *v105; // r15
  size_t v106; // rax
  const char *v107; // rsi
  const char **v108; // r12
  __int64 v109; // rsi
  unsigned __int8 *v110; // rsi
  __int64 *v111; // rdx
  __int64 v112; // rsi
  int v113; // edi
  unsigned __int64 v114; // rdi
  _QWORD *v115; // rax
  char v116; // dl
  int v117; // esi
  char v118; // [rsp+14h] [rbp-ECh]
  __int64 *v119; // [rsp+18h] [rbp-E8h]
  unsigned __int8 v120; // [rsp+20h] [rbp-E0h]
  unsigned __int8 v121; // [rsp+20h] [rbp-E0h]
  __int64 v122; // [rsp+20h] [rbp-E0h]
  int v123; // [rsp+20h] [rbp-E0h]
  char v124; // [rsp+20h] [rbp-E0h]
  __int64 v125; // [rsp+28h] [rbp-D8h]
  __int64 v126; // [rsp+28h] [rbp-D8h]
  __int64 v127; // [rsp+28h] [rbp-D8h]
  __int64 v128; // [rsp+28h] [rbp-D8h]
  char v129; // [rsp+30h] [rbp-D0h]
  char v130; // [rsp+30h] [rbp-D0h]
  __int64 v131; // [rsp+30h] [rbp-D0h]
  __int64 v132; // [rsp+30h] [rbp-D0h]
  int v133; // [rsp+30h] [rbp-D0h]
  _QWORD *v134; // [rsp+30h] [rbp-D0h]
  __int64 v135; // [rsp+38h] [rbp-C8h]
  int v136; // [rsp+38h] [rbp-C8h]
  int v137; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v138; // [rsp+38h] [rbp-C8h]
  int v139; // [rsp+38h] [rbp-C8h]
  int v140; // [rsp+38h] [rbp-C8h]
  int v141; // [rsp+38h] [rbp-C8h]
  __int64 v142; // [rsp+38h] [rbp-C8h]
  char v143; // [rsp+38h] [rbp-C8h]
  char v144; // [rsp+38h] [rbp-C8h]
  char v145; // [rsp+40h] [rbp-C0h]
  char v146; // [rsp+40h] [rbp-C0h]
  char v147; // [rsp+40h] [rbp-C0h]
  __int64 v148; // [rsp+40h] [rbp-C0h]
  _QWORD *v149; // [rsp+40h] [rbp-C0h]
  int v150; // [rsp+40h] [rbp-C0h]
  __int64 v151; // [rsp+40h] [rbp-C0h]
  char v152; // [rsp+40h] [rbp-C0h]
  char v153; // [rsp+40h] [rbp-C0h]
  char v154; // [rsp+40h] [rbp-C0h]
  char v155; // [rsp+40h] [rbp-C0h]
  __int64 v156; // [rsp+40h] [rbp-C0h]
  char v157; // [rsp+40h] [rbp-C0h]
  char v158; // [rsp+40h] [rbp-C0h]
  char v159; // [rsp+40h] [rbp-C0h]
  __int64 v160; // [rsp+40h] [rbp-C0h]
  char v161; // [rsp+40h] [rbp-C0h]
  char v162; // [rsp+40h] [rbp-C0h]
  char v163; // [rsp+40h] [rbp-C0h]
  char v164; // [rsp+40h] [rbp-C0h]
  __int64 v165; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v166[2]; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v167; // [rsp+58h] [rbp-A8h]
  unsigned __int64 v168; // [rsp+60h] [rbp-A0h] BYREF
  _BYTE *v169; // [rsp+68h] [rbp-98h]
  _BYTE *v170; // [rsp+70h] [rbp-90h]
  char *v171; // [rsp+80h] [rbp-80h]
  __int64 v172; // [rsp+88h] [rbp-78h]
  char v173; // [rsp+90h] [rbp-70h] BYREF
  const char *v174[4]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v175; // [rsp+C0h] [rbp-40h]

  v11 = a7;
  v166[0] = (unsigned __int64)a3;
  if ( !(_BYTE)a7 )
  {
    v12 = (_QWORD *)a5[2];
    if ( v12 )
    {
      v14 = a5 + 1;
      do
      {
        while ( 1 )
        {
          v15 = v12[2];
          v16 = v12[3];
          if ( v12[4] >= (unsigned __int64)a3 )
            break;
          v12 = (_QWORD *)v12[3];
          if ( !v16 )
            goto LABEL_7;
        }
        v14 = v12;
        v12 = (_QWORD *)v12[2];
      }
      while ( v15 );
LABEL_7:
      if ( a5 + 1 != v14 && v14[4] <= (unsigned __int64)a3 )
        return *(_QWORD *)sub_2CE6A60(a5, v166);
    }
    v17 = sub_2CE8360((__int64)a1, a2, a3);
    v11 = a7;
    if ( !v17 )
    {
      v30 = v166[0];
      *(_QWORD *)sub_2CE6A60(a5, v166) = v30;
      return v166[0];
    }
  }
  v18 = v166[0];
  v171 = &v173;
  v172 = 0x200000000LL;
  v19 = *(_BYTE *)v166[0];
  if ( *(_BYTE *)v166[0] == 3 )
  {
LABEL_26:
    *(_QWORD *)sub_2CE6A60(a5, v166) = v18;
    return v166[0];
  }
  switch ( v19 )
  {
    case 0x3Cu:
    case 0x4Du:
    case 0x16u:
    case 0x4Fu:
LABEL_14:
      v145 = v11;
      goto LABEL_15;
    case 0x55u:
      v31 = *(_QWORD *)(v166[0] - 32);
      if ( v31
        && !*(_BYTE *)v31
        && *(_QWORD *)(v31 + 24) == *(_QWORD *)(v166[0] + 80)
        && (*(_BYTE *)(v31 + 33) & 0x20) != 0 )
      {
        if ( *(_DWORD *)(v31 + 36) != 8170 )
          return v18;
        v147 = v11;
        v32 = sub_2CE9A90(
                (_DWORD)a1,
                a2,
                *(_QWORD *)(v166[0] - 32LL * (*(_DWORD *)(v166[0] + 4) & 0x7FFFFFF)),
                v166[0],
                (_DWORD)a5,
                a6,
                (unsigned __int8)v11);
        v21 = v147;
        v33 = v32;
        v34 = *(_QWORD *)(v32 + 8);
        v35 = v34;
        if ( (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17 <= 1 )
          v35 = **(_QWORD **)(v34 + 16);
        if ( *(_DWORD *)(v35 + 8) <= 0x1FFu )
        {
          v174[0] = (const char *)v34;
          v174[1] = (const char *)v34;
          v48 = (__int64 *)sub_B43CA0(v166[0]);
          v168 = v33;
          v49 = 0;
          v50 = sub_B6E160(v48, 0x1FEAu, (__int64)v174, 2);
          v175 = 257;
          v169 = *(_BYTE **)(v166[0] + 32 * (1LL - (*(_DWORD *)(v166[0] + 4) & 0x7FFFFFF)));
          if ( v50 )
            v49 = *(_QWORD *)(v50 + 24);
          v51 = sub_BD2C40(88, 3u);
          v21 = v147;
          v18 = (unsigned __int64)v51;
          if ( v51 )
          {
            sub_B44260((__int64)v51, **(_QWORD **)(v49 + 16), 56, 3u, v166[0] + 24, 0);
            *(_QWORD *)(v18 + 72) = 0;
            sub_B4A290(v18, v49, v50, (__int64 *)&v168, 2, (__int64)v174, 0, 0);
            v21 = v147;
          }
        }
        else
        {
          v18 = v33;
        }
LABEL_16:
        if ( v21 )
          return v18;
LABEL_34:
        *(_QWORD *)sub_2CE6A60(a5, v166) = v18;
        return v18;
      }
      goto LABEL_14;
    case 0x4Eu:
      v129 = v11;
      v23 = sub_2CE9A90((_DWORD)a1, a2, *(_QWORD *)(v166[0] - 32), v166[0], (_DWORD)a5, a6, (unsigned __int8)v11);
      v24 = (__int64 *)sub_BD5C60(v166[0]);
      v135 = sub_BCE3C0(v24, a6);
      v174[0] = "bitCast";
      v175 = 259;
      v25 = sub_BD2C40(72, 1u);
      v26 = v25;
      v27 = v129;
      if ( v25 )
      {
        sub_B51BF0((__int64)v25, v23, v135, (__int64)v174, v166[0] + 24, 0);
        v27 = v129;
      }
LABEL_29:
      v28 = *(const char **)(v166[0] + 48);
      v29 = (const char **)(v26 + 6);
      v174[0] = v28;
      if ( v28 )
      {
        v146 = v27;
        sub_B96E90((__int64)v174, (__int64)v28, 1);
        v27 = v146;
        if ( v29 == v174 )
        {
          if ( v174[0] )
          {
            sub_B91220((__int64)v174, (__int64)v174[0]);
            v27 = v146;
          }
          goto LABEL_33;
        }
        v80 = v26[6];
        if ( !v80 )
        {
LABEL_117:
          v81 = (unsigned __int8 *)v174[0];
          v26[6] = v174[0];
          if ( v81 )
          {
            v153 = v27;
            sub_B976B0((__int64)v174, v81, (__int64)(v26 + 6));
            v27 = v153;
          }
LABEL_33:
          v18 = (unsigned __int64)v26;
          if ( v27 )
            return v18;
          goto LABEL_34;
        }
      }
      else
      {
        if ( v29 == v174 )
          goto LABEL_33;
        v80 = v26[6];
        if ( !v80 )
          goto LABEL_33;
      }
      v152 = v27;
      sub_B91220((__int64)(v26 + 6), v80);
      v27 = v152;
      goto LABEL_117;
    case 0x5Du:
      goto LABEL_14;
    case 0x3Fu:
      v140 = v11;
      v74 = sub_2CE9A90(
              (_DWORD)a1,
              a2,
              *(_QWORD *)(v166[0] - 32LL * (*(_DWORD *)(v166[0] + 4) & 0x7FFFFFF)),
              v166[0],
              (_DWORD)a5,
              a6,
              (unsigned __int8)v11);
      v75 = v140;
      v168 = 0;
      v151 = v74;
      v169 = 0;
      v170 = 0;
      v76 = (const char **)(v166[0] + 32 * (1LL - (*(_DWORD *)(v166[0] + 4) & 0x7FFFFFF)));
      if ( (const char **)v166[0] == v76 )
      {
        v82 = 0;
      }
      else
      {
        v77 = 0;
        v78 = 0;
        while ( 1 )
        {
          v79 = *v76;
          v174[0] = *v76;
          if ( v78 == v77 )
          {
            v141 = v75;
            sub_928380((__int64)&v168, v78, v174);
            v78 = v169;
            v75 = v141;
          }
          else
          {
            if ( v78 )
            {
              *(_QWORD *)v78 = v79;
              v78 = v169;
            }
            v78 += 8;
            v169 = v78;
          }
          v76 += 4;
          if ( (const char **)v18 == v76 )
            break;
          v77 = v170;
        }
        v82 = (__int64 *)v78;
      }
      v83 = (__int64 *)v168;
      v174[0] = "getElem";
      v175 = 259;
      v118 = v75;
      v128 = *(_QWORD *)(v18 + 72);
      v119 = v82;
      v142 = (__int64)((__int64)v82 - v168) >> 3;
      v123 = v142 + 1;
      v84 = sub_BD2C40(88, (int)v142 + 1);
      v85 = v118;
      v86 = v84;
      if ( v84 )
      {
        v87 = *(_QWORD *)(v151 + 8);
        v88 = v123 & 0x7FFFFFF;
        if ( (unsigned int)*(unsigned __int8 *)(v87 + 8) - 17 > 1 && v83 != v119 )
        {
          v111 = v83;
          while ( 1 )
          {
            v112 = *(_QWORD *)(*v111 + 8);
            v113 = *(unsigned __int8 *)(v112 + 8);
            if ( v113 == 17 )
              break;
            if ( v113 == 18 )
            {
              v116 = 1;
              goto LABEL_165;
            }
            if ( v119 == ++v111 )
              goto LABEL_122;
          }
          v116 = 0;
LABEL_165:
          v117 = *(_DWORD *)(v112 + 32);
          BYTE4(v167) = v116;
          LODWORD(v167) = v117;
          v87 = sub_BCE1B0((__int64 *)v87, v167);
          v85 = v118;
          v88 = v123 & 0x7FFFFFF;
        }
LABEL_122:
        v124 = v85;
        sub_B44260((__int64)v86, v87, 34, v88, v18 + 24, 0);
        v86[9] = v128;
        v86[10] = sub_B4DC50(v128, (__int64)v83, v142);
        sub_B4D9A0((__int64)v86, v151, v83, v142, (__int64)v174);
        v85 = v124;
      }
      v154 = v85;
      v89 = sub_B4DE30(v18);
      sub_B4DE00((__int64)v86, v89);
      v90 = 0;
      v91 = v154;
      v92 = off_4C5D0D0[0];
      if ( off_4C5D0D0[0] )
      {
        v93 = strlen(off_4C5D0D0[0]);
        v91 = v154;
        v90 = v93;
      }
      if ( *(_QWORD *)(v18 + 48) || (*(_BYTE *)(v18 + 7) & 0x20) != 0 )
      {
        v155 = v91;
        v94 = sub_B91F50(v18, v92, v90);
        v91 = v155;
        v95 = v94;
        if ( v94 )
        {
          v96 = 0;
          v97 = off_4C5D0D0[0];
          if ( off_4C5D0D0[0] )
          {
            v143 = v155;
            v156 = v94;
            v98 = strlen(off_4C5D0D0[0]);
            v91 = v143;
            v95 = v156;
            v96 = v98;
          }
          v157 = v91;
          sub_B9A090((__int64)v86, v97, v96, v95);
          v91 = v157;
        }
      }
      v99 = 0;
      v100 = off_4C5D0D8[0];
      if ( off_4C5D0D8[0] )
      {
        v158 = v91;
        v101 = strlen(off_4C5D0D8[0]);
        v91 = v158;
        v99 = v101;
      }
      if ( *(_QWORD *)(v18 + 48) || (*(_BYTE *)(v18 + 7) & 0x20) != 0 )
      {
        v159 = v91;
        v102 = sub_B91F50(v18, v100, v99);
        v91 = v159;
        v103 = v102;
        if ( v102 )
        {
          v104 = 0;
          v105 = off_4C5D0D8[0];
          if ( off_4C5D0D8[0] )
          {
            v144 = v159;
            v160 = v102;
            v106 = strlen(off_4C5D0D8[0]);
            v91 = v144;
            v103 = v160;
            v104 = v106;
          }
          v161 = v91;
          sub_B9A090((__int64)v86, v105, v104, v103);
          v91 = v161;
        }
      }
      v107 = *(const char **)(v18 + 48);
      v108 = (const char **)(v86 + 6);
      v174[0] = v107;
      if ( v107 )
      {
        v162 = v91;
        sub_B96E90((__int64)v174, (__int64)v107, 1);
        v91 = v162;
        if ( v108 == v174 )
        {
          if ( v174[0] )
          {
            sub_B91220((__int64)v174, (__int64)v174[0]);
            v91 = v162;
          }
          goto LABEL_141;
        }
        v109 = v86[6];
        if ( !v109 )
        {
LABEL_149:
          v110 = (unsigned __int8 *)v174[0];
          v86[6] = v174[0];
          if ( v110 )
          {
            v164 = v91;
            sub_B976B0((__int64)v174, v110, (__int64)(v86 + 6));
            v91 = v164;
          }
          goto LABEL_141;
        }
      }
      else if ( v108 == v174 || (v109 = v86[6]) == 0 )
      {
LABEL_141:
        if ( !v91 )
          *(_QWORD *)sub_2CE6A60(a5, v166) = v86;
        if ( v168 )
          j_j___libc_free_0(v168);
        return (unsigned __int64)v86;
      }
      v163 = v91;
      sub_B91220((__int64)(v86 + 6), v109);
      v91 = v163;
      goto LABEL_149;
  }
  if ( v19 <= 0x1Cu )
  {
    v145 = v11;
    if ( v19 != 5 )
      goto LABEL_26;
    goto LABEL_15;
  }
  if ( v19 != 84 )
  {
    if ( v19 == 86 )
    {
      v130 = v11;
      v136 = (unsigned __int8)v11;
      v148 = sub_2CEAB40((_DWORD)a1, a2, *(_QWORD *)(v166[0] - 64), v166[0], (_DWORD)a5, a6, (unsigned __int8)v11);
      v36 = sub_2CEAB40((_DWORD)a1, a2, *(_QWORD *)(v166[0] - 32), v166[0], (_DWORD)a5, a6, v136);
      v174[0] = "selectInst";
      v175 = 259;
      v37 = *(_QWORD *)(v166[0] - 96);
      v38 = sub_BD2C40(72, 3u);
      v27 = v130;
      v26 = v38;
      if ( v38 )
      {
        sub_B44260((__int64)v38, *(_QWORD *)(v148 + 8), 57, 3u, v166[0] + 24, 0);
        sub_AC2B30((__int64)(v26 - 12), v37);
        sub_AC2B30((__int64)(v26 - 8), v148);
        sub_AC2B30((__int64)(v26 - 4), v36);
        sub_BD6B50((unsigned __int8 *)v26, v174);
        v27 = v130;
      }
      goto LABEL_29;
    }
    v145 = v11;
    if ( v19 != 61 )
      goto LABEL_26;
LABEL_15:
    v20 = sub_2CE76E0((__int64)a1, v166[0], a6, a2);
    v21 = v145;
    v18 = (unsigned __int64)v20;
    goto LABEL_16;
  }
  v39 = (_QWORD *)a1[24];
  v149 = a1 + 23;
  if ( !v39 )
    goto LABEL_57;
  v40 = a1 + 23;
  v41 = (_QWORD *)a1[24];
  do
  {
    if ( v41[4] < v166[0] )
    {
      v41 = (_QWORD *)v41[3];
    }
    else
    {
      v40 = v41;
      v41 = (_QWORD *)v41[2];
    }
  }
  while ( v41 );
  if ( v40 != v149 && v40[4] <= v166[0] )
  {
    v52 = (__int64)(a1 + 23);
    do
    {
      if ( v39[4] < v166[0] )
      {
        v39 = (_QWORD *)v39[3];
      }
      else
      {
        v52 = (__int64)v39;
        v39 = (_QWORD *)v39[2];
      }
    }
    while ( v39 );
    if ( (_QWORD *)v52 == v149 || *(_QWORD *)(v52 + 32) > v166[0] )
    {
      v53 = sub_22077B0(0x30u);
      v54 = v52;
      *(_QWORD *)(v53 + 32) = v166[0];
      v52 = v53;
      *(_QWORD *)(v53 + 40) = 0;
      v55 = sub_2CE6C00(a1 + 22, v54, (unsigned __int64 *)(v53 + 32));
      if ( v56 )
      {
        v57 = v149 == v56 || v55 || v56[4] > v18;
        sub_220F040(v57, v52, v56, v149);
        ++a1[27];
      }
      else
      {
        v114 = v52;
        v52 = (__int64)v55;
        j_j___libc_free_0(v114);
      }
    }
    return *(_QWORD *)(v52 + 40);
  }
  else
  {
LABEL_57:
    v120 = v11;
    v42 = (__int64 *)sub_BD5C60(v166[0]);
    v125 = sub_BCE3C0(v42, a6);
    v174[0] = "phiNode";
    v175 = 259;
    v137 = *(_DWORD *)(v166[0] + 4) & 0x7FFFFFF;
    v43 = sub_BD2DA0(80);
    v44 = v120;
    v45 = v43;
    if ( v43 )
    {
      v131 = v43;
      sub_B44260(v43, v125, 55, 0x8000000u, v166[0] + 24, 0);
      *(_DWORD *)(v131 + 72) = v137;
      sub_BD6B50((unsigned __int8 *)v131, v174);
      sub_BD2A10(v131, *(_DWORD *)(v131 + 72), 1);
      v44 = v120;
      v45 = v131;
    }
    v46 = (_QWORD *)a1[24];
    v47 = a1 + 23;
    if ( !v46 )
      goto LABEL_84;
    do
    {
      if ( v46[4] < v18 )
      {
        v46 = (_QWORD *)v46[3];
      }
      else
      {
        v47 = v46;
        v46 = (_QWORD *)v46[2];
      }
    }
    while ( v46 );
    if ( v47 == v149 || v47[4] > v18 )
    {
LABEL_84:
      v121 = v44;
      v126 = v45;
      v132 = (__int64)v47;
      v58 = sub_22077B0(0x30u);
      *(_QWORD *)(v58 + 32) = v18;
      *(_QWORD *)(v58 + 40) = 0;
      v138 = v58;
      v59 = sub_2CE6C00(a1 + 22, v132, (unsigned __int64 *)(v58 + 32));
      if ( v60 )
      {
        v61 = v149 == v60 || v59 || v18 < v60[4];
        sub_220F040(v61, v138, v60, v149);
        ++a1[27];
        v47 = (_QWORD *)v138;
        v45 = v126;
        v44 = v121;
      }
      else
      {
        v134 = v59;
        j_j___libc_free_0(v138);
        v44 = v121;
        v45 = v126;
        v47 = v134;
      }
    }
    v47[5] = v45;
    if ( !v44 )
    {
      v165 = v45;
      v115 = (_QWORD *)sub_2CE6A60(a5, v166);
      v45 = v165;
      v44 = 0;
      *v115 = v165;
    }
    if ( (*(_DWORD *)(v18 + 4) & 0x7FFFFFF) != 0 )
    {
      v139 = (int)a5;
      v62 = v18;
      v133 = a6;
      v63 = 0;
      v64 = v45;
      v150 = v44;
      for ( i = sub_2CEAB40((_DWORD)a1, a2, **(_QWORD **)(v62 - 8), v62, v139, v133, v44);
            ;
            i = sub_2CEAB40((_DWORD)a1, a2, *(_QWORD *)(*(_QWORD *)(v62 - 8) + 32 * v63), v62, v139, v133, v150) )
      {
        v71 = i;
        v72 = *(_QWORD *)(*(_QWORD *)(v62 - 8) + 32LL * *(unsigned int *)(v62 + 72) + 8 * v63);
        v73 = *(_DWORD *)(v64 + 4) & 0x7FFFFFF;
        if ( v73 == *(_DWORD *)(v64 + 72) )
        {
          v122 = *(_QWORD *)(*(_QWORD *)(v62 - 8) + 32LL * *(unsigned int *)(v62 + 72) + 8 * v63);
          v127 = v71;
          sub_B48D90(v64);
          v72 = v122;
          v71 = v127;
          v73 = *(_DWORD *)(v64 + 4) & 0x7FFFFFF;
        }
        v66 = (v73 + 1) & 0x7FFFFFF;
        v67 = v66 | *(_DWORD *)(v64 + 4) & 0xF8000000;
        v68 = *(_QWORD *)(v64 - 8) + 32LL * (unsigned int)(v66 - 1);
        *(_DWORD *)(v64 + 4) = v67;
        if ( *(_QWORD *)v68 )
        {
          v69 = *(_QWORD *)(v68 + 8);
          **(_QWORD **)(v68 + 16) = v69;
          if ( v69 )
            *(_QWORD *)(v69 + 16) = *(_QWORD *)(v68 + 16);
        }
        *(_QWORD *)v68 = v71;
        if ( v71 )
        {
          v70 = *(_QWORD *)(v71 + 16);
          *(_QWORD *)(v68 + 8) = v70;
          if ( v70 )
            *(_QWORD *)(v70 + 16) = v68 + 8;
          *(_QWORD *)(v68 + 16) = v71 + 16;
          *(_QWORD *)(v71 + 16) = v68;
        }
        ++v63;
        *(_QWORD *)(*(_QWORD *)(v64 - 8)
                  + 32LL * *(unsigned int *)(v64 + 72)
                  + 8LL * ((*(_DWORD *)(v64 + 4) & 0x7FFFFFFu) - 1)) = v72;
        if ( (*(_DWORD *)(v62 + 4) & 0x7FFFFFFu) <= (unsigned int)v63 )
          break;
      }
      return v64;
    }
    else
    {
      return v45;
    }
  }
}
