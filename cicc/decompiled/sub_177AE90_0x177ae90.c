// Function: sub_177AE90
// Address: 0x177ae90
//
__int64 __fastcall sub_177AE90(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  __int64 v11; // r13
  unsigned __int16 v13; // ax
  char v14; // al
  __int64 v15; // r15
  __int64 v16; // rdi
  __int64 *v17; // rbx
  _QWORD *v18; // rax
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 v22; // rbx
  unsigned int v23; // eax
  int v24; // r8d
  int v25; // r9d
  unsigned int v26; // r15d
  unsigned int v27; // eax
  unsigned int v28; // ebx
  __int64 v29; // r15
  __int64 v30; // rsi
  __int64 v31; // rcx
  unsigned int v32; // esi
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 ***v35; // r15
  __int64 v36; // rbx
  const char *v37; // rax
  __int64 v38; // rcx
  const char *v39; // rdx
  __int64 **v40; // rdx
  char v41; // al
  char v42; // al
  __int64 v43; // rbx
  __int64 v44; // r12
  _QWORD *v45; // rax
  double v46; // xmm4_8
  double v47; // xmm5_8
  char v48; // al
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  char v52; // al
  __int64 v53; // r15
  unsigned int v54; // ebx
  unsigned __int8 *v55; // r13
  unsigned int i; // eax
  __int64 *v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rbx
  __int64 v60; // rax
  __int64 v61; // rbx
  _QWORD *v62; // rax
  _QWORD *v63; // r14
  __int64 v64; // rdi
  unsigned __int64 v65; // rsi
  __int64 v66; // rax
  __int64 **v67; // rsi
  _QWORD *v68; // rdi
  __int64 v69; // rdx
  bool v70; // zf
  __int64 v71; // rsi
  __int64 v72; // rsi
  __int64 v73; // rdx
  unsigned __int8 *v74; // rsi
  __int64 v75; // rdi
  __int64 v76; // rbx
  __int64 v77; // r12
  _QWORD *v78; // rax
  double v79; // xmm4_8
  double v80; // xmm5_8
  int v81; // eax
  __int64 v82; // rax
  int v83; // eax
  __int64 v84; // rbx
  unsigned int v85; // eax
  __int64 v86; // rsi
  unsigned __int64 v87; // r9
  _QWORD *v88; // rax
  __int64 v89; // rax
  unsigned int v90; // esi
  int v91; // eax
  __int64 v92; // rax
  __int64 v93; // r15
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 *v96; // rax
  __int64 v97; // rsi
  unsigned __int64 v98; // rcx
  __int64 v99; // rcx
  __int64 v100; // rdx
  unsigned __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rsi
  __int64 v104; // r15
  int v105; // ebx
  __int64 j; // r13
  __int64 *v107; // rax
  __int64 v108; // rdi
  __int64 v109; // rbx
  _QWORD *v110; // rax
  _QWORD *v111; // r15
  __int64 v112; // rdi
  unsigned __int64 v113; // rsi
  __int64 v114; // rax
  __int64 v115; // rsi
  __int64 v116; // rsi
  unsigned __int8 *v117; // rsi
  __int64 v118; // rdi
  __int64 v119; // rbx
  __int64 v120; // r12
  _QWORD *v121; // rax
  double v122; // xmm4_8
  double v123; // xmm5_8
  char v124; // al
  __int64 v125; // rax
  _QWORD *v126; // r15
  __int64 v127; // rbx
  __int64 v128; // rax
  unsigned __int8 *v129; // rax
  __int64 v130; // rbx
  __int64 v131; // r12
  _QWORD *v132; // rax
  _QWORD *v133; // r15
  __int64 v134; // rbx
  __int64 v135; // rax
  unsigned __int8 *v136; // rax
  __int64 v137; // rbx
  __int64 v138; // r12
  _QWORD *v139; // rax
  __int64 v140; // rax
  unsigned int v141; // r15d
  __int64 v142; // rax
  __int64 v143; // r15
  __int64 v144; // rdx
  __int64 v145; // rcx
  _QWORD *v146; // rax
  _QWORD *v147; // rbx
  __int64 v148; // rsi
  __int64 *v149; // r15
  __int64 v150; // rax
  __int64 v151; // rbx
  __int64 v152; // r12
  _QWORD *v153; // rax
  __int64 *v154; // rax
  __int64 v155; // rax
  unsigned int v156; // r15d
  __int64 v157; // rax
  __int64 v158; // rsi
  unsigned __int8 *v159; // rsi
  unsigned int v160; // r15d
  __int64 v161; // r14
  const char *v162; // rdx
  _QWORD *v163; // rax
  __int64 v164; // r12
  __int64 v165; // r14
  const char *v166; // rdx
  _QWORD *v167; // r12
  __int16 v168; // dx
  int v169; // eax
  __int16 v170; // si
  __int16 v171; // si
  int v172; // eax
  int v173; // r12d
  __int64 v174; // rax
  int v175; // r12d
  __int64 v176; // rax
  __int64 v177; // [rsp+10h] [rbp-150h]
  int v178; // [rsp+10h] [rbp-150h]
  __int64 *v179; // [rsp+18h] [rbp-148h]
  __int64 v180; // [rsp+20h] [rbp-140h]
  __int64 v181; // [rsp+28h] [rbp-138h]
  _BYTE *v182; // [rsp+28h] [rbp-138h]
  __int64 *v183; // [rsp+30h] [rbp-130h]
  unsigned __int64 v184; // [rsp+30h] [rbp-130h]
  __int64 v185; // [rsp+38h] [rbp-128h]
  __int64 v186; // [rsp+38h] [rbp-128h]
  _BYTE *v187; // [rsp+40h] [rbp-120h]
  _QWORD *v188; // [rsp+40h] [rbp-120h]
  __int64 v189; // [rsp+48h] [rbp-118h]
  unsigned __int8 *v190; // [rsp+48h] [rbp-118h]
  unsigned __int64 *v191; // [rsp+48h] [rbp-118h]
  __int64 v192; // [rsp+50h] [rbp-110h]
  unsigned int v193; // [rsp+58h] [rbp-108h]
  __int64 v194; // [rsp+58h] [rbp-108h]
  __int64 v195; // [rsp+70h] [rbp-F0h]
  __int64 v196; // [rsp+78h] [rbp-E8h]
  unsigned __int8 *v197; // [rsp+80h] [rbp-E0h]
  unsigned __int64 *v198; // [rsp+80h] [rbp-E0h]
  __int64 v199; // [rsp+80h] [rbp-E0h]
  unsigned __int64 v200; // [rsp+80h] [rbp-E0h]
  __int64 ***v201; // [rsp+80h] [rbp-E0h]
  __int64 ***v202; // [rsp+88h] [rbp-D8h]
  __int64 v203; // [rsp+88h] [rbp-D8h]
  __int64 v204; // [rsp+88h] [rbp-D8h]
  __int64 v205; // [rsp+88h] [rbp-D8h]
  __int64 v206; // [rsp+88h] [rbp-D8h]
  __int64 v207; // [rsp+88h] [rbp-D8h]
  __int64 v208; // [rsp+90h] [rbp-D0h]
  _QWORD **v209; // [rsp+90h] [rbp-D0h]
  __int64 v210; // [rsp+98h] [rbp-C8h]
  __int64 v211; // [rsp+98h] [rbp-C8h]
  unsigned int v212; // [rsp+ACh] [rbp-B4h] BYREF
  _QWORD *v213; // [rsp+B0h] [rbp-B0h] BYREF
  _QWORD *v214; // [rsp+B8h] [rbp-A8h] BYREF
  _QWORD v215[2]; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 *v216[2]; // [rsp+D0h] [rbp-90h] BYREF
  __int64 *v217[2]; // [rsp+E0h] [rbp-80h] BYREF
  const char *v218; // [rsp+F0h] [rbp-70h] BYREF
  const char *v219; // [rsp+F8h] [rbp-68h]
  __int64 v220; // [rsp+100h] [rbp-60h]
  __int64 *v221; // [rsp+110h] [rbp-50h] BYREF
  char *v222; // [rsp+118h] [rbp-48h]
  __int16 v223; // [rsp+120h] [rbp-40h]

  v10 = a2;
  v11 = a2;
  v210 = *(_QWORD *)(a2 - 24);
  v13 = *(_WORD *)(a2 + 18);
  if ( ((v13 >> 7) & 6) != 0 || (v13 & 1) != 0 || !*(_QWORD *)(a2 + 8) )
  {
LABEL_14:
    v15 = a1[333];
    goto LABEL_15;
  }
  v14 = sub_1649A90(v210);
  v15 = a1[333];
  if ( !v14 )
  {
    v16 = *(_QWORD *)(a2 + 8);
    if ( v16 )
    {
      if ( !*(_QWORD *)(v16 + 8) )
      {
        v17 = sub_1648700(v16);
        if ( (unsigned int)*((unsigned __int8 *)v17 + 16) - 60 <= 0xC )
        {
          if ( sub_15FB940((__int64)v17, v15)
            && !sub_1642F90(*v17, 128)
            && (!sub_15F32D0(a2) || (unsigned __int8)sub_1776710(*v17)) )
          {
            v223 = 257;
            v18 = sub_17779C0((__int64)a1, a2, *v17, (__int64)&v221);
            sub_164D160((__int64)v17, (__int64)v18, a3, a4, a5, a6, v19, v20, a9, a10);
            sub_170BC50((__int64)a1, (__int64)v17);
            return v10;
          }
          goto LABEL_14;
        }
      }
    }
  }
LABEL_15:
  v22 = a1[330];
  v208 = a1[332];
  v23 = sub_15AAE50(v15, *(_QWORD *)a2);
  v26 = sub_1AE99B0(v210, v23, v15, a2, v22, v208);
  v27 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1;
  if ( !v27 )
  {
    v32 = sub_15A9FE0(a1[333], *(_QWORD *)a2);
    if ( v26 <= v32 )
    {
      sub_15F8F50(v11, v32);
      goto LABEL_17;
    }
LABEL_44:
    sub_15F8F50(v11, v26);
    goto LABEL_17;
  }
  if ( v26 > v27 )
    goto LABEL_44;
LABEL_17:
  if ( *(_BYTE *)(v210 + 16) == 56 && (unsigned __int8)sub_1776BE0(a1, v210, v11, (unsigned int *)&v221, v24, v25) )
  {
    v93 = sub_15F4880(v210);
    v94 = sub_15A0680(
            **(_QWORD **)(v210 + 24 * ((unsigned int)v221 - (unsigned __int64)(*(_DWORD *)(v210 + 20) & 0xFFFFFFF))),
            0,
            0);
    if ( (*(_BYTE *)(v93 + 23) & 0x40) != 0 )
      v95 = *(_QWORD *)(v93 - 8);
    else
      v95 = v93 - 24LL * (*(_DWORD *)(v93 + 20) & 0xFFFFFFF);
    v96 = (__int64 *)(v95 + 24LL * (unsigned int)v221);
    if ( *v96 )
    {
      v97 = v96[1];
      v98 = v96[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v98 = v97;
      if ( v97 )
        *(_QWORD *)(v97 + 16) = *(_QWORD *)(v97 + 16) & 3LL | v98;
    }
    *v96 = v94;
    if ( v94 )
    {
      v99 = *(_QWORD *)(v94 + 8);
      v96[1] = v99;
      if ( v99 )
        *(_QWORD *)(v99 + 16) = (unsigned __int64)(v96 + 1) | *(_QWORD *)(v99 + 16) & 3LL;
      v96[2] = (v94 + 8) | v96[2] & 3;
      *(_QWORD *)(v94 + 8) = v96;
    }
    sub_15F2120(v93, v210);
    if ( *(_QWORD *)(v11 - 24) )
    {
      v100 = *(_QWORD *)(v11 - 16);
      v101 = *(_QWORD *)(v11 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v101 = v100;
      if ( v100 )
        *(_QWORD *)(v100 + 16) = *(_QWORD *)(v100 + 16) & 3LL | v101;
    }
    *(_QWORD *)(v11 - 24) = v93;
    v102 = *(_QWORD *)(v93 + 8);
    *(_QWORD *)(v11 - 16) = v102;
    if ( v102 )
      *(_QWORD *)(v102 + 16) = (v11 - 16) | *(_QWORD *)(v102 + 16) & 3LL;
    *(_QWORD *)(v11 - 8) = (v93 + 8) | *(_QWORD *)(v11 - 8) & 3LL;
    *(_QWORD *)(v93 + 8) = v11 - 24;
    sub_170B990(*a1, v93);
    return v10;
  }
  if ( sub_15F32D0(v11) )
    goto LABEL_26;
  v28 = *(unsigned __int16 *)(v11 + 18);
  if ( (v28 & 1) != 0 )
    goto LABEL_26;
  v209 = *(_QWORD ***)v11;
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)v11 + 8LL) - 13 > 1 )
    goto LABEL_26;
  v29 = a1[333];
  if ( 1 << (v28 >> 1) >> 1 > (unsigned int)sub_15A9FE0(v29, (__int64)v209) )
    goto LABEL_26;
  v30 = (__int64)v209;
  v31 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v30 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v82 = *(_QWORD *)(v30 + 32);
        v30 = *(_QWORD *)(v30 + 24);
        v31 *= v82;
        continue;
      case 1:
        v50 = 16;
        break;
      case 2:
        v50 = 32;
        break;
      case 3:
      case 9:
        v50 = 64;
        break;
      case 4:
        v50 = 80;
        break;
      case 5:
      case 6:
        v50 = 128;
        break;
      case 7:
        v204 = v31;
        v81 = sub_15A9520(v29, 0);
        v31 = v204;
        v50 = (unsigned int)(8 * v81);
        break;
      case 0xB:
        v50 = *(_DWORD *)(v30 + 8) >> 8;
        break;
      case 0xD:
        v207 = v31;
        v88 = (_QWORD *)sub_15A9930(v29, v30);
        v31 = v207;
        v50 = 8LL * *v88;
        break;
      case 0xE:
        v196 = v31;
        v84 = 1;
        v199 = *(_QWORD *)(v30 + 24);
        v206 = *(_QWORD *)(v30 + 32);
        v85 = sub_15A9FE0(v29, v199);
        v86 = v199;
        v31 = v196;
        v87 = v85;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v86 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v92 = *(_QWORD *)(v86 + 32);
              v86 = *(_QWORD *)(v86 + 24);
              v84 *= v92;
              continue;
            case 1:
              v89 = 16;
              goto LABEL_94;
            case 2:
              v89 = 32;
              goto LABEL_94;
            case 3:
            case 9:
              v89 = 64;
              goto LABEL_94;
            case 4:
              v89 = 80;
              goto LABEL_94;
            case 5:
            case 6:
              v89 = 128;
              goto LABEL_94;
            case 7:
              v90 = 0;
              v200 = v87;
              goto LABEL_99;
            case 0xB:
              v89 = *(_DWORD *)(v86 + 8) >> 8;
              goto LABEL_94;
            case 0xD:
              sub_15A9930(v29, v86);
              JUMPOUT(0x177B8F5);
            case 0xE:
              v195 = *(_QWORD *)(v86 + 24);
              sub_15A9FE0(v29, v195);
              sub_127FA20(v29, v195);
              JUMPOUT(0x177B8C3);
            case 0xF:
              v200 = v87;
              v90 = *(_DWORD *)(v86 + 8) >> 8;
LABEL_99:
              v91 = sub_15A9520(v29, v90);
              v87 = v200;
              v31 = v196;
              v89 = (unsigned int)(8 * v91);
LABEL_94:
              v50 = 8 * v206 * v87 * ((v87 + ((unsigned __int64)(v84 * v89 + 7) >> 3) - 1) / v87);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v205 = v31;
        v83 = sub_15A9520(v29, *(_DWORD *)(v30 + 8) >> 8);
        v31 = v205;
        v50 = (unsigned int)(8 * v83);
        break;
    }
    break;
  }
  if ( (unsigned __int64)(v50 * v31 + 7) >> 3 >= (unsigned int)dword_4FA27C0 )
    goto LABEL_26;
  v215[0] = sub_1649960(v11);
  v215[1] = v51;
  v52 = *((_BYTE *)v209 + 8);
  if ( v52 == 13 )
  {
    if ( !(unsigned __int8)sub_1776740(*(_QWORD *)(v11 - 24)) )
    {
      v193 = *((_DWORD *)v209 + 3);
      if ( v193 == 1 )
      {
        v223 = 259;
        v221 = (__int64 *)".unpack";
        v125 = sub_1643D80((__int64)v209, 0);
        v126 = sub_17779C0((__int64)a1, v11, v125, (__int64)&v221);
        v218 = 0;
        v219 = 0;
        v220 = 0;
        sub_14A8180(v11, (__int64 *)&v218, 0);
        sub_1626170((__int64)v126, (__int64 *)&v218);
        v127 = a1[1];
        v223 = 261;
        v221 = v215;
        LODWORD(v217[0]) = 0;
        v128 = sub_1599EF0(v209);
        v129 = sub_1777F70(v127, v128, (__int64)v126, v217, 1, (__int64 *)&v221);
        v130 = *(_QWORD *)(v11 + 8);
        v35 = (__int64 ***)v129;
        if ( v130 )
        {
          v131 = *a1;
          do
          {
            v132 = sub_1648700(v130);
            sub_170B990(v131, (__int64)v132);
            v130 = *(_QWORD *)(v130 + 8);
          }
          while ( v130 );
          goto LABEL_41;
        }
      }
      else
      {
        v53 = a1[333];
        v189 = sub_15A9930(v53, (__int64)v209);
        if ( (*(_BYTE *)(v189 + 12) & 1) == 0 )
        {
          v54 = 1 << (*(unsigned __int16 *)(v11 + 18) >> 1) >> 1;
          if ( !v54 )
            v54 = sub_15A9FE0(v53, (__int64)v209);
          v187 = *(_BYTE **)(v11 - 24);
          v185 = sub_1643350(*v209);
          v183 = (__int64 *)sub_159C470(v185, 0, 0);
          v212 = 0;
          v202 = (__int64 ***)sub_1599EF0(v209);
          if ( v193 )
          {
            v177 = v10;
            v192 = v11;
            v55 = (unsigned __int8 *)v202;
            v181 = v54;
            for ( i = 0; i < v193; v212 = i )
            {
              v216[0] = v183;
              v57 = (__int64 *)sub_159C470(v185, i, 0);
              v223 = 773;
              v216[1] = v57;
              v58 = a1[1];
              v221 = v215;
              v222 = ".elt";
              v197 = sub_1709730(v58, (__int64)v209, v187, v216, 2u, (__int64 *)&v221);
              v59 = *(_QWORD *)(v189 + 8LL * v212 + 16) | v181;
              v221 = v215;
              v60 = v59;
              v223 = 773;
              v61 = a1[1];
              v222 = ".unpack";
              v203 = v60 & -v60;
              v62 = sub_1648A60(64, 1u);
              v63 = v62;
              if ( v62 )
                sub_15F9210((__int64)v62, *(_QWORD *)(*(_QWORD *)v197 + 24LL), (__int64)v197, 0, 0, 0);
              v64 = *(_QWORD *)(v61 + 8);
              if ( v64 )
              {
                v198 = *(unsigned __int64 **)(v61 + 16);
                sub_157E9D0(v64 + 40, (__int64)v63);
                v65 = *v198;
                v66 = v63[3] & 7LL;
                v63[4] = v198;
                v65 &= 0xFFFFFFFFFFFFFFF8LL;
                v63[3] = v65 | v66;
                *(_QWORD *)(v65 + 8) = v63 + 3;
                *v198 = *v198 & 7 | (unsigned __int64)(v63 + 3);
              }
              v67 = &v221;
              v68 = v63;
              sub_164B780((__int64)v63, (__int64 *)&v221);
              v70 = *(_QWORD *)(v61 + 80) == 0;
              v213 = v63;
              if ( v70 )
                goto LABEL_200;
              (*(void (__fastcall **)(__int64, _QWORD **))(v61 + 88))(v61 + 64, &v213);
              v71 = *(_QWORD *)v61;
              if ( *(_QWORD *)v61 )
              {
                v218 = *(const char **)v61;
                sub_1623A60((__int64)&v218, v71, 2);
                v72 = v63[6];
                v73 = (__int64)(v63 + 6);
                if ( v72 )
                {
                  sub_161E7C0((__int64)(v63 + 6), v72);
                  v73 = (__int64)(v63 + 6);
                }
                v74 = (unsigned __int8 *)v218;
                v63[6] = v218;
                if ( v74 )
                  sub_1623210((__int64)&v218, v74, v73);
              }
              sub_15F8F50((__int64)v63, v203);
              v218 = 0;
              v219 = 0;
              v220 = 0;
              sub_14A8180(v192, (__int64 *)&v218, 0);
              sub_1626170((__int64)v63, (__int64 *)&v218);
              v75 = a1[1];
              v223 = 257;
              v55 = sub_1777F70(v75, (__int64)v55, (__int64)v63, &v212, 1, (__int64 *)&v221);
              i = v212 + 1;
            }
            v202 = (__int64 ***)v55;
            v10 = v177;
            v11 = v192;
          }
          v223 = 261;
          v221 = v215;
          sub_164B780((__int64)v202, (__int64 *)&v221);
          v76 = *(_QWORD *)(v11 + 8);
          if ( v76 )
          {
            v77 = *a1;
            do
            {
              v78 = sub_1648700(v76);
              sub_170B990(v77, (__int64)v78);
              v76 = *(_QWORD *)(v76 + 8);
            }
            while ( v76 );
            if ( v202 == (__int64 ***)v11 )
              v202 = (__int64 ***)sub_1599EF0(*v202);
            sub_164D160(v11, (__int64)v202, a3, a4, a5, a6, v79, v80, a9, a10);
            return v10;
          }
        }
      }
    }
LABEL_26:
    v33 = a1[329];
    LOBYTE(v217[0]) = 0;
    v34 = sub_13F9660((__int64 *)v11, (_QWORD *)v33, v217, qword_4F99140[20]);
    v35 = (__int64 ***)v34;
    if ( v34 )
    {
      if ( LOBYTE(v217[0]) )
        sub_1AEC340(v34, v11);
      v36 = a1[1];
      v37 = sub_1649960(v11);
      v38 = *(_QWORD *)v11;
      v218 = v37;
      v219 = v39;
      v221 = (__int64 *)&v218;
      v223 = 773;
      v222 = ".cast";
      v40 = *v35;
      if ( (__int64 **)v38 != *v35 )
      {
        v41 = *((_BYTE *)v40 + 8);
        if ( v41 == 16 )
          v41 = *(_BYTE *)(*v40[2] + 8);
        if ( v41 == 15 )
        {
          v124 = *(_BYTE *)(v38 + 8);
          if ( v124 == 16 )
            v124 = *(_BYTE *)(**(_QWORD **)(v38 + 16) + 8LL);
          if ( v124 != 11 )
            goto LABEL_37;
          v35 = (__int64 ***)sub_1708970(v36, 45, (__int64)v35, (__int64 **)v38, (__int64 *)&v221);
        }
        else
        {
          if ( v41 != 11 )
            goto LABEL_37;
          v42 = *(_BYTE *)(v38 + 8);
          if ( v42 == 16 )
            v42 = *(_BYTE *)(**(_QWORD **)(v38 + 16) + 8LL);
          if ( v42 != 15 )
          {
LABEL_37:
            v35 = (__int64 ***)sub_1708970(v36, 47, (__int64)v35, (__int64 **)v38, (__int64 *)&v221);
            goto LABEL_38;
          }
          v35 = (__int64 ***)sub_1708970(v36, 46, (__int64)v35, (__int64 **)v38, (__int64 *)&v221);
        }
      }
LABEL_38:
      v43 = *(_QWORD *)(v11 + 8);
      if ( v43 )
      {
        v44 = *a1;
        do
        {
          v45 = sub_1648700(v43);
          sub_170B990(v44, (__int64)v45);
          v43 = *(_QWORD *)(v43 + 8);
        }
        while ( v43 );
LABEL_41:
        if ( v35 == (__int64 ***)v11 )
          v35 = (__int64 ***)sub_1599EF0(*v35);
        sub_164D160(v11, (__int64)v35, a3, a4, a5, a6, v46, v47, a9, a10);
        return v10;
      }
      return 0;
    }
    if ( ((*(unsigned __int16 *)(v11 + 18) >> 7) & 6) != 0 || (*(_WORD *)(v11 + 18) & 1) != 0 )
      return 0;
    v48 = *(_BYTE *)(v210 + 16);
    if ( v48 == 56 )
    {
      v154 = *(__int64 **)(v210 - 24LL * (*(_DWORD *)(v210 + 20) & 0xFFFFFFF));
      if ( *((_BYTE *)v154 + 16) != 15 )
        goto LABEL_50;
      v155 = *v154;
      if ( *(_BYTE *)(v155 + 8) == 16 )
        v155 = **(_QWORD **)(v155 + 16);
      v156 = *(_DWORD *)(v155 + 8);
      v157 = sub_15F2060(v11);
      v33 = v156 >> 8;
      if ( !sub_15E4690(v157, v33) )
      {
LABEL_166:
        v143 = sub_1599EF0(*(__int64 ***)v11);
        v211 = sub_15A06D0(*(__int64 ***)v210, v33, v144, v145);
        v146 = sub_1648A60(64, 2u);
        v147 = v146;
        if ( v146 )
          sub_15F9660((__int64)v146, v143, v211, v11);
        v148 = *(_QWORD *)(v11 + 48);
        v149 = v147 + 6;
        v221 = (__int64 *)v148;
        if ( v148 )
        {
          sub_1623A60((__int64)&v221, v148, 2);
          if ( v149 == (__int64 *)&v221 )
          {
            if ( v221 )
              sub_161E7C0((__int64)(v147 + 6), (__int64)v221);
            goto LABEL_172;
          }
          v158 = v147[6];
          if ( !v158 )
          {
LABEL_185:
            v159 = (unsigned __int8 *)v221;
            v147[6] = v221;
            if ( v159 )
              sub_1623210((__int64)&v221, v159, (__int64)(v147 + 6));
LABEL_172:
            v150 = sub_1599EF0(*(__int64 ***)v11);
            v151 = *(_QWORD *)(v11 + 8);
            v35 = (__int64 ***)v150;
            if ( v151 )
            {
              v152 = *a1;
              do
              {
                v153 = sub_1648700(v151);
                sub_170B990(v152, (__int64)v153);
                v151 = *(_QWORD *)(v151 + 8);
              }
              while ( v151 );
              goto LABEL_41;
            }
            return 0;
          }
        }
        else
        {
          if ( v149 == (__int64 *)&v221 )
            goto LABEL_172;
          v158 = v147[6];
          if ( !v158 )
            goto LABEL_172;
        }
        sub_161E7C0((__int64)(v147 + 6), v158);
        goto LABEL_185;
      }
      v48 = *(_BYTE *)(v210 + 16);
    }
    if ( v48 == 9 )
      goto LABEL_166;
    if ( v48 == 15 )
    {
      v140 = **(_QWORD **)(v11 - 24);
      if ( *(_BYTE *)(v140 + 8) == 16 )
        v140 = **(_QWORD **)(v140 + 16);
      v141 = *(_DWORD *)(v140 + 8);
      v142 = sub_15F2060(v11);
      v33 = v141 >> 8;
      if ( !sub_15E4690(v142, v33) )
        goto LABEL_166;
    }
LABEL_50:
    v49 = *(_QWORD *)(v210 + 8);
    if ( v49 && !*(_QWORD *)(v49 + 8) && *(_BYTE *)(v210 + 16) == 79 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)v210 + 8LL) != 15 || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v210 + 24LL) + 8LL) != 13)
        && !byte_4FA26E0 )
      {
        v160 = 1 << (*(unsigned __int16 *)(v11 + 18) >> 1) >> 1;
        if ( (unsigned __int8)sub_13F86A0(*(_QWORD *)(v210 - 48), v160, a1[333], v210, 0) )
        {
          if ( (unsigned __int8)sub_13F86A0(*(_QWORD *)(v210 - 24), v160, a1[333], v210, 0) )
          {
            v161 = a1[1];
            v218 = sub_1649960(*(_QWORD *)(v210 - 48));
            v221 = (__int64 *)&v218;
            v222 = ".val";
            v219 = v162;
            v223 = 773;
            v163 = sub_1778150(v161, *(_QWORD *)(v210 - 48), (__int64 *)&v221);
            v164 = a1[1];
            v165 = (__int64)v163;
            v218 = sub_1649960(*(_QWORD *)(v210 - 24));
            v221 = (__int64 *)&v218;
            v222 = ".val";
            v219 = v166;
            v223 = 773;
            v167 = sub_1778150(v164, *(_QWORD *)(v210 - 24), (__int64 *)&v221);
            sub_15F8F50(v165, v160);
            v168 = *(_WORD *)(v165 + 18) & 0x8000;
            v169 = (*(unsigned __int16 *)(v11 + 18) >> 7) & 7;
            v170 = *(_WORD *)(v165 + 18) & 0x7C7F;
            *(_BYTE *)(v165 + 56) = *(_BYTE *)(v11 + 56);
            *(_WORD *)(v165 + 18) = v168 | v170 | ((_WORD)v169 << 7);
            sub_15F8F50((__int64)v167, v160);
            v171 = *((_WORD *)v167 + 9);
            v172 = (*(unsigned __int16 *)(v11 + 18) >> 7) & 7;
            *((_BYTE *)v167 + 56) = *(_BYTE *)(v11 + 56);
            *((_WORD *)v167 + 9) = v171 & 0x8000 | v171 & 0x7C7F | ((_WORD)v172 << 7);
            v223 = 257;
            return sub_14EDD70(*(_QWORD *)(v210 - 72), (_QWORD *)v165, (__int64)v167, (__int64)&v221, 0, 0);
          }
        }
      }
      if ( *(_BYTE *)(*(_QWORD *)(v210 - 48) + 16LL) == 15 )
      {
        v175 = sub_1776BC0(v11);
        v176 = sub_15F2060(v210);
        if ( !sub_15E4690(v176, v175) )
        {
          sub_1593B40((_QWORD *)(v11 - 24), *(_QWORD *)(v210 - 24));
          return v10;
        }
      }
      if ( *(_BYTE *)(*(_QWORD *)(v210 - 24) + 16LL) == 15 )
      {
        v173 = sub_1776BC0(v11);
        v174 = sub_15F2060(v210);
        if ( !sub_15E4690(v174, v173) )
        {
          sub_1593B40((_QWORD *)(v11 - 24), *(_QWORD *)(v210 - 48));
          return v10;
        }
      }
    }
    return 0;
  }
  if ( v52 != 14 )
    goto LABEL_26;
  v103 = (__int64)v209[3];
  v188 = v209[4];
  if ( v188 == (_QWORD *)1 )
  {
    v221 = (__int64 *)".unpack";
    v223 = 259;
    v133 = sub_17779C0((__int64)a1, v11, v103, (__int64)&v221);
    v218 = 0;
    v219 = 0;
    v220 = 0;
    sub_14A8180(v11, (__int64 *)&v218, 0);
    sub_1626170((__int64)v133, (__int64 *)&v218);
    v134 = a1[1];
    v221 = v215;
    v223 = 261;
    LODWORD(v217[0]) = 0;
    v135 = sub_1599EF0(v209);
    v136 = sub_1777F70(v134, v135, (__int64)v133, v217, 1, (__int64 *)&v221);
    v137 = *(_QWORD *)(v11 + 8);
    v35 = (__int64 ***)v136;
    if ( v137 )
    {
      v138 = *a1;
      do
      {
        v139 = sub_1648700(v137);
        sub_170B990(v138, (__int64)v139);
        v137 = *(_QWORD *)(v137 + 8);
      }
      while ( v137 );
      goto LABEL_41;
    }
    goto LABEL_26;
  }
  if ( (unsigned __int64)v188 > a1[342] )
    goto LABEL_26;
  v104 = a1[333];
  v184 = sub_12BE0A0(v104, v103);
  v105 = 1 << (*(unsigned __int16 *)(v11 + 18) >> 1) >> 1;
  if ( !v105 )
    v105 = sub_15A9FE0(v104, (__int64)v209);
  v182 = *(_BYTE **)(v11 - 24);
  v180 = sub_1643360(*v209);
  v179 = (__int64 *)sub_159C470(v180, 0, 0);
  v201 = (__int64 ***)sub_1599EF0(v209);
  if ( v188 )
  {
    v194 = 0;
    v178 = v105;
    v186 = v11;
    for ( j = 0; (_QWORD *)j != v188; ++j )
    {
      v217[0] = v179;
      v107 = (__int64 *)sub_159C470(v180, j, 0);
      v108 = a1[1];
      v217[1] = v107;
      v221 = v215;
      v223 = 773;
      v222 = ".elt";
      v190 = sub_1709730(v108, (__int64)v209, v182, v217, 2u, (__int64 *)&v221);
      v218 = (const char *)v215;
      v109 = a1[1];
      v219 = ".unpack";
      LOWORD(v220) = 773;
      v110 = sub_1648A60(64, 1u);
      v111 = v110;
      if ( v110 )
        sub_15F9210((__int64)v110, *(_QWORD *)(*(_QWORD *)v190 + 24LL), (__int64)v190, 0, 0, 0);
      v112 = *(_QWORD *)(v109 + 8);
      if ( v112 )
      {
        v191 = *(unsigned __int64 **)(v109 + 16);
        sub_157E9D0(v112 + 40, (__int64)v111);
        v113 = *v191;
        v114 = v111[3] & 7LL;
        v111[4] = v191;
        v113 &= 0xFFFFFFFFFFFFFFF8LL;
        v111[3] = v113 | v114;
        *(_QWORD *)(v113 + 8) = v111 + 3;
        *v191 = *v191 & 7 | (unsigned __int64)(v111 + 3);
      }
      v67 = (__int64 **)&v218;
      v68 = v111;
      sub_164B780((__int64)v111, (__int64 *)&v218);
      v70 = *(_QWORD *)(v109 + 80) == 0;
      v214 = v111;
      if ( v70 )
LABEL_200:
        sub_4263D6(v68, v67, v69);
      (*(void (__fastcall **)(__int64, _QWORD **))(v109 + 88))(v109 + 64, &v214);
      v115 = *(_QWORD *)v109;
      if ( *(_QWORD *)v109 )
      {
        v221 = *(__int64 **)v109;
        sub_1623A60((__int64)&v221, v115, 2);
        v116 = v111[6];
        if ( v116 )
          sub_161E7C0((__int64)(v111 + 6), v116);
        v117 = (unsigned __int8 *)v221;
        v111[6] = v221;
        if ( v117 )
          sub_1623210((__int64)&v221, v117, (__int64)(v111 + 6));
      }
      sub_15F8F50((__int64)v111, (v178 | v194) & -(v178 | v194));
      v218 = 0;
      v219 = 0;
      v220 = 0;
      sub_14A8180(v186, (__int64 *)&v218, 0);
      sub_1626170((__int64)v111, (__int64 *)&v218);
      v118 = a1[1];
      v223 = 257;
      LODWORD(v216[0]) = j;
      v194 += v184;
      v201 = (__int64 ***)sub_1777F70(v118, (__int64)v201, (__int64)v111, v216, 1, (__int64 *)&v221);
    }
    v11 = v186;
  }
  v223 = 261;
  v221 = v215;
  sub_164B780((__int64)v201, (__int64 *)&v221);
  v119 = *(_QWORD *)(v11 + 8);
  if ( !v119 )
    goto LABEL_26;
  v120 = *a1;
  do
  {
    v121 = sub_1648700(v119);
    sub_170B990(v120, (__int64)v121);
    v119 = *(_QWORD *)(v119 + 8);
  }
  while ( v119 );
  if ( v201 == (__int64 ***)v11 )
    v201 = (__int64 ***)sub_1599EF0(*v201);
  sub_164D160(v11, (__int64)v201, a3, a4, a5, a6, v122, v123, a9, a10);
  return v10;
}
