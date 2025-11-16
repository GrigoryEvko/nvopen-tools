// Function: sub_181B970
// Address: 0x181b970
//
__int64 __fastcall sub_181B970(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, double a6, double a7, double a8)
{
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  _BYTE *v15; // rdi
  _BYTE *v16; // rsi
  _BYTE *v17; // rax
  char v18; // dl
  unsigned int v19; // r14d
  _QWORD *v20; // r15
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 *v23; // rax
  _QWORD *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rcx
  int v28; // r8d
  unsigned int v29; // edx
  __int64 *v30; // rbx
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  _BYTE *v34; // r15
  __int64 v35; // rcx
  _QWORD *v36; // rax
  __int64 v37; // r10
  __int64 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r10
  __int64 v48; // r14
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r10
  __int64 v55; // r9
  bool v56; // al
  __int64 v57; // rax
  _QWORD *v58; // r14
  __int64 v59; // rax
  __int64 v60; // rsi
  unsigned int v61; // ecx
  _QWORD *v62; // rdx
  _QWORD *v63; // r8
  __int64 v64; // rax
  _BYTE *v65; // r15
  _BYTE *v66; // r14
  char *v67; // rax
  int v68; // r8d
  int v69; // r9d
  __int64 v70; // r14
  __int64 *v71; // r15
  __int64 v72; // rcx
  __int64 v73; // r13
  char *v74; // rdx
  char *v75; // rdi
  __int64 v76; // rax
  char *v77; // rax
  _BYTE *v78; // rsi
  _QWORD *v79; // rdi
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rdx
  __int64 *v83; // r15
  __int64 *v84; // rbx
  __int64 v85; // r13
  _QWORD *v86; // rax
  _QWORD *v87; // r14
  unsigned __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rax
  __int64 v91; // rcx
  _QWORD *v92; // rax
  __int64 v93; // r13
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rax
  _QWORD *v98; // rax
  unsigned __int64 v99; // rsi
  __int64 v100; // rax
  _QWORD *v101; // rax
  __int64 v102; // r13
  __int64 v103; // r13
  __int64 v104; // r15
  __int64 v105; // rax
  __int64 v106; // rcx
  __int64 v107; // r8
  __int64 v108; // r9
  __int64 v109; // rcx
  __int64 v110; // r8
  __int64 v111; // r9
  signed __int64 v112; // rsi
  char *v113; // r15
  int i; // edx
  int v115; // r9d
  __int64 v116; // [rsp+10h] [rbp-3C0h]
  size_t n; // [rsp+18h] [rbp-3B8h]
  __int64 *v118; // [rsp+20h] [rbp-3B0h]
  __int64 v119; // [rsp+28h] [rbp-3A8h]
  __int64 v120; // [rsp+30h] [rbp-3A0h]
  __int64 v121; // [rsp+38h] [rbp-398h]
  __int64 v122; // [rsp+48h] [rbp-388h]
  __int64 v123; // [rsp+50h] [rbp-380h]
  __int64 v124; // [rsp+60h] [rbp-370h]
  unsigned int v125; // [rsp+6Ch] [rbp-364h]
  __int64 v126; // [rsp+70h] [rbp-360h]
  __int64 *v127; // [rsp+78h] [rbp-358h]
  __int64 v128; // [rsp+88h] [rbp-348h]
  __int64 v129; // [rsp+90h] [rbp-340h]
  __int64 *v130; // [rsp+90h] [rbp-340h]
  __int64 v131; // [rsp+90h] [rbp-340h]
  __int64 v132; // [rsp+90h] [rbp-340h]
  __int64 v133; // [rsp+98h] [rbp-338h]
  __int64 v134; // [rsp+98h] [rbp-338h]
  char *v135; // [rsp+98h] [rbp-338h]
  __int64 v136; // [rsp+98h] [rbp-338h]
  __int64 v137; // [rsp+98h] [rbp-338h]
  _QWORD *v138; // [rsp+98h] [rbp-338h]
  __int64 v139; // [rsp+98h] [rbp-338h]
  unsigned __int64 *v140; // [rsp+98h] [rbp-338h]
  __int64 v141; // [rsp+98h] [rbp-338h]
  __int64 v142; // [rsp+A0h] [rbp-330h]
  __int64 v143; // [rsp+A0h] [rbp-330h]
  _BYTE *v144; // [rsp+A0h] [rbp-330h]
  _QWORD *v145; // [rsp+A8h] [rbp-328h]
  __int64 v146; // [rsp+A8h] [rbp-328h]
  __int64 v147; // [rsp+A8h] [rbp-328h]
  __int64 v148; // [rsp+A8h] [rbp-328h]
  __int64 v149; // [rsp+A8h] [rbp-328h]
  __int64 v150; // [rsp+A8h] [rbp-328h]
  __int64 j; // [rsp+A8h] [rbp-328h]
  __int64 v152; // [rsp+A8h] [rbp-328h]
  __int64 v153; // [rsp+A8h] [rbp-328h]
  __int64 v154[2]; // [rsp+B0h] [rbp-320h] BYREF
  __int16 v155; // [rsp+C0h] [rbp-310h]
  _BYTE *v156; // [rsp+D0h] [rbp-300h] BYREF
  __int64 v157; // [rsp+D8h] [rbp-2F8h]
  _BYTE v158[16]; // [rsp+E0h] [rbp-2F0h] BYREF
  _QWORD *v159; // [rsp+F0h] [rbp-2E0h] BYREF
  __int64 v160; // [rsp+F8h] [rbp-2D8h]
  __int64 *v161; // [rsp+100h] [rbp-2D0h]
  __int64 v162; // [rsp+108h] [rbp-2C8h]
  __int64 v163; // [rsp+110h] [rbp-2C0h]
  int v164; // [rsp+118h] [rbp-2B8h]
  __int64 v165; // [rsp+120h] [rbp-2B0h]
  __int64 v166; // [rsp+128h] [rbp-2A8h]
  _QWORD *v167; // [rsp+140h] [rbp-290h] BYREF
  __int64 v168; // [rsp+148h] [rbp-288h]
  __int64 *v169; // [rsp+150h] [rbp-280h]
  _QWORD *v170; // [rsp+190h] [rbp-240h] BYREF
  __int64 v171; // [rsp+198h] [rbp-238h]
  _QWORD v172[3]; // [rsp+1A0h] [rbp-230h] BYREF
  int v173; // [rsp+1B8h] [rbp-218h]
  __int64 v174; // [rsp+1C0h] [rbp-210h]
  __int64 v175; // [rsp+1C8h] [rbp-208h]

  if ( *(_BYTE *)(a2 + 16) == 53 )
  {
    v26 = *(unsigned int *)(a1 + 184);
    if ( (_DWORD)v26 )
    {
      v27 = *(_QWORD *)(a1 + 168);
      v28 = 1;
      v29 = (v26 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v30 = (__int64 *)(v27 + 16LL * v29);
      v31 = *v30;
      if ( a2 == *v30 )
      {
LABEL_23:
        if ( v30 != (__int64 *)(v27 + 16 * v26) )
        {
          sub_17CE510((__int64)&v170, a5, 0, 0, 0);
          LOWORD(v169) = 257;
          v22 = (__int64)sub_156E5B0((__int64 *)&v170, v30[1], (__int64)&v167);
          sub_17CD270((__int64 *)&v170);
          return v22;
        }
      }
      else
      {
        while ( v31 != -8 )
        {
          v29 = (v26 - 1) & (v28 + v29);
          v30 = (__int64 *)(v27 + 16LL * v29);
          v31 = *v30;
          if ( a2 == *v30 )
            goto LABEL_23;
          ++v28;
        }
      }
    }
  }
  v142 = a5;
  v156 = v158;
  v157 = 0x200000000LL;
  v13 = sub_15F2050(a5);
  v14 = sub_1632FA0(v13);
  sub_14AD470(a2, (__int64)&v156, v14, 0, 6u);
  v15 = v156;
  v16 = &v156[8 * (unsigned int)v157];
  if ( v156 == v16 )
  {
LABEL_14:
    v22 = *(_QWORD *)(*(_QWORD *)a1 + 200LL);
  }
  else
  {
    v17 = v156;
    while ( 1 )
    {
      v18 = *(_BYTE *)(*(_QWORD *)v17 + 16LL);
      if ( (v18 & 0xFB) != 0 && (v18 != 3 || (*(_BYTE *)(*(_QWORD *)v17 + 80LL) & 1) == 0) )
        break;
      v17 += 8;
      if ( v16 == v17 )
        goto LABEL_14;
    }
    v19 = 2 * a4;
    v20 = (_QWORD *)sub_18165C0(*(_QWORD *)a1, a2, v142, a6, a7, a8);
    if ( a3 == 1 )
    {
      v25 = sub_1648A60(64, 1u);
      v22 = (__int64)v25;
      if ( v25 )
        sub_15F9100((__int64)v25, v20, byte_3F871B3, v142);
      sub_15F8F50(v22, v19);
      v15 = v156;
      goto LABEL_15;
    }
    if ( a3 == 2 )
    {
      sub_17CE510((__int64)&v170, v142, 0, 0, 0);
      LOWORD(v169) = 257;
      v32 = sub_159C470(*(_QWORD *)(*(_QWORD *)a1 + 192LL), 1, 0);
      v33 = sub_12815B0((__int64 *)&v170, *(_QWORD *)(*(_QWORD *)a1 + 176LL), v20, v32, (__int64)&v167);
      LOWORD(v169) = 257;
      v145 = sub_156E5B0((__int64 *)&v170, v33, (__int64)&v167);
      sub_15F8F50((__int64)v145, v19);
      LOWORD(v161) = 257;
      v34 = sub_156E5B0((__int64 *)&v170, (__int64)v20, (__int64)&v159);
      sub_15F8F50((__int64)v34, v19);
      v22 = sub_181A560((__int64 *)a1, v34, (unsigned __int64)v145, v142);
      sub_17CD270((__int64 *)&v170);
      v15 = v156;
    }
    else
    {
      if ( a3 )
      {
        if ( *(_BYTE *)(a1 + 272) || (a3 & 3) != 0 )
        {
          sub_17CE510((__int64)&v170, v142, 0, 0, 0);
          LOWORD(v169) = 257;
          v21 = *(_QWORD *)a1;
          v159 = v20;
          v160 = sub_159C470(*(_QWORD *)(v21 + 192), a3, 0);
          v22 = sub_1285290(
                  (__int64 *)&v170,
                  *(_QWORD *)(**(_QWORD **)(*(_QWORD *)a1 + 344LL) + 24LL),
                  *(_QWORD *)(*(_QWORD *)a1 + 344LL),
                  (int)&v159,
                  2,
                  (__int64)&v167,
                  0);
          v167 = *(_QWORD **)(v22 + 56);
          v23 = (__int64 *)sub_16498A0(v22);
          v167 = (_QWORD *)sub_1563AB0((__int64 *)&v167, v23, 0, 58);
          *(_QWORD *)(v22 + 56) = v167;
          sub_17CD270((__int64 *)&v170);
          v15 = v156;
          goto LABEL_15;
        }
        v35 = *(_QWORD *)(a1 + 8);
        v133 = v142;
        LOWORD(v172[0]) = 257;
        v143 = v35;
        v146 = *(_QWORD *)(*(_QWORD *)a1 + 168LL);
        v36 = (_QWORD *)sub_22077B0(64);
        v128 = (__int64)v36;
        v37 = v133;
        if ( v36 )
        {
          sub_157FB60(v36, v146, (__int64)&v170, v143, 0);
          v37 = v133;
        }
        v147 = v37;
        v162 = sub_157E9C0(v128);
        LOWORD(v172[0]) = 257;
        v160 = v128;
        v161 = (__int64 *)(v128 + 40);
        v38 = *(_QWORD *)a1;
        v167 = v20;
        v159 = 0;
        v163 = 0;
        v164 = 0;
        v165 = 0;
        v166 = 0;
        v168 = sub_159C470(*(_QWORD *)(v38 + 192), a3, 0);
        v122 = sub_1285290(
                 (__int64 *)&v159,
                 *(_QWORD *)(**(_QWORD **)(*(_QWORD *)a1 + 344LL) + 24LL),
                 *(_QWORD *)(*(_QWORD *)a1 + 344LL),
                 (int)&v167,
                 2,
                 (__int64)&v170,
                 0);
        v170 = *(_QWORD **)(v122 + 56);
        v39 = (__int64 *)sub_16498A0(v122);
        v170 = (_QWORD *)sub_1563AB0((__int64 *)&v170, v39, 0, 58);
        *(_QWORD *)(v122 + 56) = v170;
        sub_17CE510((__int64)&v167, v147, 0, 0, 0);
        v40 = *(_QWORD *)a1;
        LOWORD(v172[0]) = 257;
        v41 = sub_1647230(*(_QWORD **)(v40 + 168), 0);
        v42 = sub_12AA3B0((__int64 *)&v167, 0x2Fu, (__int64)v20, v41, (__int64)&v170);
        v125 = v19;
        LOWORD(v172[0]) = 257;
        v144 = (_BYTE *)v42;
        v127 = sub_156E5B0((__int64 *)&v167, v42, (__int64)&v170);
        sub_15F8F50((__int64)v127, v19);
        v43 = *(_QWORD *)a1;
        LOWORD(v172[0]) = 257;
        v44 = sub_12AA3B0((__int64 *)&v167, 0x24u, (__int64)v127, *(_QWORD *)(v43 + 176), (__int64)&v170);
        v155 = 257;
        v119 = v44;
        v45 = sub_15A0680(*v127, 16, 0);
        if ( *((_BYTE *)v127 + 16) > 0x10u || *(_BYTE *)(v45 + 16) > 0x10u )
        {
          LOWORD(v172[0]) = 257;
          v48 = sub_15FB440(23, v127, v45, (__int64)&v170, 0);
          sub_18149C0(v48, v154, v168, v169);
          sub_12A86E0((__int64 *)&v167, v48);
          v47 = v147;
        }
        else
        {
          v46 = sub_15A2D50(v127, v45, 0, 0, a6, a7, a8);
          v47 = v147;
          v48 = v46;
        }
        v155 = 257;
        v148 = v47;
        v49 = sub_15A0680(*v127, 48, 0);
        v50 = v49;
        if ( *((_BYTE *)v127 + 16) > 0x10u || *(_BYTE *)(v49 + 16) > 0x10u )
        {
          v141 = v148;
          LOWORD(v172[0]) = 257;
          v153 = sub_15FB440(24, v127, v49, (__int64)&v170, 0);
          sub_18149C0(v153, v154, v168, v169);
          v50 = v153;
          sub_12A86E0((__int64 *)&v167, v153);
          v54 = v141;
          v55 = v153;
        }
        else
        {
          v51 = sub_15A2D80(v127, v49, 0, a6, a7, a8);
          v54 = v148;
          v55 = v51;
        }
        v155 = 257;
        if ( *(_BYTE *)(v55 + 16) <= 0x10u )
        {
          v149 = v54;
          v134 = v55;
          v56 = sub_1593BB0(v55, v50, v52, v53);
          v54 = v149;
          if ( v56 )
          {
LABEL_39:
            LOWORD(v172[0]) = 257;
            v150 = v54;
            v120 = sub_12AA0C0((__int64 *)&v167, 0x20u, v127, v48, (__int64)&v170);
            v58 = *(_QWORD **)(v150 + 40);
            LOWORD(v172[0]) = 257;
            v121 = (__int64)v58;
            v123 = sub_157FBF0(v58, (__int64 *)(v150 + 24), (__int64)&v170);
            v126 = a1 + 16;
            v59 = *(unsigned int *)(a1 + 64);
            if ( !(_DWORD)v59 )
              goto LABEL_77;
            v60 = *(_QWORD *)(a1 + 48);
            v61 = (v59 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
            v62 = (_QWORD *)(v60 + 16LL * v61);
            v63 = (_QWORD *)*v62;
            if ( v58 != (_QWORD *)*v62 )
            {
              for ( i = 1; ; i = v115 )
              {
                if ( v63 == (_QWORD *)-8LL )
                  goto LABEL_77;
                v115 = i + 1;
                v61 = (v59 - 1) & (i + v61);
                v62 = (_QWORD *)(v60 + 16LL * v61);
                v63 = (_QWORD *)*v62;
                if ( v58 == (_QWORD *)*v62 )
                  break;
              }
            }
            if ( v62 == (_QWORD *)(v60 + 16 * v59) || (v64 = v62[1]) == 0 )
            {
LABEL_77:
              v86 = sub_1648A60(56, 3u);
              v87 = v86;
              if ( v86 )
                sub_15F83E0((__int64)v86, v128, v128, v120, 0);
              v88 = sub_157EBA0(v121);
              sub_1AA6530(v88, v87, v89);
              sub_15D0040(v126, v128, v121);
              if ( a3 != 4 )
              {
                v124 = a3;
                for ( j = 4; j != v124; j += 4 )
                {
                  v90 = *(_QWORD *)a1;
                  v91 = *(_QWORD *)(a1 + 8);
                  LOWORD(v172[0]) = 257;
                  v131 = v91;
                  v136 = *(_QWORD *)(v90 + 168);
                  v92 = (_QWORD *)sub_22077B0(64);
                  v93 = (__int64)v92;
                  if ( v92 )
                    sub_157FB60(v92, v136, (__int64)&v170, v131, 0);
                  sub_15D0040(v126, v93, v87[5]);
                  v94 = sub_157E9C0(v93);
                  v170 = 0;
                  v172[1] = v94;
                  v172[0] = v93 + 40;
                  v95 = *(_QWORD *)a1;
                  v155 = 257;
                  v172[2] = 0;
                  v173 = 0;
                  v174 = 0;
                  v175 = 0;
                  v171 = v93;
                  v137 = sub_159C470(*(_QWORD *)(v95 + 192), 1, 0);
                  v96 = sub_1643360(*(_QWORD **)(*(_QWORD *)a1 + 168LL));
                  v97 = sub_12815B0((__int64 *)&v170, v96, v144, v137, (__int64)v154);
                  v155 = 257;
                  v144 = (_BYTE *)v97;
                  v138 = sub_156E5B0((__int64 *)&v170, v97, (__int64)v154);
                  sub_15F8F50((__int64)v138, v125);
                  v155 = 257;
                  v139 = sub_12AA0C0((__int64 *)&v170, 0x20u, v127, (__int64)v138, (__int64)v154);
                  sub_1593B40(v87 - 3, v93);
                  v155 = 257;
                  v98 = sub_1648A60(56, 3u);
                  v87 = v98;
                  if ( v98 )
                    sub_15F83E0((__int64)v98, v128, v128, v139, 0);
                  if ( v171 )
                  {
                    v140 = (unsigned __int64 *)v172[0];
                    sub_157E9D0(v171 + 40, (__int64)v87);
                    v99 = *v140;
                    v100 = v87[3] & 7LL;
                    v87[4] = v140;
                    v99 &= 0xFFFFFFFFFFFFFFF8LL;
                    v87[3] = v99 | v100;
                    *(_QWORD *)(v99 + 8) = v87 + 3;
                    *v140 = *v140 & 7 | (unsigned __int64)(v87 + 3);
                  }
                  sub_164B780((__int64)v87, v154);
                  sub_12A86E0((__int64 *)&v170, (__int64)v87);
                  sub_17CD270((__int64 *)&v170);
                }
              }
              sub_1593B40(v87 - 3, v123);
              LOWORD(v172[0]) = 257;
              v101 = sub_1648A60(56, 1u);
              v102 = (__int64)v101;
              if ( v101 )
                sub_15F8320((__int64)v101, v123, 0);
              sub_18149C0(v102, (__int64 *)&v170, v160, v161);
              sub_12A86E0((__int64 *)&v159, v102);
              v103 = *(_QWORD *)(v123 + 48);
              LOWORD(v172[0]) = 257;
              if ( v103 )
                v103 -= 24;
              v104 = *(_QWORD *)(*(_QWORD *)a1 + 176LL);
              v105 = sub_1648B60(64);
              v22 = v105;
              if ( v105 )
              {
                sub_15F1EA0(v105, v104, 53, 0, 0, v103);
                *(_DWORD *)(v22 + 56) = 2;
                sub_164B780(v22, (__int64 *)&v170);
                sub_1648880(v22, *(_DWORD *)(v22 + 56), 1);
              }
              sub_1704F80(v22, v122, v128, v106, v107, v108);
              sub_1704F80(v22, v119, v87[5], v109, v110, v111);
              sub_17CD270((__int64 *)&v167);
              sub_17CD270((__int64 *)&v159);
              v15 = v156;
              goto LABEL_15;
            }
            v65 = *(_BYTE **)(v64 + 32);
            v66 = *(_BYTE **)(v64 + 24);
            n = v65 - v66;
            if ( (unsigned __int64)(v65 - v66) > 0x7FFFFFFFFFFFFFF8LL )
              sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
            if ( !n )
            {
              if ( v66 == v65 )
              {
                sub_15D0040(v126, v123, v121);
              }
              else
              {
                memcpy(0, v66, 0);
                sub_15D0040(v126, v123, v121);
                j_j___libc_free_0(0, 0);
              }
              goto LABEL_77;
            }
            v67 = (char *)sub_22077B0(n);
            v118 = (__int64 *)v67;
            v135 = &v67[n];
            if ( v66 == v65 )
            {
              v70 = sub_15D0040(v126, v123, v121);
              if ( v135 == (char *)v118 )
                goto LABEL_76;
            }
            else
            {
              v113 = v67;
              memcpy(v67, v66, n);
              v70 = sub_15D0040(v126, v123, v121);
              if ( v135 == v113 )
              {
LABEL_76:
                j_j___libc_free_0(v118, n);
                goto LABEL_77;
              }
            }
            v71 = v118;
            v116 = a3;
            while ( 1 )
            {
              v72 = *v71;
              *(_BYTE *)(a1 + 88) = 0;
              v73 = *(_QWORD *)(v72 + 8);
              if ( v73 != v70 )
                break;
LABEL_74:
              if ( v135 == (char *)++v71 )
              {
                a3 = v116;
                if ( v118 )
                  goto LABEL_76;
                goto LABEL_77;
              }
            }
            v74 = *(char **)(v73 + 32);
            v75 = *(char **)(v73 + 24);
            v76 = (v74 - v75) >> 5;
            if ( v76 > 0 )
            {
              v77 = &v75[32 * v76];
              while ( v72 != *(_QWORD *)v75 )
              {
                if ( v72 == *((_QWORD *)v75 + 1) )
                {
                  v75 += 8;
                  goto LABEL_56;
                }
                if ( v72 == *((_QWORD *)v75 + 2) )
                {
                  v75 += 16;
                  goto LABEL_56;
                }
                if ( v72 == *((_QWORD *)v75 + 3) )
                {
                  v75 += 24;
                  goto LABEL_56;
                }
                v75 += 32;
                if ( v75 == v77 )
                  goto LABEL_99;
              }
              goto LABEL_56;
            }
            v77 = *(char **)(v73 + 24);
LABEL_99:
            v112 = v74 - v77;
            if ( v74 - v77 != 16 )
            {
              if ( v112 != 24 )
              {
                v75 = *(char **)(v73 + 32);
                if ( v112 == 8 )
                {
LABEL_102:
                  if ( v72 != *(_QWORD *)v77 )
                    v77 = *(char **)(v73 + 32);
                  v75 = v77;
                  goto LABEL_56;
                }
                goto LABEL_56;
              }
              v75 = v77;
              if ( v72 == *(_QWORD *)v77 )
                goto LABEL_56;
              v77 += 8;
            }
            v75 = v77;
            if ( v72 != *(_QWORD *)v77 )
            {
              v77 += 8;
              goto LABEL_102;
            }
LABEL_56:
            if ( v75 + 8 != v74 )
            {
              v129 = v72;
              memmove(v75, v75 + 8, v74 - (v75 + 8));
              v74 = *(char **)(v73 + 32);
              v72 = v129;
            }
            *(_QWORD *)(v73 + 32) = v74 - 8;
            *(_QWORD *)(v72 + 8) = v70;
            v170 = (_QWORD *)v72;
            v78 = *(_BYTE **)(v70 + 32);
            if ( v78 == *(_BYTE **)(v70 + 40) )
            {
              v132 = v72;
              sub_15CE310(v70 + 24, v78, &v170);
              v72 = v132;
            }
            else
            {
              if ( v78 )
              {
                *(_QWORD *)v78 = v72;
                v78 = *(_BYTE **)(v70 + 32);
              }
              *(_QWORD *)(v70 + 32) = v78 + 8;
            }
            if ( *(_DWORD *)(v72 + 16) != *(_DWORD *)(*(_QWORD *)(v72 + 8) + 16LL) + 1 )
            {
              v172[0] = v72;
              v170 = v172;
              v79 = v172;
              v130 = v71;
              v171 = 0x4000000001LL;
              LODWORD(v80) = 1;
              do
              {
                v81 = (unsigned int)v80;
                v80 = (unsigned int)(v80 - 1);
                v82 = v79[v81 - 1];
                LODWORD(v171) = v80;
                v83 = *(__int64 **)(v82 + 32);
                v84 = *(__int64 **)(v82 + 24);
                *(_DWORD *)(v82 + 16) = *(_DWORD *)(*(_QWORD *)(v82 + 8) + 16LL) + 1;
                if ( v84 != v83 )
                {
                  do
                  {
                    v85 = *v84;
                    if ( *(_DWORD *)(*v84 + 16) != *(_DWORD *)(*(_QWORD *)(*v84 + 8) + 16LL) + 1 )
                    {
                      if ( HIDWORD(v171) <= (unsigned int)v80 )
                      {
                        sub_16CD150((__int64)&v170, v172, 0, 8, v68, v69);
                        v80 = (unsigned int)v171;
                      }
                      v170[v80] = v85;
                      v80 = (unsigned int)(v171 + 1);
                      LODWORD(v171) = v171 + 1;
                    }
                    ++v84;
                  }
                  while ( v83 != v84 );
                  v79 = v170;
                }
              }
              while ( (_DWORD)v80 );
              v71 = v130;
              if ( v79 != v172 )
                _libc_free((unsigned __int64)v79);
            }
            goto LABEL_74;
          }
          v55 = v134;
          if ( *(_BYTE *)(v48 + 16) <= 0x10u )
          {
            v57 = sub_15A2D10((__int64 *)v48, v134, a6, a7, a8);
            v54 = v149;
            v48 = v57;
            goto LABEL_39;
          }
        }
        v152 = v54;
        LOWORD(v172[0]) = 257;
        v48 = sub_15FB440(27, (__int64 *)v48, v55, (__int64)&v170, 0);
        sub_18149C0(v48, v154, v168, v169);
        sub_12A86E0((__int64 *)&v167, v48);
        v54 = v152;
        goto LABEL_39;
      }
      v15 = v156;
      v22 = *(_QWORD *)(*(_QWORD *)a1 + 200LL);
    }
  }
LABEL_15:
  if ( v15 != v158 )
    _libc_free((unsigned __int64)v15);
  return v22;
}
