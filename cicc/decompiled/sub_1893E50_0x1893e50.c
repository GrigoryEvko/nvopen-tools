// Function: sub_1893E50
// Address: 0x1893e50
//
void __fastcall sub_1893E50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  unsigned int v11; // r14d
  __int64 v12; // rax
  unsigned __int64 *v13; // r15
  unsigned __int64 *v14; // rbx
  int v15; // eax
  unsigned __int64 v16; // r12
  __int64 v17; // rax
  unsigned __int64 *v18; // rdx
  __int64 *v19; // rdx
  unsigned __int64 v20; // rcx
  __int64 *v21; // rax
  __int64 v22; // r8
  unsigned __int64 *v23; // r14
  unsigned __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 *v26; // rdi
  __int64 v27; // rcx
  _QWORD *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // r9
  __int64 v33; // rbx
  __int64 v34; // r14
  __int64 v35; // r15
  __int64 v36; // r8
  int v37; // r9d
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // rbx
  __int16 v41; // dx
  __int64 *v42; // rax
  __int64 v43; // r14
  _QWORD *v44; // rax
  _QWORD *v45; // r15
  unsigned __int64 *v46; // r14
  __int64 v47; // rax
  unsigned __int64 v48; // rcx
  __int64 v49; // rsi
  __int64 v50; // rdx
  unsigned __int8 *v51; // rsi
  __int64 v52; // rax
  __int64 *v53; // r8
  __int64 *v54; // rbx
  unsigned __int8 *v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rbx
  __int64 v58; // r14
  _BYTE *v59; // rax
  __int64 v60; // rax
  unsigned __int64 v61; // r15
  __int64 v62; // rbx
  _QWORD *v63; // rax
  _QWORD *v64; // r8
  __int64 v65; // rax
  __int64 *v66; // rdx
  __int64 *i; // r9
  unsigned __int64 v68; // rcx
  __int64 *v69; // rax
  _BOOL4 v70; // r11d
  __int64 v71; // rax
  __int64 *j; // r9
  unsigned __int64 v73; // rcx
  __int64 *v74; // rax
  _BOOL4 v75; // r11d
  unsigned __int64 v76; // rcx
  __int64 *v77; // rax
  _BOOL4 v78; // r8d
  __int64 v79; // rax
  unsigned __int8 *v80; // rsi
  unsigned __int8 *v81; // r15
  unsigned __int8 *v82; // rdi
  signed __int64 v83; // rsi
  unsigned __int64 v84; // rcx
  __int64 *v85; // rax
  _BOOL4 v86; // r8d
  __int64 v87; // rax
  double v88; // xmm4_8
  double v89; // xmm5_8
  __int64 v90; // r14
  _QWORD *v91; // rax
  __int64 v92; // rax
  __int64 v93; // r15
  __int64 v94; // rbx
  __int64 v95; // r13
  char v96; // r12
  __int64 v97; // r12
  __int64 v98; // rsi
  unsigned __int8 *v99; // rsi
  __int64 v100; // rsi
  unsigned __int8 *v101; // rsi
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  _BOOL4 v105; // [rsp+8h] [rbp-1B8h]
  _BOOL4 v106; // [rsp+8h] [rbp-1B8h]
  _QWORD *v107; // [rsp+8h] [rbp-1B8h]
  _QWORD *v108; // [rsp+8h] [rbp-1B8h]
  __int64 *v109; // [rsp+10h] [rbp-1B0h]
  __int64 *v110; // [rsp+10h] [rbp-1B0h]
  _BOOL4 v111; // [rsp+10h] [rbp-1B0h]
  __int64 *v112; // [rsp+10h] [rbp-1B0h]
  __int64 *v113; // [rsp+10h] [rbp-1B0h]
  _QWORD *v114; // [rsp+18h] [rbp-1A8h]
  _QWORD *v115; // [rsp+18h] [rbp-1A8h]
  __int64 *v116; // [rsp+18h] [rbp-1A8h]
  __int64 *v117; // [rsp+18h] [rbp-1A8h]
  __int64 *v118; // [rsp+18h] [rbp-1A8h]
  __int64 *v119; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 *v121; // [rsp+28h] [rbp-198h]
  _BOOL4 v122; // [rsp+28h] [rbp-198h]
  __int64 v123; // [rsp+28h] [rbp-198h]
  unsigned __int64 *v124; // [rsp+30h] [rbp-190h]
  __int64 v125; // [rsp+38h] [rbp-188h]
  __int64 v126; // [rsp+38h] [rbp-188h]
  unsigned __int64 *v127; // [rsp+40h] [rbp-180h]
  __int64 *v128; // [rsp+40h] [rbp-180h]
  __int64 *v129; // [rsp+40h] [rbp-180h]
  __int64 v130; // [rsp+48h] [rbp-178h]
  _QWORD *v132; // [rsp+58h] [rbp-168h]
  __int64 v133; // [rsp+58h] [rbp-168h]
  __int64 v134; // [rsp+58h] [rbp-168h]
  __int64 v135; // [rsp+58h] [rbp-168h]
  __int64 *v136; // [rsp+58h] [rbp-168h]
  unsigned __int8 *v137; // [rsp+60h] [rbp-160h] BYREF
  unsigned __int8 *v138; // [rsp+68h] [rbp-158h] BYREF
  _BYTE *v139; // [rsp+70h] [rbp-150h] BYREF
  _BYTE *v140; // [rsp+78h] [rbp-148h]
  _BYTE *v141; // [rsp+80h] [rbp-140h]
  unsigned __int8 *v142; // [rsp+90h] [rbp-130h] BYREF
  unsigned __int8 *v143; // [rsp+98h] [rbp-128h]
  unsigned __int8 *v144; // [rsp+A0h] [rbp-120h]
  unsigned __int8 *v145; // [rsp+B0h] [rbp-110h] BYREF
  _QWORD *v146; // [rsp+B8h] [rbp-108h]
  unsigned __int64 *v147; // [rsp+C0h] [rbp-100h]
  __int64 v148; // [rsp+C8h] [rbp-F8h]
  __int64 v149; // [rsp+D0h] [rbp-F0h]
  int v150; // [rsp+D8h] [rbp-E8h]
  __int64 v151; // [rsp+E0h] [rbp-E0h]
  __int64 v152; // [rsp+E8h] [rbp-D8h]
  __int64 **v153; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v154; // [rsp+108h] [rbp-B8h] BYREF
  __int64 *v155; // [rsp+110h] [rbp-B0h] BYREF
  __int64 *v156; // [rsp+118h] [rbp-A8h]
  __int64 *v157; // [rsp+120h] [rbp-A0h]
  __int64 v158; // [rsp+128h] [rbp-98h]

  v11 = (unsigned __int8)byte_4FAC680;
  v130 = a2;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  if ( byte_4FAC680 )
  {
    v12 = *(_QWORD *)(a3 + 80);
    v125 = v12;
    if ( !v12 )
    {
      LODWORD(v154) = 0;
      v155 = 0;
      v156 = &v154;
      v157 = &v154;
      v158 = 0;
      BUG();
    }
    v13 = (unsigned __int64 *)(v12 + 16);
    LODWORD(v154) = 0;
    v132 = (_QWORD *)(v12 - 24);
    v155 = 0;
    v156 = &v154;
    v157 = &v154;
    v158 = 0;
    v14 = *(unsigned __int64 **)(v12 + 24);
    v124 = (unsigned __int64 *)(v12 + 16);
    if ( v14 == (unsigned __int64 *)(v12 + 16) )
    {
      v22 = 0;
LABEL_38:
      sub_1890AA0(v22);
      v28 = (_QWORD *)sub_157EBA0((__int64)v132);
      sub_15F20C0(v28);
      v126 = 0;
      goto LABEL_39;
    }
    while ( 1 )
    {
      if ( !v14 )
        BUG();
      v15 = *((unsigned __int8 *)v14 - 8);
      v16 = (unsigned __int64)(v14 - 3);
      if ( (_BYTE)v15 != 78 )
      {
        v18 = 0;
        if ( (unsigned int)(v15 - 25) <= 9 )
          v18 = v14 - 3;
        goto LABEL_11;
      }
      v17 = *(v14 - 6);
      if ( *(_BYTE *)(v17 + 16) || (*(_BYTE *)(v17 + 33) & 0x20) == 0 )
      {
LABEL_157:
        v18 = 0;
LABEL_11:
        if ( v18 != (unsigned __int64 *)sub_157EBA0((__int64)v132) )
          goto LABEL_22;
        v19 = v155;
        if ( !v155 )
          goto LABEL_180;
        while ( 1 )
        {
          v20 = v19[4];
          v21 = (__int64 *)v19[3];
          a2 = 0;
          if ( v16 < v20 )
          {
            v21 = (__int64 *)v19[2];
            a2 = v11;
          }
          if ( !v21 )
            break;
          v19 = v21;
        }
        if ( (_BYTE)a2 )
          goto LABEL_153;
        if ( v20 < v16 )
          goto LABEL_148;
        goto LABEL_22;
      }
      if ( *(_DWORD *)(v17 + 36) == 38 )
        break;
      if ( (*(_BYTE *)(v17 + 33) & 0x20) == 0 )
        goto LABEL_157;
      v18 = 0;
      if ( *(_DWORD *)(v17 + 36) != 36 )
        goto LABEL_11;
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v16 + 24 * (1LL - (*((_DWORD *)v14 - 1) & 0xFFFFFFF))) + 24LL) + 32LL) )
      {
        a2 = 1;
        v60 = sub_1601A30((__int64)(v14 - 3), 1);
        if ( v60 )
        {
          if ( *(_BYTE *)(v60 + 16) == 53 && *(_QWORD *)(v60 + 8) )
          {
            v127 = v13;
            v61 = v60;
            v121 = v14;
            v62 = *(_QWORD *)(v60 + 8);
            while ( 1 )
            {
              v63 = sub_1648700(v62);
              v64 = v63;
              if ( *((_BYTE *)v63 + 16) != 55 )
                goto LABEL_88;
              v65 = *(v63 - 6);
              if ( !v65 || *(_BYTE *)(v65 + 16) != 17 )
                goto LABEL_88;
              v66 = v155;
              if ( v155 )
              {
                for ( i = v155; ; i = v69 )
                {
                  v68 = i[4];
                  v69 = (__int64 *)i[3];
                  a2 = 0;
                  if ( v61 < v68 )
                  {
                    v69 = (__int64 *)i[2];
                    a2 = v11;
                  }
                  if ( !v69 )
                    break;
                }
                if ( !(_BYTE)a2 )
                {
                  if ( v68 >= v61 )
                    goto LABEL_104;
LABEL_100:
                  v70 = 1;
                  if ( i != &v154 )
                    v70 = v61 < i[4];
LABEL_102:
                  v114 = v64;
                  v105 = v70;
                  v109 = i;
                  v71 = sub_22077B0(40);
                  *(_QWORD *)(v71 + 32) = v61;
                  a2 = v71;
                  sub_220F040(v105, v71, v109, &v154);
                  ++v158;
                  v66 = v155;
                  v64 = v114;
                  goto LABEL_103;
                }
                if ( v156 == i )
                  goto LABEL_100;
              }
              else
              {
                i = &v154;
                if ( v156 == &v154 )
                {
                  v70 = 1;
                  goto LABEL_102;
                }
              }
              v108 = v64;
              v113 = v155;
              v119 = i;
              v104 = sub_220EF80(i);
              i = v119;
              v66 = v113;
              v64 = v108;
              if ( *(_QWORD *)(v104 + 32) < v61 )
                goto LABEL_100;
LABEL_103:
              if ( v66 )
              {
LABEL_104:
                for ( j = v66; ; j = v74 )
                {
                  v73 = j[4];
                  v74 = (__int64 *)j[3];
                  a2 = 0;
                  if ( (unsigned __int64)v64 < v73 )
                  {
                    v74 = (__int64 *)j[2];
                    a2 = v11;
                  }
                  if ( !v74 )
                    break;
                }
                if ( !(_BYTE)a2 )
                {
                  if ( (unsigned __int64)v64 <= v73 )
                    goto LABEL_117;
LABEL_111:
                  v75 = 1;
                  if ( j != &v154 )
                    v75 = (unsigned __int64)v64 < j[4];
LABEL_113:
                  v106 = v75;
                  v110 = j;
                  v115 = v64;
                  a2 = sub_22077B0(40);
                  *(_QWORD *)(a2 + 32) = v115;
                  sub_220F040(v106, a2, v110, &v154);
                  ++v158;
                  v66 = v155;
                  goto LABEL_114;
                }
                if ( v156 == j )
                  goto LABEL_111;
                goto LABEL_192;
              }
              j = &v154;
              if ( v156 == &v154 )
              {
                v75 = 1;
                goto LABEL_113;
              }
LABEL_192:
              v107 = v64;
              v112 = v66;
              v118 = j;
              v103 = sub_220EF80(j);
              v64 = v107;
              j = v118;
              v66 = v112;
              if ( *(_QWORD *)(v103 + 32) < (unsigned __int64)v107 )
                goto LABEL_111;
LABEL_114:
              if ( !v66 )
              {
                v66 = &v154;
                if ( v156 == &v154 )
                {
                  v78 = 1;
                }
                else
                {
LABEL_189:
                  v117 = v66;
                  v102 = sub_220EF80(v66);
                  v66 = v117;
                  if ( *(_QWORD *)(v102 + 32) >= v16 )
                    goto LABEL_88;
LABEL_122:
                  v78 = 1;
                  if ( v66 != &v154 )
                    v78 = v16 < v66[4];
                }
                v111 = v78;
                v116 = v66;
                v79 = sub_22077B0(40);
                *(_QWORD *)(v79 + 32) = v16;
                a2 = v79;
                sub_220F040(v111, v79, v116, &v154);
                ++v158;
                goto LABEL_88;
              }
              while ( 1 )
              {
LABEL_117:
                v76 = v66[4];
                v77 = (__int64 *)v66[3];
                a2 = 0;
                if ( v16 < v76 )
                {
                  v77 = (__int64 *)v66[2];
                  a2 = v11;
                }
                if ( !v77 )
                  break;
                v66 = v77;
              }
              if ( (_BYTE)a2 )
              {
                if ( v156 == v66 )
                  goto LABEL_122;
                goto LABEL_189;
              }
              if ( v16 > v76 )
                goto LABEL_122;
LABEL_88:
              v62 = *(_QWORD *)(v62 + 8);
              if ( !v62 )
              {
                v13 = v127;
                v14 = v121;
                break;
              }
            }
          }
        }
      }
LABEL_22:
      v14 = (unsigned __int64 *)v14[1];
      if ( v13 == v14 )
      {
        v22 = (__int64)v155;
        v23 = *(unsigned __int64 **)(v125 + 24);
        if ( v124 != v23 )
        {
          do
          {
            v24 = (unsigned __int64)(v23 - 3);
            if ( !v23 )
              v24 = 0;
            v145 = (unsigned __int8 *)v24;
            if ( !v22 )
              goto LABEL_33;
            v25 = (__int64 *)v22;
            v26 = &v154;
            do
            {
              while ( 1 )
              {
                a2 = v25[2];
                v27 = v25[3];
                if ( v25[4] >= v24 )
                  break;
                v25 = (__int64 *)v25[3];
                if ( !v27 )
                  goto LABEL_31;
              }
              v26 = v25;
              v25 = (__int64 *)v25[2];
            }
            while ( a2 );
LABEL_31:
            if ( v26 == &v154 || v26[4] > v24 )
            {
LABEL_33:
              a2 = (__int64)v140;
              if ( v140 == v141 )
              {
                sub_170B610((__int64)&v139, v140, &v145);
                v22 = (__int64)v155;
              }
              else
              {
                if ( v140 )
                {
                  *(_QWORD *)v140 = v24;
                  a2 = (__int64)v140;
                  v22 = (__int64)v155;
                }
                a2 += 8;
                v140 = (_BYTE *)a2;
              }
            }
            v23 = (unsigned __int64 *)v23[1];
          }
          while ( v14 != v23 );
        }
        goto LABEL_38;
      }
    }
    if ( !*(_WORD *)(*(_QWORD *)(v14[3 * (1LL - (*((_DWORD *)v14 - 1) & 0xFFFFFFF)) - 3] + 24) + 32LL) )
      goto LABEL_22;
    v19 = v155;
    if ( v155 )
    {
      while ( 1 )
      {
        v84 = v19[4];
        v85 = (__int64 *)v19[3];
        a2 = 0;
        if ( v16 < v84 )
        {
          v85 = (__int64 *)v19[2];
          a2 = v11;
        }
        if ( !v85 )
          break;
        v19 = v85;
      }
      if ( !(_BYTE)a2 )
      {
        if ( v16 <= v84 )
          goto LABEL_22;
LABEL_148:
        v86 = 1;
        if ( v19 == &v154 )
          goto LABEL_149;
        goto LABEL_156;
      }
LABEL_153:
      if ( v156 == v19 )
        goto LABEL_148;
    }
    else
    {
LABEL_180:
      v19 = &v154;
      if ( v156 == &v154 )
      {
        v86 = 1;
        goto LABEL_149;
      }
    }
    v129 = v19;
    v92 = sub_220EF80(v19);
    v19 = v129;
    if ( *(_QWORD *)(v92 + 32) >= v16 )
      goto LABEL_22;
    v86 = 1;
    if ( v129 == &v154 )
    {
LABEL_149:
      v122 = v86;
      v128 = v19;
      v87 = sub_22077B0(40);
      *(_QWORD *)(v87 + 32) = v16;
      a2 = v87;
      sub_220F040(v122, v87, v128, &v154);
      ++v158;
      goto LABEL_22;
    }
LABEL_156:
    v86 = v16 < v19[4];
    goto LABEL_149;
  }
  LOWORD(v155) = 257;
  v94 = *(_QWORD *)(a3 + 40);
  v95 = *(_QWORD *)(a3 + 24);
  v96 = *(_BYTE *)(a3 + 32) & 0xF;
  v126 = sub_1648B60(120);
  if ( v126 )
  {
    a2 = v95;
    sub_15E2490(v126, v95, v96, (__int64)&v153, v94);
  }
  LOWORD(v155) = 257;
  v97 = sub_15E0530(v130);
  v132 = (_QWORD *)sub_22077B0(64);
  if ( v132 )
  {
    a2 = v97;
    sub_157FB60(v132, v97, (__int64)&v153, v126, 0);
  }
  v124 = v132 + 5;
LABEL_39:
  v29 = sub_157E9C0((__int64)v132);
  v146 = v132;
  v148 = v29;
  v147 = v124;
  v30 = v126;
  v153 = &v155;
  if ( byte_4FAC680 )
    v30 = a3;
  v145 = 0;
  v149 = 0;
  v150 = 0;
  v31 = *(_QWORD *)(v130 + 24);
  v151 = 0;
  v152 = 0;
  v133 = v30;
  v154 = 0x1000000000LL;
  if ( (*(_BYTE *)(v30 + 18) & 1) == 0 )
  {
    v32 = *(_QWORD *)(v30 + 88);
    v33 = v32 + 40LL * *(_QWORD *)(v30 + 96);
    goto LABEL_43;
  }
  v93 = v30;
  sub_15E08E0(v30, a2);
  v32 = *(_QWORD *)(v93 + 88);
  v33 = v32 + 40LL * *(_QWORD *)(v93 + 96);
  if ( (*(_BYTE *)(v93 + 18) & 1) == 0 )
  {
LABEL_43:
    if ( v32 != v33 )
      goto LABEL_44;
LABEL_160:
    v39 = (unsigned int)v154;
    goto LABEL_48;
  }
  sub_15E08E0(v93, a2);
  v32 = *(_QWORD *)(v93 + 88);
  if ( v32 == v33 )
    goto LABEL_160;
LABEL_44:
  LODWORD(v34) = 0;
  v35 = v32;
  do
  {
    v34 = (unsigned int)(v34 + 1);
    v36 = sub_18910C0((__int64 *)&v145, v35, *(_QWORD *)(*(_QWORD *)(v31 + 16) + 8 * v34));
    v38 = (unsigned int)v154;
    if ( (unsigned int)v154 >= HIDWORD(v154) )
    {
      v123 = v36;
      sub_16CD150((__int64)&v153, &v155, 0, 8, v36, v37);
      v38 = (unsigned int)v154;
      v36 = v123;
    }
    v35 += 40;
    v153[v38] = (__int64 *)v36;
    v39 = (unsigned int)(v154 + 1);
    LODWORD(v154) = v154 + 1;
  }
  while ( v33 != v35 );
LABEL_48:
  LOWORD(v144) = 257;
  v40 = sub_1285290((__int64 *)&v145, *(_QWORD *)(v130 + 24), v130, (int)v153, v39, (__int64)&v142, 0);
  v41 = *(_WORD *)(v40 + 18) & 0xFFFC | 1;
  *(_WORD *)(v40 + 18) = v41;
  *(_WORD *)(v40 + 18) = v41 & 0x8000 | (*(_WORD *)(v130 + 18) >> 2) & 0xFFC | 1;
  *(_QWORD *)(v40 + 56) = *(_QWORD *)(v130 + 112);
  v42 = *(__int64 **)(*(_QWORD *)(v133 + 24) + 16LL);
  if ( *(_BYTE *)(*v42 + 8) )
  {
    v90 = sub_18910C0((__int64 *)&v145, v40, *v42);
    LOWORD(v144) = 257;
    v135 = v148;
    v91 = sub_1648A60(56, v90 != 0);
    v45 = v91;
    if ( v91 )
      sub_15F6F90((__int64)v91, v135, v90, 0);
  }
  else
  {
    v43 = v148;
    LOWORD(v144) = 257;
    v44 = sub_1648A60(56, 0);
    v45 = v44;
    if ( v44 )
      sub_15F6F90((__int64)v44, v43, 0, 0);
  }
  if ( v146 )
  {
    v46 = v147;
    sub_157E9D0((__int64)(v146 + 5), (__int64)v45);
    v47 = v45[3];
    v48 = *v46;
    v45[4] = v46;
    v48 &= 0xFFFFFFFFFFFFFFF8LL;
    v45[3] = v48 | v47 & 7;
    *(_QWORD *)(v48 + 8) = v45 + 3;
    *v46 = *v46 & 7 | (unsigned __int64)(v45 + 3);
  }
  sub_164B780((__int64)v45, (__int64 *)&v142);
  if ( v145 )
  {
    v138 = v145;
    sub_1623A60((__int64)&v138, (__int64)v145, 2);
    v49 = v45[6];
    v50 = (__int64)(v45 + 6);
    if ( v49 )
    {
      sub_161E7C0((__int64)(v45 + 6), v49);
      v50 = (__int64)(v45 + 6);
    }
    v51 = v138;
    v45[6] = v138;
    if ( v51 )
      sub_1623210((__int64)&v138, v51, v50);
  }
  if ( !byte_4FAC680 )
  {
    sub_15E4330(v126, a3);
    sub_164B7C0(v126, a3);
    sub_1893BC0(a1, a3);
    sub_164D160(a3, v126, a4, a5, a6, a7, v88, v89, a10, a11);
    sub_15E3D00(a3);
    goto LABEL_76;
  }
  v52 = sub_1626D20(a3);
  if ( v52 )
  {
    v134 = v52;
    sub_15C7110(&v137, *(_DWORD *)(v52 + 28), 0, v52, 0);
    sub_15C7110(&v138, *(_DWORD *)(v134 + 28), 0, v134, 0);
    v53 = (__int64 *)(v40 + 48);
    v142 = v137;
    if ( v137 )
    {
      sub_1623A60((__int64)&v142, (__int64)v137, 2);
      v53 = (__int64 *)(v40 + 48);
      if ( (unsigned __int8 **)(v40 + 48) == &v142 )
      {
        if ( v142 )
          sub_161E7C0((__int64)&v142, (__int64)v142);
        goto LABEL_64;
      }
      v98 = *(_QWORD *)(v40 + 48);
      if ( !v98 )
      {
LABEL_169:
        v99 = v142;
        *(_QWORD *)(v40 + 48) = v142;
        if ( v99 )
          sub_1623210((__int64)&v142, v99, (__int64)v53);
LABEL_64:
        v54 = v45 + 6;
        v142 = v138;
        if ( v138 )
        {
          sub_1623A60((__int64)&v142, (__int64)v138, 2);
          if ( v54 == (__int64 *)&v142 )
          {
            if ( v142 )
              sub_161E7C0((__int64)&v142, (__int64)v142);
            goto LABEL_68;
          }
          v100 = v45[6];
          if ( !v100 )
          {
LABEL_178:
            v101 = v142;
            v45[6] = v142;
            if ( v101 )
            {
              sub_1623210((__int64)&v142, v101, (__int64)(v45 + 6));
              v55 = v138;
LABEL_69:
              if ( v55 )
                sub_161E7C0((__int64)&v138, (__int64)v55);
              goto LABEL_71;
            }
LABEL_68:
            v55 = v138;
            goto LABEL_69;
          }
        }
        else if ( v54 == (__int64 *)&v142 || (v100 = v45[6]) == 0 )
        {
LABEL_71:
          if ( v137 )
            sub_161E7C0((__int64)&v137, (__int64)v137);
          goto LABEL_73;
        }
        sub_161E7C0((__int64)(v45 + 6), v100);
        goto LABEL_178;
      }
    }
    else
    {
      if ( v53 == (__int64 *)&v142 )
        goto LABEL_64;
      v98 = *(_QWORD *)(v40 + 48);
      if ( !v98 )
        goto LABEL_64;
    }
    v136 = v53;
    sub_161E7C0((__int64)v53, v98);
    v53 = v136;
    goto LABEL_169;
  }
LABEL_73:
  v142 = 0;
  v143 = 0;
  v56 = *(_QWORD *)(a3 + 80);
  v57 = a3 + 72;
  v144 = 0;
  v58 = *(_QWORD *)(v56 + 8);
  if ( v58 == a3 + 72 )
  {
    v59 = v140;
    if ( v140 == v139 )
      goto LABEL_76;
  }
  else
  {
    do
    {
      while ( 1 )
      {
        v81 = (unsigned __int8 *)(v58 - 24);
        if ( !v58 )
          v81 = 0;
        sub_157EE90((__int64)v81);
        v138 = v81;
        v80 = v143;
        if ( v143 != v144 )
          break;
        sub_15D0700((__int64)&v142, v143, &v138);
        v58 = *(_QWORD *)(v58 + 8);
        if ( v57 == v58 )
          goto LABEL_132;
      }
      if ( v143 )
      {
        *(_QWORD *)v143 = v81;
        v80 = v143;
      }
      v143 = v80 + 8;
      v58 = *(_QWORD *)(v58 + 8);
    }
    while ( v57 != v58 );
LABEL_132:
    v82 = v143;
    if ( v143 == v142 )
    {
      v83 = v144 - v143;
      if ( !v143 )
      {
        v59 = v140;
        if ( v140 == v139 )
          goto LABEL_76;
        goto LABEL_75;
      }
    }
    else
    {
      do
      {
        sub_157F980(*((_QWORD *)v82 - 1));
        v82 = v143 - 8;
        v143 = v82;
      }
      while ( v142 != v82 );
      v83 = v144 - v82;
    }
    j_j___libc_free_0(v82, v83);
    v59 = v140;
    if ( v140 == v139 )
      goto LABEL_76;
  }
  do
  {
LABEL_75:
    sub_15F20C0(*((_QWORD **)v59 - 1));
    v59 = v140 - 8;
    v140 = v59;
  }
  while ( v139 != v59 );
LABEL_76:
  if ( v153 != &v155 )
    _libc_free((unsigned __int64)v153);
  if ( v145 )
    sub_161E7C0((__int64)&v145, (__int64)v145);
  if ( v139 )
    j_j___libc_free_0(v139, v141 - v139);
}
