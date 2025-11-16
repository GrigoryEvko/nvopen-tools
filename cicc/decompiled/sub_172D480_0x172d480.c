// Function: sub_172D480
// Address: 0x172d480
//
unsigned __int8 *__fastcall sub_172D480(
        __int64 a1,
        __int64 *a2,
        unsigned __int8 a3,
        __int64 a4,
        double a5,
        double a6,
        double a7)
{
  int v7; // eax
  __int64 v8; // r11
  int v9; // eax
  __int64 v10; // r9
  char v15; // al
  _QWORD ***v16; // r11
  __int64 *v17; // r10
  char v18; // al
  __int64 *v19; // r10
  __int64 *v20; // r11
  __int64 *v21; // rdx
  __int64 *v22; // r15
  __int64 *v23; // r14
  __int64 v24; // r8
  __int64 *v25; // rdi
  unsigned int v26; // eax
  int v27; // r9d
  __int64 v28; // r8
  __int64 v29; // r11
  __int64 v30; // r10
  unsigned int v31; // edx
  int v32; // ebx
  char v33; // cl
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // edx
  unsigned __int64 v37; // rax
  __int64 *v38; // rax
  unsigned int v39; // esi
  __int64 *v40; // rax
  __int64 v41; // rax
  __int64 *v42; // rax
  unsigned __int8 *v43; // rax
  unsigned __int8 *v44; // r12
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // rdx
  unsigned __int8 *v48; // rsi
  __int64 v49; // rax
  unsigned __int8 *v50; // r12
  unsigned __int8 *v51; // rax
  unsigned __int8 *v52; // rax
  unsigned __int8 *v53; // rax
  __int64 v54; // rcx
  char v55; // al
  __int64 *v56; // rax
  char v57; // al
  __int64 v58; // rax
  unsigned int v59; // eax
  __int64 **v60; // rdi
  _QWORD *v61; // rsi
  __int64 *v62; // rdi
  char v63; // al
  unsigned int v64; // esi
  __int64 **v65; // rax
  __int64 *v66; // rdi
  __int64 *v67; // rax
  __int64 *v68; // rax
  bool v69; // al
  __int16 v70; // dx
  bool v71; // al
  _QWORD *v72; // rdi
  __int64 *v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  unsigned int v76; // edx
  int v77; // eax
  __int64 v78; // r9
  bool v79; // al
  unsigned __int8 *v80; // r15
  __int64 v81; // rax
  __int64 v82; // r12
  unsigned __int8 *v83; // rax
  char v84; // al
  __int64 *v85; // rax
  const void **v86; // [rsp+8h] [rbp-118h]
  const void **v87; // [rsp+10h] [rbp-110h]
  __int64 *v88; // [rsp+18h] [rbp-108h]
  char v89; // [rsp+18h] [rbp-108h]
  __int64 v90; // [rsp+18h] [rbp-108h]
  unsigned __int8 v91; // [rsp+20h] [rbp-100h]
  __int64 v92; // [rsp+20h] [rbp-100h]
  char v93; // [rsp+20h] [rbp-100h]
  __int64 v94; // [rsp+20h] [rbp-100h]
  char v95; // [rsp+20h] [rbp-100h]
  char v96; // [rsp+20h] [rbp-100h]
  __int64 *v97; // [rsp+48h] [rbp-D8h]
  __int64 *v98; // [rsp+50h] [rbp-D0h]
  __int64 *v99; // [rsp+50h] [rbp-D0h]
  __int64 v100; // [rsp+50h] [rbp-D0h]
  __int64 *v101; // [rsp+50h] [rbp-D0h]
  char v102; // [rsp+50h] [rbp-D0h]
  char v103; // [rsp+50h] [rbp-D0h]
  __int64 v104; // [rsp+50h] [rbp-D0h]
  __int64 v105; // [rsp+50h] [rbp-D0h]
  __int64 v106; // [rsp+50h] [rbp-D0h]
  __int64 v107; // [rsp+50h] [rbp-D0h]
  __int64 *v108; // [rsp+50h] [rbp-D0h]
  __int64 *v109; // [rsp+58h] [rbp-C8h]
  __int64 v110; // [rsp+58h] [rbp-C8h]
  char v111; // [rsp+58h] [rbp-C8h]
  __int64 *v112; // [rsp+58h] [rbp-C8h]
  __int64 v113; // [rsp+58h] [rbp-C8h]
  __int64 v114; // [rsp+58h] [rbp-C8h]
  __int16 v115; // [rsp+58h] [rbp-C8h]
  __int64 v116; // [rsp+58h] [rbp-C8h]
  __int16 v117; // [rsp+58h] [rbp-C8h]
  char v118; // [rsp+58h] [rbp-C8h]
  __int64 v119; // [rsp+58h] [rbp-C8h]
  char v120; // [rsp+58h] [rbp-C8h]
  char v121; // [rsp+58h] [rbp-C8h]
  char v122; // [rsp+59h] [rbp-C7h]
  __int64 *v123; // [rsp+60h] [rbp-C0h]
  __int64 v124; // [rsp+60h] [rbp-C0h]
  char v125; // [rsp+60h] [rbp-C0h]
  __int64 v126; // [rsp+60h] [rbp-C0h]
  _QWORD ***v127; // [rsp+60h] [rbp-C0h]
  __int64 v128; // [rsp+60h] [rbp-C0h]
  __int64 v129; // [rsp+60h] [rbp-C0h]
  __int16 v130; // [rsp+60h] [rbp-C0h]
  __int64 *v131; // [rsp+60h] [rbp-C0h]
  __int64 v132; // [rsp+60h] [rbp-C0h]
  __int64 v133; // [rsp+60h] [rbp-C0h]
  __int64 v134; // [rsp+60h] [rbp-C0h]
  __int64 v135; // [rsp+60h] [rbp-C0h]
  __int64 v136; // [rsp+60h] [rbp-C0h]
  __int64 *v137; // [rsp+60h] [rbp-C0h]
  _QWORD ***v138; // [rsp+68h] [rbp-B8h]
  __int64 *v139; // [rsp+68h] [rbp-B8h]
  unsigned int v140; // [rsp+68h] [rbp-B8h]
  __int64 v141; // [rsp+68h] [rbp-B8h]
  __int64 v142; // [rsp+68h] [rbp-B8h]
  bool v143; // [rsp+68h] [rbp-B8h]
  __int64 v144; // [rsp+68h] [rbp-B8h]
  _QWORD *v145; // [rsp+68h] [rbp-B8h]
  __int16 v146; // [rsp+68h] [rbp-B8h]
  __int64 v147; // [rsp+68h] [rbp-B8h]
  __int64 v148; // [rsp+68h] [rbp-B8h]
  unsigned int v149; // [rsp+68h] [rbp-B8h]
  __int64 v150; // [rsp+68h] [rbp-B8h]
  __int64 v151; // [rsp+68h] [rbp-B8h]
  __int64 v152; // [rsp+68h] [rbp-B8h]
  __int64 v153; // [rsp+68h] [rbp-B8h]
  char v154; // [rsp+69h] [rbp-B7h]
  int v155; // [rsp+78h] [rbp-A8h] BYREF
  int v156; // [rsp+7Ch] [rbp-A4h] BYREF
  __int64 *v157; // [rsp+80h] [rbp-A0h] BYREF
  __int64 *v158; // [rsp+88h] [rbp-98h] BYREF
  __int64 *v159; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v160; // [rsp+98h] [rbp-88h]
  __int64 *v161; // [rsp+A0h] [rbp-80h] BYREF
  unsigned int v162; // [rsp+A8h] [rbp-78h]
  __int64 *v163; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v164; // [rsp+B8h] [rbp-68h]
  __int64 *v165; // [rsp+C0h] [rbp-60h] BYREF
  unsigned int v166; // [rsp+C8h] [rbp-58h]
  __int64 **v167; // [rsp+D0h] [rbp-50h] BYREF
  __int64 **v168; // [rsp+D8h] [rbp-48h]
  __int16 v169; // [rsp+E0h] [rbp-40h]

  v7 = *(unsigned __int16 *)(a1 + 18);
  v8 = *(_QWORD *)(a1 - 48);
  BYTE1(v7) &= ~0x80u;
  v155 = v7;
  v9 = *((unsigned __int16 *)a2 + 9);
  BYTE1(v9) &= ~0x80u;
  v156 = v9;
  if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 11 || *(_BYTE *)(*(_QWORD *)*(a2 - 6) + 8LL) != 11 )
    return 0;
  v138 = (_QWORD ***)v8;
  v157 = *(__int64 **)(a1 - 24);
  v15 = sub_17275C0(v8, v157, &v155, &v158, (__int64 *)&v159, (__int64 *)&v157);
  v16 = v138;
  v17 = a2;
  if ( v15 )
  {
    v109 = 0;
    v16 = 0;
    v139 = 0;
    goto LABEL_7;
  }
  v55 = *((_BYTE *)v138 + 16);
  if ( v55 == 50 )
  {
    if ( !*(v138 - 6) )
      goto LABEL_87;
    v158 = (__int64 *)*(v138 - 6);
    v68 = (__int64 *)*(v138 - 3);
    if ( !v68 )
      goto LABEL_87;
  }
  else if ( v55 != 5
         || *((_WORD *)v138 + 9) != 26
         || (v74 = *((_DWORD *)v138 + 5) & 0xFFFFFFF, !v138[-3 * v74])
         || (v158 = (__int64 *)v138[-3 * (*((_DWORD *)v138 + 5) & 0xFFFFFFF)],
             (v68 = (__int64 *)v138[3 * (1 - v74)]) == 0) )
  {
LABEL_87:
    v158 = (__int64 *)v138;
    v56 = (__int64 *)sub_15A04A0(*v138);
    v16 = v138;
    v17 = a2;
    v159 = v56;
    goto LABEL_88;
  }
  v159 = v68;
LABEL_88:
  v139 = v157;
  v57 = *((_BYTE *)v157 + 16);
  if ( v57 == 50 )
  {
    if ( *(v157 - 6) )
    {
      v109 = (__int64 *)*(v157 - 3);
      if ( v109 )
      {
        v139 = (__int64 *)*(v157 - 6);
        goto LABEL_7;
      }
    }
LABEL_91:
    v101 = v17;
    v127 = v16;
    v58 = sub_15A04A0((_QWORD **)*v157);
    v16 = v127;
    v17 = v101;
    v109 = (__int64 *)v58;
    goto LABEL_7;
  }
  if ( v57 != 5 )
    goto LABEL_91;
  if ( *((_WORD *)v157 + 9) != 26 )
    goto LABEL_91;
  v75 = *((_DWORD *)v157 + 5) & 0xFFFFFFF;
  if ( !v157[-3 * v75] )
    goto LABEL_91;
  v109 = (__int64 *)v157[3 * (1 - v75)];
  if ( !v109 )
    goto LABEL_91;
  v139 = (__int64 *)v157[-3 * v75];
LABEL_7:
  v123 = (__int64 *)v16;
  if ( (unsigned int)(v155 - 32) > 1 )
    return 0;
  v98 = v17;
  v97 = (__int64 *)*(v17 - 6);
  v161 = (__int64 *)*(v17 - 3);
  v18 = sub_17275C0((__int64)v97, v161, &v156, &v163, (__int64 *)&v165, (__int64 *)&v161);
  v19 = v98;
  v20 = v123;
  if ( v18 )
  {
    v21 = v158;
    v22 = v165;
    v23 = v158;
    if ( v163 != v158 )
    {
      if ( v163 == v159 )
      {
        v23 = v163;
      }
      else if ( v139 == v163 )
      {
        v23 = v139;
      }
      else if ( v109 == v163 )
      {
        v23 = v109;
      }
      else
      {
        if ( v139 != v165 && v158 != v165 && v159 != v165 && v109 != v165 )
          return 0;
        v23 = v165;
        v22 = v163;
      }
    }
    goto LABEL_16;
  }
  v167 = &v163;
  v168 = &v165;
  v63 = sub_13D5EF0(&v167, (__int64)v97);
  v20 = v123;
  v19 = v98;
  if ( v63 )
  {
    v22 = v165;
  }
  else
  {
    v163 = v97;
    v73 = (__int64 *)sub_15A04A0((_QWORD **)*v97);
    v19 = v98;
    v165 = v73;
    v20 = v123;
    v22 = v73;
  }
  v23 = v163;
  v21 = v158;
  if ( v163 == v158 || v163 == v159 || v139 == v163 || v109 == v163 )
  {
LABEL_16:
    v97 = v161;
    goto LABEL_17;
  }
  if ( v139 == v22 || v158 == v22 || v159 == v22 || v109 == v22 )
  {
    v97 = v161;
    v23 = v22;
    v22 = v163;
LABEL_17:
    v24 = (__int64)v159;
    if ( (unsigned int)(v156 - 32) > 1 )
      return 0;
    goto LABEL_18;
  }
  if ( (unsigned int)(v156 - 32) > 1 )
    return 0;
  v108 = v19;
  v137 = v20;
  v167 = &v163;
  v168 = &v165;
  v84 = sub_13D5EF0(&v167, (__int64)v161);
  v20 = v137;
  v19 = v108;
  if ( v84 )
  {
    v22 = v165;
  }
  else
  {
    v163 = v161;
    v85 = (__int64 *)sub_15A04A0((_QWORD **)*v161);
    v19 = v108;
    v20 = v137;
    v165 = v85;
    v22 = v85;
  }
  v23 = v158;
  v24 = (__int64)v159;
  if ( v163 == v158 )
    goto LABEL_129;
  if ( v163 == v159 )
  {
    v21 = v158;
    v23 = v163;
    goto LABEL_149;
  }
  if ( v139 == v163 )
  {
    v23 = v139;
    v24 = (__int64)v109;
    goto LABEL_24;
  }
  if ( v109 == v163 )
  {
    v23 = v109;
    v24 = (__int64)v139;
    goto LABEL_24;
  }
  if ( v22 != v139 && v22 != v158 && v22 != v159 && v22 != v109 )
    return 0;
  v21 = v158;
  v23 = v22;
  v22 = v163;
LABEL_18:
  if ( v23 == v21 )
  {
LABEL_129:
    v20 = v157;
    goto LABEL_24;
  }
  if ( (__int64 *)v24 == v23 )
  {
LABEL_149:
    v20 = v157;
    v24 = (__int64)v21;
    goto LABEL_24;
  }
  v25 = v139;
  v24 = (__int64)v109;
  if ( v139 != v23 )
  {
    if ( v109 != v23 )
    {
      v25 = 0;
      v20 = 0;
    }
    v24 = (__int64)v25;
  }
LABEL_24:
  v99 = v19;
  v110 = (__int64)v20;
  v124 = v24;
  v140 = sub_1727650((__int64)v23, v24, (__int64)v20, v155, a5, a6, a7);
  v26 = sub_1727650((__int64)v23, (__int64)v22, (__int64)v97, v156, a5, a6, a7);
  LOWORD(v27) = v140;
  v28 = v124;
  v29 = v110;
  v30 = (__int64)v99;
  v31 = v26 & v140;
  if ( (v26 & v140) == 0 )
  {
    if ( !a3 )
    {
      v27 = (2 * (_WORD)v140) & 0x2AA | (v140 >> 1) & 0x155;
      v26 = (2 * (_WORD)v26) & 0x2AA | (v26 >> 1) & 0x155;
    }
    if ( (v27 & 0x20) != 0 && (v26 & 0x100) != 0 )
    {
      if ( *(_BYTE *)(v124 + 16) == 13
        && *(_BYTE *)(v110 + 16) == 13
        && *((_BYTE *)v22 + 16) == 13
        && *((_BYTE *)v97 + 16) == 13 )
      {
        return (unsigned __int8 *)sub_1729A60(
                                    (__int64 *)a1,
                                    (__int64)v99,
                                    a3,
                                    (unsigned __int8 *)v23,
                                    v124,
                                    (__int64)v22,
                                    a5,
                                    a6,
                                    a7,
                                    (__int64)v97,
                                    v156,
                                    a4);
      }
    }
    else if ( (v27 & 0x100) != 0
           && (v26 & 0x20) != 0
           && *((_BYTE *)v22 + 16) == 13
           && *((_BYTE *)v97 + 16) == 13
           && *(_BYTE *)(v124 + 16) == 13
           && *(_BYTE *)(v110 + 16) == 13 )
    {
      return (unsigned __int8 *)sub_1729A60(
                                  v99,
                                  a1,
                                  a3,
                                  (unsigned __int8 *)v23,
                                  (__int64)v22,
                                  v124,
                                  a5,
                                  a6,
                                  a7,
                                  v110,
                                  v155,
                                  a4);
    }
    return 0;
  }
  if ( a3 )
  {
    v32 = 32;
    v33 = 0;
    if ( (v31 & 0x10) == 0 )
      goto LABEL_27;
LABEL_52:
    v169 = 257;
    v43 = sub_172AC10(a4, v124, (__int64)v22, (__int64 *)&v167, a5, a6, a7);
    v169 = 257;
    v44 = sub_1729500(a4, (unsigned __int8 *)v23, (__int64)v43, (__int64 *)&v167, a5, a6, a7);
    v47 = sub_15A06D0((__int64 **)*v23, (__int64)v23, v45, v46);
    v169 = 257;
    if ( v44[16] > 0x10u || *(_BYTE *)(v47 + 16) > 0x10u )
      return sub_1727440(a4, v32, (__int64)v44, v47, (__int64 *)&v167);
    v48 = v44;
    goto LABEL_55;
  }
  v32 = 33;
  v33 = 1;
  v31 = (2 * (_WORD)v31) & 0x2AA | (v31 >> 1) & 0x155;
  if ( (v31 & 0x10) != 0 )
    goto LABEL_52;
LABEL_27:
  if ( (v31 & 4) != 0 )
  {
    v169 = 257;
    v50 = sub_172AC10(a4, v124, (__int64)v22, (__int64 *)&v167, a5, a6, a7);
    v169 = 257;
    v51 = sub_1729500(a4, (unsigned __int8 *)v23, (__int64)v50, (__int64 *)&v167, a5, a6, a7);
    v169 = 257;
    v48 = v51;
    if ( v51[16] <= 0x10u && v50[16] <= 0x10u )
    {
      v47 = (__int64)v50;
      goto LABEL_55;
    }
    v54 = (__int64)v50;
    return sub_1727440(a4, v32, (__int64)v48, v54, (__int64 *)&v167);
  }
  if ( (v31 & 1) != 0 )
  {
    v169 = 257;
    v52 = sub_1729500(a4, (unsigned __int8 *)v124, (__int64)v22, (__int64 *)&v167, a5, a6, a7);
    v169 = 257;
    v53 = sub_1729500(a4, (unsigned __int8 *)v23, (__int64)v52, (__int64 *)&v167, a5, a6, a7);
    v169 = 257;
    v48 = v53;
    if ( v53[16] <= 0x10u && *((_BYTE *)v23 + 16) <= 0x10u )
    {
      v47 = (__int64)v23;
LABEL_55:
      v144 = sub_15A37B0(v32, v48, (_QWORD *)v47, 0);
      v49 = sub_14DBA30(v144, *(_QWORD *)(a4 + 96), 0);
      v10 = v144;
      if ( v49 )
        return (unsigned __int8 *)v49;
      return (unsigned __int8 *)v10;
    }
    v54 = (__int64)v23;
    return sub_1727440(a4, v32, (__int64)v48, v54, (__int64 *)&v167);
  }
  if ( *(_BYTE *)(v124 + 16) != 13 || *((_BYTE *)v22 + 16) != 13 )
    return 0;
  if ( (v31 & 0x28) == 0 )
  {
LABEL_32:
    if ( (v31 & 2) == 0 )
    {
LABEL_33:
      if ( (v31 & 0x100) != 0 && *(_BYTE *)(v29 + 16) == 13 && *((_BYTE *)v97 + 16) == 13 )
      {
        if ( v155 != v32 )
        {
          v125 = v33;
          v141 = v28;
          v34 = sub_15A2D30((__int64 *)v28, v29, a5, a6, a7);
          v33 = v125;
          v28 = v141;
          v29 = v34;
        }
        if ( v156 != v32 )
        {
          v111 = v33;
          v126 = v28;
          v142 = v29;
          v35 = sub_15A2D30(v22, (__int64)v97, a5, a6, a7);
          v33 = v111;
          v28 = v126;
          v97 = (__int64 *)v35;
          v29 = v142;
        }
        v36 = *(_DWORD *)(v29 + 32);
        v164 = v36;
        if ( v36 > 0x40 )
        {
          v121 = v33;
          v136 = v28;
          v153 = v29;
          sub_16A4FD0((__int64)&v163, (const void **)(v29 + 24));
          v36 = v164;
          v29 = v153;
          v28 = v136;
          v33 = v121;
          if ( v164 > 0x40 )
          {
            sub_16A8F00((__int64 *)&v163, v97 + 3);
            v36 = v164;
            v38 = v163;
            v33 = v121;
            v28 = v136;
            v29 = v153;
LABEL_43:
            v166 = v36;
            v165 = v38;
            v164 = 0;
            v39 = *(_DWORD *)(v28 + 32);
            v160 = v39;
            if ( v39 > 0x40 )
            {
              v120 = v33;
              v134 = v29;
              v151 = v28;
              sub_16A4FD0((__int64)&v159, (const void **)(v28 + 24));
              v39 = v160;
              v28 = v151;
              v29 = v134;
              v33 = v120;
              if ( v160 > 0x40 )
              {
                v135 = v151;
                v152 = v29;
                sub_16A8890((__int64 *)&v159, v22 + 3);
                v39 = v160;
                v36 = v166;
                v41 = (__int64)v159;
                v33 = v120;
                v28 = v135;
                v29 = v152;
LABEL_46:
                v162 = v39;
                v161 = (__int64 *)v41;
                v160 = 0;
                if ( v36 > 0x40 )
                {
                  v118 = v33;
                  v132 = v28;
                  v148 = v29;
                  sub_16A8890((__int64 *)&v165, (__int64 *)&v161);
                  v76 = v166;
                  v42 = v165;
                  v166 = 0;
                  v29 = v148;
                  v28 = v132;
                  LODWORD(v168) = v76;
                  v33 = v118;
                  v167 = (__int64 **)v165;
                  if ( v76 > 0x40 )
                  {
                    v95 = v118;
                    v149 = v76;
                    v119 = v29;
                    v77 = sub_16A57B0((__int64)&v167);
                    v33 = v95;
                    v28 = v132;
                    v29 = v119;
                    v143 = v149 == v77;
LABEL_49:
                    v100 = v28;
                    v112 = (__int64 *)v29;
                    v91 = v33;
                    sub_135E100((__int64 *)&v167);
                    sub_135E100((__int64 *)&v161);
                    sub_135E100((__int64 *)&v159);
                    sub_135E100((__int64 *)&v165);
                    sub_135E100((__int64 *)&v163);
                    if ( !v143 )
                      return (unsigned __int8 *)sub_15A0680(*(_QWORD *)a1, v91, 0);
                    v169 = 257;
                    v80 = sub_172AC10(a4, v100, (__int64)v22, (__int64 *)&v167, a5, a6, a7);
                    v81 = sub_15A2D10(v112, (__int64)v97, a5, a6, a7);
                    v169 = 257;
                    v82 = v81;
                    v83 = sub_1729500(a4, (unsigned __int8 *)v23, (__int64)v80, (__int64 *)&v167, a5, a6, a7);
                    v169 = 257;
                    return sub_17203D0(a4, v32, (__int64)v83, v82, (__int64 *)&v167);
                  }
                }
                else
                {
                  v42 = (__int64 *)((unsigned __int64)v165 & v41);
                  LODWORD(v168) = v36;
                  v165 = v42;
                  v167 = (__int64 **)v42;
                  v166 = 0;
                }
                v143 = v42 == 0;
                goto LABEL_49;
              }
              v40 = v159;
              v36 = v166;
            }
            else
            {
              v40 = *(__int64 **)(v28 + 24);
            }
            v41 = v22[3] & (unsigned __int64)v40;
            v159 = (__int64 *)v41;
            goto LABEL_46;
          }
          v37 = (unsigned __int64)v163;
        }
        else
        {
          v37 = *(_QWORD *)(v29 + 24);
        }
        v38 = (__int64 *)(v97[3] ^ v37);
        v163 = v38;
        goto LABEL_43;
      }
      return 0;
    }
    v64 = *(_DWORD *)(v28 + 32);
    LODWORD(v168) = v64;
    if ( v64 > 0x40 )
    {
      v90 = v30;
      v96 = v33;
      v107 = v29;
      v122 = BYTE1(v31);
      v133 = v28;
      v87 = (const void **)(v28 + 24);
      sub_16A4FD0((__int64)&v167, (const void **)(v28 + 24));
      v64 = (unsigned int)v168;
      v28 = v133;
      BYTE1(v31) = v122;
      v29 = v107;
      v33 = v96;
      v30 = v90;
      if ( (unsigned int)v168 > 0x40 )
      {
        sub_16A89F0((__int64 *)&v167, v22 + 3);
        v67 = (__int64 *)v167;
        BYTE1(v31) = v122;
        v29 = v107;
        v166 = (unsigned int)v168;
        v28 = v133;
        v165 = (__int64 *)v167;
        v33 = v96;
        v30 = v90;
        if ( (unsigned int)v168 > 0x40 )
        {
          if ( !sub_16A5220((__int64)&v165, v87) )
          {
            v79 = sub_16A5220((__int64)&v165, (const void **)v22 + 3);
            v30 = v90;
            if ( !v79 )
            {
              BYTE1(v31) = v122;
              v29 = v107;
              v28 = v133;
              v33 = v96;
              goto LABEL_112;
            }
            goto LABEL_140;
          }
LABEL_142:
          v78 = a1;
          goto LABEL_141;
        }
        v66 = *(__int64 **)(v133 + 24);
LABEL_110:
        if ( v67 != v66 )
        {
          if ( v67 != (__int64 *)v22[3] )
          {
LABEL_112:
            v103 = v33;
            v114 = v28;
            v129 = v29;
            v154 = BYTE1(v31);
            sub_135E100((__int64 *)&v165);
            BYTE1(v31) = v154;
            v29 = v129;
            v28 = v114;
            v33 = v103;
            goto LABEL_33;
          }
LABEL_140:
          v78 = v30;
LABEL_141:
          v150 = v78;
          sub_135E100((__int64 *)&v165);
          return (unsigned __int8 *)v150;
        }
        goto LABEL_142;
      }
      v65 = v167;
      v66 = *(__int64 **)(v133 + 24);
    }
    else
    {
      v65 = *(__int64 ***)(v28 + 24);
      v66 = (__int64 *)v65;
    }
    v67 = (__int64 *)(v22[3] | (unsigned __int64)v65);
    v166 = v64;
    v165 = v67;
    goto LABEL_110;
  }
  v59 = *(_DWORD *)(v124 + 32);
  LODWORD(v168) = v59;
  if ( v59 <= 0x40 )
  {
    v60 = *(__int64 ***)(v124 + 24);
    v61 = v60;
LABEL_94:
    v62 = (__int64 *)(v22[3] & (unsigned __int64)v60);
    v166 = v59;
    v145 = v62;
    v165 = v62;
LABEL_95:
    v10 = a1;
    if ( v61 != v145 )
    {
      v10 = v30;
      if ( (_QWORD *)v22[3] != v145 )
        goto LABEL_97;
    }
    return (unsigned __int8 *)v10;
  }
  v88 = v99;
  v93 = v33;
  v104 = v110;
  v115 = v31;
  v86 = (const void **)(v124 + 24);
  sub_16A4FD0((__int64)&v167, (const void **)(v124 + 24));
  v59 = (unsigned int)v168;
  v28 = v124;
  LOWORD(v31) = v115;
  v29 = v104;
  v33 = v93;
  v30 = (__int64)v88;
  if ( (unsigned int)v168 <= 0x40 )
  {
    v60 = v167;
    v61 = *(_QWORD **)(v124 + 24);
    goto LABEL_94;
  }
  v105 = v124;
  v116 = v29;
  v130 = v31;
  sub_16A8890((__int64 *)&v167, v22 + 3);
  LOWORD(v31) = v130;
  v29 = v116;
  v165 = (__int64 *)v167;
  v28 = v105;
  v145 = v167;
  v30 = (__int64)v88;
  v166 = (unsigned int)v168;
  v33 = v93;
  if ( (unsigned int)v168 <= 0x40 )
  {
    v61 = *(_QWORD **)(v105 + 24);
    goto LABEL_95;
  }
  v69 = sub_16A5220((__int64)&v165, v86);
  v70 = v130;
  if ( v69 )
  {
    v30 = a1;
  }
  else
  {
    v131 = v88;
    v89 = v93;
    v94 = v105;
    v106 = v116;
    v117 = v70;
    v71 = sub_16A5220((__int64)&v165, (const void **)v22 + 3);
    v30 = (__int64)v131;
    if ( !v71 )
    {
      LOWORD(v31) = v117;
      v29 = v106;
      v28 = v94;
      v33 = v89;
LABEL_97:
      v92 = v30;
      v102 = v33;
      v113 = v28;
      v128 = v29;
      v146 = v31;
      sub_135E100((__int64 *)&v165);
      LOWORD(v31) = v146;
      v29 = v128;
      v28 = v113;
      v33 = v102;
      v30 = v92;
      goto LABEL_32;
    }
  }
  v72 = v145;
  if ( v145 )
  {
    v147 = v30;
    j_j___libc_free_0_0(v72);
    return (unsigned __int8 *)v147;
  }
  return (unsigned __int8 *)v30;
}
