// Function: sub_22C85B0
// Address: 0x22c85b0
//
__int64 __fastcall sub_22C85B0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, char a5, char a6)
{
  __int64 v9; // r14
  __int64 v10; // r15
  int v11; // r10d
  __int64 v12; // rdi
  unsigned int v14; // eax
  int v15; // r10d
  char v16; // al
  int v17; // eax
  int v18; // r11d
  char v19; // al
  char v20; // r9
  int v21; // r10d
  __int64 v22; // rax
  bool v23; // al
  int v24; // r10d
  unsigned __int8 *v25; // rsi
  unsigned __int8 *v26; // rax
  unsigned __int8 *v27; // rsi
  unsigned __int8 *v28; // rax
  __int64 v29; // rax
  char v30; // al
  char v31; // al
  __int64 v32; // rax
  char v33; // al
  unsigned int v34; // r13d
  __int64 v35; // rax
  char v36; // al
  char v37; // al
  int v38; // r10d
  char v39; // r9
  int v40; // eax
  signed __int64 v41; // r8
  unsigned __int64 v42; // r8
  int v43; // r10d
  char v44; // r9
  char v45; // r9
  bool v46; // al
  unsigned int v47; // eax
  __int64 v48; // rax
  __int64 v49; // rax
  unsigned __int64 v50; // r14
  unsigned int v51; // r13d
  int v52; // ebx
  unsigned int v53; // r15d
  int v54; // ebx
  __int64 v55; // rbx
  unsigned int v56; // esi
  unsigned int v57; // r14d
  unsigned int v58; // ebx
  unsigned __int8 *v59; // r13
  unsigned __int8 *v60; // r15
  unsigned __int8 *v61; // rdi
  int v62; // r10d
  int v63; // eax
  unsigned __int64 v64; // rax
  __int64 v65; // rsi
  __int64 *v66; // r13
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  unsigned int v70; // edx
  unsigned int v71; // eax
  unsigned int v72; // eax
  __int64 *v73; // r13
  unsigned int v74; // eax
  unsigned int v75; // eax
  unsigned int v76; // eax
  char v77; // cl
  __int64 v78; // rdx
  unsigned int v79; // r8d
  int v80; // r10d
  char v81; // r9
  bool v82; // al
  unsigned int v83; // ecx
  int v84; // eax
  unsigned int v85; // [rsp+4h] [rbp-17Ch]
  unsigned int v86; // [rsp+4h] [rbp-17Ch]
  char v87; // [rsp+4h] [rbp-17Ch]
  bool v88; // [rsp+18h] [rbp-168h]
  unsigned __int64 v89; // [rsp+18h] [rbp-168h]
  char v90; // [rsp+18h] [rbp-168h]
  unsigned __int64 v91; // [rsp+18h] [rbp-168h]
  int v92; // [rsp+20h] [rbp-160h]
  char v93; // [rsp+20h] [rbp-160h]
  char v94; // [rsp+20h] [rbp-160h]
  char v95; // [rsp+20h] [rbp-160h]
  char v96; // [rsp+20h] [rbp-160h]
  char v97; // [rsp+20h] [rbp-160h]
  char v98; // [rsp+20h] [rbp-160h]
  int v99; // [rsp+20h] [rbp-160h]
  int v100; // [rsp+20h] [rbp-160h]
  char v101; // [rsp+20h] [rbp-160h]
  unsigned int v102; // [rsp+20h] [rbp-160h]
  unsigned int v103; // [rsp+28h] [rbp-158h]
  char v104; // [rsp+28h] [rbp-158h]
  int v105; // [rsp+28h] [rbp-158h]
  int v106; // [rsp+28h] [rbp-158h]
  int v107; // [rsp+28h] [rbp-158h]
  char v108; // [rsp+28h] [rbp-158h]
  int v109; // [rsp+28h] [rbp-158h]
  int v110; // [rsp+28h] [rbp-158h]
  char v111; // [rsp+28h] [rbp-158h]
  char v112; // [rsp+28h] [rbp-158h]
  unsigned int v113; // [rsp+30h] [rbp-150h]
  int v114; // [rsp+30h] [rbp-150h]
  int v115; // [rsp+30h] [rbp-150h]
  int v116; // [rsp+30h] [rbp-150h]
  int v117; // [rsp+30h] [rbp-150h]
  char v118; // [rsp+30h] [rbp-150h]
  int v119; // [rsp+30h] [rbp-150h]
  int v120; // [rsp+30h] [rbp-150h]
  int v121; // [rsp+30h] [rbp-150h]
  int v122; // [rsp+30h] [rbp-150h]
  int v123; // [rsp+30h] [rbp-150h]
  int v124; // [rsp+30h] [rbp-150h]
  int v125; // [rsp+30h] [rbp-150h]
  int v126; // [rsp+30h] [rbp-150h]
  int v127; // [rsp+30h] [rbp-150h]
  int v128; // [rsp+38h] [rbp-148h]
  __int64 *v131; // [rsp+58h] [rbp-128h] BYREF
  __int64 v132; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v133; // [rsp+68h] [rbp-118h]
  unsigned __int64 v134; // [rsp+70h] [rbp-110h] BYREF
  unsigned int v135; // [rsp+78h] [rbp-108h]
  unsigned __int64 v136; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v137; // [rsp+88h] [rbp-F8h]
  unsigned __int8 *v138; // [rsp+90h] [rbp-F0h] BYREF
  unsigned int v139; // [rsp+98h] [rbp-E8h]
  unsigned __int8 *v140; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int v141; // [rsp+A8h] [rbp-D8h]
  unsigned __int64 v142; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned int v143; // [rsp+B8h] [rbp-C8h]
  unsigned __int64 v144; // [rsp+C0h] [rbp-C0h] BYREF
  unsigned int v145; // [rsp+C8h] [rbp-B8h]
  unsigned __int8 *v146; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned int v147; // [rsp+D8h] [rbp-A8h]
  unsigned __int8 *v148; // [rsp+E0h] [rbp-A0h] BYREF
  unsigned int v149; // [rsp+E8h] [rbp-98h]
  unsigned __int64 v150; // [rsp+F0h] [rbp-90h] BYREF
  unsigned int v151; // [rsp+F8h] [rbp-88h]
  unsigned __int8 *v152; // [rsp+100h] [rbp-80h] BYREF
  unsigned int v153; // [rsp+108h] [rbp-78h]
  char v154; // [rsp+110h] [rbp-70h]
  unsigned __int8 *v155; // [rsp+120h] [rbp-60h] BYREF
  unsigned __int64 *v156; // [rsp+128h] [rbp-58h] BYREF
  unsigned __int8 *v157; // [rsp+130h] [rbp-50h] BYREF
  unsigned int v158; // [rsp+138h] [rbp-48h]
  char v159; // [rsp+140h] [rbp-40h]

  v9 = *(_QWORD *)(a4 - 64);
  v10 = *(_QWORD *)(a4 - 32);
  v11 = *(_WORD *)(a4 + 2) & 0x3F;
  if ( !a5 )
    v11 = sub_B52870(v11);
  if ( *(_BYTE *)v10 <= 0x15u && (*(_WORD *)(a4 + 2) & 0x3Fu) - 32 <= 1 && a3 == v9 )
  {
    if ( v11 == 32 )
    {
      LOWORD(v155) = 0;
      sub_22C0310((__int64)&v155, (unsigned __int8 *)v10, 0);
    }
    else
    {
      if ( (unsigned __int8)(*(_BYTE *)v10 - 12) <= 1u )
        goto LABEL_6;
      LOWORD(v155) = 0;
      sub_22C0430((__int64)&v155, (unsigned __int8 *)v10);
    }
    sub_22C0650(a1, (unsigned __int8 *)&v155);
    *(_BYTE *)(a1 + 40) = 1;
    sub_22C0090((unsigned __int8 *)&v155);
    return a1;
  }
LABEL_6:
  v12 = *(_QWORD *)(a3 + 8);
  if ( *(_BYTE *)(v12 + 8) == 12 )
  {
    v128 = v11;
    v14 = sub_BCB060(v12);
    v15 = v128;
    v103 = v14;
    v133 = v14;
    if ( v14 > 0x40 )
    {
      sub_C43690((__int64)&v132, 0, 0);
      v15 = v128;
    }
    else
    {
      v132 = 0;
    }
    if ( a3 == v9 || (v113 = v15, v16 = sub_22C3580((__int64)&v132, (_BYTE *)v9, (_BYTE *)a3, v15), v15 = v113, v16) )
    {
      sub_22C8110(a1, a2, v15, v10, (__int64)&v132, a4, a6);
LABEL_14:
      sub_969240(&v132);
      return a1;
    }
    v17 = sub_B52F50(v113);
    v18 = v17;
    if ( a3 == v10
      || (v92 = v113,
          v114 = v17,
          v19 = sub_22C3580((__int64)&v132, (_BYTE *)v10, (_BYTE *)a3, v17),
          v18 = v114,
          (v20 = v19) != 0) )
    {
      sub_22C8110(a1, a2, v18, v9, (__int64)&v132, a4, a6);
      goto LABEL_14;
    }
    v21 = v92;
    if ( *(_BYTE *)v9 == 85 )
    {
      v22 = *(_QWORD *)(v9 - 32);
      if ( v22 )
      {
        if ( !*(_BYTE *)v22 && *(_QWORD *)(v22 + 24) == *(_QWORD *)(v9 + 80) && *(_DWORD *)(v22 + 36) == 66 )
        {
          v49 = *(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
          if ( v49 )
          {
            if ( a3 == v49 )
            {
              if ( *(_BYTE *)v10 == 17 )
              {
                v50 = (unsigned int)sub_BCB060(*(_QWORD *)(v10 + 8));
                sub_AB1A50((__int64)&v146, v92, v10 + 24);
                v51 = v50;
                sub_AB0A00((__int64)&v155, (__int64)&v146);
                v52 = (int)v156;
                if ( (unsigned int)v156 > 0x40 )
                {
                  v53 = v50;
                  if ( v52 - (unsigned int)sub_C444A0((__int64)&v155) <= 0x40 )
                  {
                    v53 = *(_QWORD *)v155;
                    if ( v50 < *(_QWORD *)v155 )
                      v53 = v50;
                  }
                }
                else
                {
                  v53 = (unsigned int)v155;
                  if ( v50 < (unsigned __int64)v155 )
                    v53 = v50;
                }
                sub_969240((__int64 *)&v155);
                sub_AB0910((__int64)&v155, (__int64)&v146);
                v54 = (int)v156;
                if ( (unsigned int)v156 > 0x40 )
                {
                  v62 = sub_C444A0((__int64)&v155);
                  v63 = v54;
                  LODWORD(v55) = v50;
                  if ( (unsigned int)(v63 - v62) <= 0x40 )
                  {
                    v55 = *(_QWORD *)v155;
                    if ( v50 < *(_QWORD *)v155 )
                      LODWORD(v55) = v50;
                  }
                }
                else
                {
                  LODWORD(v55) = (_DWORD)v155;
                  if ( v50 < (unsigned __int64)v155 )
                    LODWORD(v55) = v50;
                }
                sub_969240((__int64 *)&v155);
                v135 = v50;
                if ( (unsigned int)v50 > 0x40 )
                {
                  sub_C43690((__int64)&v134, 0, 0);
                  if ( !v53 )
                  {
                    v137 = v50;
                    goto LABEL_150;
                  }
                  if ( v53 > 0x40 )
                    goto LABEL_109;
                  v64 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v53);
                  if ( v135 > 0x40 )
                  {
                    *(_QWORD *)v134 |= v64;
                    v137 = v50;
                    goto LABEL_150;
                  }
                }
                else
                {
                  v134 = 0;
                  if ( !v53 )
                  {
                    v137 = v50;
                    goto LABEL_111;
                  }
                  if ( v53 > 0x40 )
                  {
LABEL_109:
                    sub_C43C90(&v134, 0, v53);
                    goto LABEL_110;
                  }
                  v64 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v53);
                }
                v134 |= v64;
LABEL_110:
                v137 = v50;
                if ( (unsigned int)v50 <= 0x40 )
                {
LABEL_111:
                  v136 = 0;
                  goto LABEL_112;
                }
LABEL_150:
                sub_C43690((__int64)&v136, 0, 0);
                v51 = v137;
LABEL_112:
                v56 = v51 - v55;
                if ( v51 - (_DWORD)v55 != v51 )
                {
                  if ( v56 <= 0x3F && v51 <= 0x40 )
                  {
                    v141 = v51;
                    v136 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v55) << v56;
                    goto LABEL_116;
                  }
                  sub_C43C90(&v136, v56, v51);
                  v51 = v137;
                }
                v141 = v51;
                if ( v51 > 0x40 )
                {
                  sub_C43780((__int64)&v140, (const void **)&v136);
LABEL_117:
                  sub_C46A40((__int64)&v140, 1);
                  v57 = v141;
                  v58 = v135;
                  v141 = 0;
                  v59 = v140;
                  v60 = (unsigned __int8 *)v134;
                  v135 = 0;
                  v143 = v57;
                  v142 = (unsigned __int64)v140;
                  v139 = v58;
                  v138 = (unsigned __int8 *)v134;
                  if ( v58 <= 0x40 )
                  {
                    if ( v140 == (unsigned __int8 *)v134 )
                    {
                      sub_AADB10((__int64)&v150, v58, 1);
                      sub_22C06B0((__int64)&v155, (__int64)&v150, 0);
                      sub_969240((__int64 *)&v152);
                      if ( v151 <= 0x40 )
                      {
LABEL_123:
                        sub_969240((__int64 *)&v142);
                        sub_969240((__int64 *)&v140);
                        sub_969240((__int64 *)&v136);
                        if ( v135 > 0x40 && v134 )
                          j_j___libc_free_0_0(v134);
                        if ( v149 > 0x40 && v148 )
                          j_j___libc_free_0_0((unsigned __int64)v148);
                        sub_969240((__int64 *)&v146);
                        goto LABEL_42;
                      }
                      goto LABEL_154;
                    }
                  }
                  else if ( sub_C43C50((__int64)&v138, (const void **)&v142) )
                  {
                    sub_AADB10((__int64)&v150, v58, 1);
                    sub_22C06B0((__int64)&v155, (__int64)&v150, 0);
                    sub_969240((__int64 *)&v152);
                    if ( v151 <= 0x40 )
                    {
LABEL_120:
                      if ( !v60 )
                        goto LABEL_123;
                      v61 = v60;
                      goto LABEL_122;
                    }
LABEL_154:
                    if ( v150 )
                      j_j___libc_free_0_0(v150);
                    if ( v58 <= 0x40 )
                      goto LABEL_123;
                    goto LABEL_120;
                  }
                  v145 = v58;
                  LODWORD(v156) = v57;
                  v155 = v59;
                  v143 = 0;
                  v144 = (unsigned __int64)v60;
                  v139 = 0;
                  sub_AADC30((__int64)&v150, (__int64)&v144, (__int64 *)&v155);
                  sub_969240((__int64 *)&v144);
                  sub_969240((__int64 *)&v155);
                  sub_22C06B0((__int64)&v155, (__int64)&v150, 0);
                  sub_969240((__int64 *)&v152);
                  if ( v151 <= 0x40 )
                    goto LABEL_123;
                  v61 = (unsigned __int8 *)v150;
                  if ( !v150 )
                    goto LABEL_123;
LABEL_122:
                  j_j___libc_free_0_0((unsigned __int64)v61);
                  goto LABEL_123;
                }
LABEL_116:
                v140 = (unsigned __int8 *)v136;
                goto LABEL_117;
              }
              LOWORD(v155) = 6;
LABEL_42:
              sub_22C0650(a1, (unsigned __int8 *)&v155);
              *(_BYTE *)(a1 + 40) = 1;
              sub_22C0090((unsigned __int8 *)&v155);
              goto LABEL_14;
            }
          }
        }
      }
    }
    v155 = (unsigned __int8 *)a3;
    v156 = (unsigned __int64 *)&v131;
    LOBYTE(v157) = 0;
    if ( *(_BYTE *)v9 == 57 )
    {
      v29 = *(_QWORD *)(v9 - 64);
      if ( a3 == v29 )
      {
        if ( v29 )
        {
          v30 = sub_991580((__int64)&v156, *(_QWORD *)(v9 - 32));
          v21 = v92;
          v20 = 0;
          if ( v30 )
          {
            LOBYTE(v151) = 0;
            v150 = (unsigned __int64)&v134;
            v31 = sub_991580((__int64)&v150, v10);
            v21 = v92;
            v20 = 0;
            if ( v31 )
            {
              if ( v92 == 32 )
              {
                v65 = v134;
                v147 = 1;
                v66 = v131;
                v146 = 0;
                v149 = 1;
                v148 = 0;
                sub_9865C0((__int64)&v144, v134);
                sub_987160((__int64)&v144, v65, v67, v68, v69);
                v70 = v145;
                v145 = 0;
                v151 = v70;
                v150 = v144;
                if ( v70 > 0x40 )
                  sub_C43B90(&v150, v66);
                else
                  v150 = *v66 & v144;
                v71 = v151;
                v151 = 0;
                LODWORD(v156) = v71;
                v155 = (unsigned __int8 *)v150;
                if ( v147 > 0x40 && v146 )
                  j_j___libc_free_0_0((unsigned __int64)v146);
                v146 = v155;
                v72 = (unsigned int)v156;
                LODWORD(v156) = 0;
                v147 = v72;
                sub_969240((__int64 *)&v155);
                sub_969240((__int64 *)&v150);
                sub_969240((__int64 *)&v144);
                v73 = v131;
                sub_9865C0((__int64)&v150, v134);
                if ( v151 > 0x40 )
                  sub_C43B90(&v150, v73);
                else
                  v150 &= *v73;
                v74 = v151;
                v151 = 0;
                LODWORD(v156) = v74;
                v155 = (unsigned __int8 *)v150;
                if ( v149 > 0x40 && v148 )
                  j_j___libc_free_0_0((unsigned __int64)v148);
                v148 = v155;
                v75 = (unsigned int)v156;
                LODWORD(v156) = 0;
                v149 = v75;
                sub_969240((__int64 *)&v155);
                sub_969240((__int64 *)&v150);
                sub_AAF050((__int64)&v150, (__int64)&v146, 0);
                sub_22C06B0((__int64)&v155, (__int64)&v150, 0);
                sub_22C0650(a1, (unsigned __int8 *)&v155);
                *(_BYTE *)(a1 + 40) = 1;
                sub_22C0090((unsigned __int8 *)&v155);
                sub_969240((__int64 *)&v152);
                sub_969240((__int64 *)&v150);
                sub_969240((__int64 *)&v148);
                sub_969240((__int64 *)&v146);
                goto LABEL_14;
              }
              if ( v92 == 33 )
              {
                sub_AAF4C0((__int64)&v150, (__int64)v131, v134);
                sub_22C06B0((__int64)&v155, (__int64)&v150, 0);
                sub_22C0650(a1, (unsigned __int8 *)&v155);
                *(_BYTE *)(a1 + 40) = 1;
                sub_22C0090((unsigned __int8 *)&v155);
                sub_969240((__int64 *)&v152);
                sub_969240((__int64 *)&v150);
                goto LABEL_14;
              }
            }
          }
        }
      }
    }
    if ( *(_BYTE *)v9 == 51 )
    {
      v32 = *(_QWORD *)(v9 - 64);
      if ( !v32 || a3 != v32 )
      {
LABEL_24:
        v104 = v20;
        v115 = v21;
        v23 = sub_B532B0(v21);
        v24 = v115;
        if ( !v23 )
          goto LABEL_25;
        v155 = (unsigned __int8 *)a3;
        v156 = &v136;
        LOBYTE(v157) = 0;
        if ( *(_BYTE *)v9 != 56 )
          goto LABEL_25;
        v35 = *(_QWORD *)(v9 - 64);
        v94 = v104;
        if ( a3 != v35 )
          goto LABEL_25;
        if ( !v35 )
          goto LABEL_25;
        v36 = sub_991580((__int64)&v156, *(_QWORD *)(v9 - 32));
        v24 = v115;
        if ( !v36
          || (v105 = v115,
              v150 = (unsigned __int64)&v134,
              LOBYTE(v151) = 0,
              v37 = sub_991580((__int64)&v150, v10),
              v24 = v115,
              (v118 = v37) == 0) )
        {
LABEL_25:
          if ( (*(_WORD *)(a4 + 2) & 0x3Fu) - 32 > 1 )
            goto LABEL_26;
          if ( *(_BYTE *)a3 != 44 )
            goto LABEL_26;
          v25 = *(unsigned __int8 **)(a3 - 64);
          v116 = v24;
          if ( !v25 )
            goto LABEL_26;
          v26 = *(unsigned __int8 **)(a3 - 32);
          v146 = *(unsigned __int8 **)(a3 - 64);
          if ( !v26 )
            goto LABEL_26;
          v150 = (unsigned __int64)v26;
          v155 = *(unsigned __int8 **)(a2 + 248);
          v156 = (unsigned __int64 *)&v146;
          sub_22BEBD0((__int64)&v155, v25);
          v27 = (unsigned __int8 *)v150;
          v155 = *(unsigned __int8 **)(a2 + 248);
          v156 = &v150;
          sub_22BEBD0((__int64)&v155, (unsigned __int8 *)v150);
          if ( (v146 != (unsigned __int8 *)v9 || v150 != v10) && (v146 != (unsigned __int8 *)v10 || v150 != v9) )
          {
LABEL_26:
            *(_BYTE *)(a1 + 40) = 1;
            *(_WORD *)a1 = 6;
            LOWORD(v155) = 0;
            sub_22C0090((unsigned __int8 *)&v155);
            goto LABEL_14;
          }
          v28 = (unsigned __int8 *)sub_AD6530(*(_QWORD *)(a3 + 8), (__int64)v27);
          LOWORD(v155) = 0;
          if ( v116 == 32 )
            sub_22C0310((__int64)&v155, v28, 0);
          else
            sub_22C0430((__int64)&v155, v28);
          goto LABEL_42;
        }
        sub_9865C0((__int64)&v138, v134);
        v38 = v105;
        v39 = v94;
        v40 = v105;
        if ( (unsigned int)(v105 - 38) <= 1 )
        {
          v40 = sub_B52870(v105);
          v39 = v118;
          v38 = v105;
        }
        if ( v40 == 41 )
        {
          v83 = v139 - 1;
          if ( v139 <= 0x40 )
          {
            if ( v138 == (unsigned __int8 *)((1LL << v83) - 1) )
              goto LABEL_180;
          }
          else
          {
            v102 = v139 - 1;
            if ( (*(_QWORD *)&v138[8 * (v83 >> 6)] & (1LL << v83)) == 0 )
            {
              v111 = v39;
              v124 = v38;
              v84 = sub_C445E0((__int64)&v138);
              v38 = v124;
              v39 = v111;
              if ( v102 == v84 )
                goto LABEL_180;
            }
          }
          v112 = v39;
          v125 = v38;
          sub_C46250((__int64)&v138);
          v39 = v112;
          v38 = v125;
        }
        v41 = v136;
        v141 = v139;
        if ( v139 > 0x40 )
        {
          v91 = v136;
          v101 = v39;
          v110 = v38;
          sub_C43780((__int64)&v140, (const void **)&v138);
          v41 = v91;
          v39 = v101;
          v38 = v110;
        }
        else
        {
          v140 = v138;
        }
        v95 = v39;
        v106 = v38;
        sub_C47AC0((__int64)&v140, v41);
        v42 = v136;
        v43 = v106;
        v44 = v95;
        LODWORD(v156) = v141;
        if ( v141 > 0x40 )
        {
          v89 = v136;
          sub_C43780((__int64)&v155, (const void **)&v140);
          v42 = v89;
          v44 = v95;
          v43 = v106;
        }
        else
        {
          v155 = v140;
        }
        v96 = v44;
        v107 = v43;
        sub_C44D10((__int64)&v155, v42);
        v38 = v107;
        v45 = v96;
        if ( (unsigned int)v156 <= 0x40 )
        {
          v46 = v155 == v138;
        }
        else
        {
          v46 = sub_C43C50((__int64)&v155, (const void **)&v138);
          v38 = v107;
          v45 = v96;
          if ( v155 )
          {
            v88 = v46;
            j_j___libc_free_0_0((unsigned __int64)v155);
            v46 = v88;
            v45 = v96;
            v38 = v107;
          }
        }
        if ( !v46 )
        {
          v159 = 0;
LABEL_77:
          if ( v141 > 0x40 && v140 )
          {
            v108 = v45;
            v119 = v38;
            j_j___libc_free_0_0((unsigned __int64)v140);
            v45 = v108;
            v38 = v119;
          }
          if ( v159 )
          {
            if ( v45 )
            {
              v109 = v38;
              sub_ABB300((__int64)&v146, (__int64)&v155);
              v47 = v147;
              v147 = 0;
              v154 = 1;
              v151 = v47;
              v150 = (unsigned __int64)v146;
              v153 = v149;
              v152 = v148;
              sub_969240((__int64 *)&v146);
              v38 = v109;
            }
            else
            {
              v154 = 0;
              v151 = (unsigned int)v156;
              if ( (unsigned int)v156 > 0x40 )
              {
                v127 = v38;
                sub_C43780((__int64)&v150, (const void **)&v155);
                v38 = v127;
              }
              else
              {
                v150 = (unsigned __int64)v155;
              }
              v153 = v158;
              if ( v158 > 0x40 )
              {
                v126 = v38;
                sub_C43780((__int64)&v152, (const void **)&v157);
                v38 = v126;
              }
              else
              {
                v152 = v157;
              }
              v154 = 1;
            }
            if ( v159 )
            {
              v123 = v38;
              v159 = 0;
              sub_969240((__int64 *)&v157);
              sub_969240((__int64 *)&v155);
              v38 = v123;
            }
LABEL_85:
            v120 = v38;
            sub_969240((__int64 *)&v138);
            v24 = v120;
            if ( v154 )
            {
              sub_AAF450((__int64)&v146, (__int64)&v150);
              sub_22C06B0((__int64)&v155, (__int64)&v146, 0);
              sub_22C0650(a1, (unsigned __int8 *)&v155);
              *(_BYTE *)(a1 + 40) = 1;
              sub_22C0090((unsigned __int8 *)&v155);
              sub_969240((__int64 *)&v148);
              sub_969240((__int64 *)&v146);
              if ( v154 )
              {
                v154 = 0;
                if ( v153 > 0x40 && v152 )
                  j_j___libc_free_0_0((unsigned __int64)v152);
                if ( v151 > 0x40 && v150 )
                  j_j___libc_free_0_0(v150);
              }
              goto LABEL_14;
            }
            goto LABEL_25;
          }
LABEL_180:
          v154 = 0;
          goto LABEL_85;
        }
        v77 = v141;
        v145 = v141;
        if ( v141 > 0x40 )
        {
          v90 = v45;
          v99 = v38;
          sub_C43780((__int64)&v144, (const void **)&v140);
          v77 = v141;
          v38 = v99;
          v45 = v90;
          v143 = v141;
          if ( v141 > 0x40 )
          {
            v85 = v141;
            sub_C43690((__int64)&v142, 0, 0);
            v38 = v99;
            v45 = v90;
            v78 = 1LL << ((unsigned __int8)v85 - 1);
            if ( v143 > 0x40 )
            {
              *(_QWORD *)(v142 + 8LL * ((v85 - 1) >> 6)) |= v78;
              v79 = v143;
              if ( v143 > 0x40 )
              {
                v86 = v143;
                v82 = sub_C43C50((__int64)&v142, (const void **)&v144);
                v38 = v99;
                v45 = v90;
                v79 = v86;
                if ( v82 )
                  goto LABEL_189;
                goto LABEL_197;
              }
LABEL_188:
              if ( v142 == v144 )
              {
LABEL_189:
                v97 = v45;
                v121 = v38;
                sub_AADB10((__int64)&v150, v79, 1);
                v80 = v121;
                v81 = v97;
LABEL_190:
                v159 = 1;
                v98 = v81;
                LODWORD(v156) = v151;
                v122 = v80;
                v155 = (unsigned __int8 *)v150;
                v151 = 0;
                v158 = v153;
                v157 = v152;
                sub_969240((__int64 *)&v150);
                sub_969240((__int64 *)&v142);
                sub_969240((__int64 *)&v144);
                v45 = v98;
                v38 = v122;
                goto LABEL_77;
              }
LABEL_197:
              v147 = v79;
              v87 = v45;
              LODWORD(v156) = v145;
              v100 = v38;
              v155 = (unsigned __int8 *)v144;
              v146 = (unsigned __int8 *)v142;
              v145 = 0;
              v143 = 0;
              sub_AADC30((__int64)&v150, (__int64)&v146, (__int64 *)&v155);
              sub_969240((__int64 *)&v146);
              sub_969240((__int64 *)&v155);
              v81 = v87;
              v80 = v100;
              goto LABEL_190;
            }
LABEL_187:
            v142 |= v78;
            v79 = v143;
            goto LABEL_188;
          }
        }
        else
        {
          v143 = v141;
          v144 = (unsigned __int64)v140;
        }
        v142 = 0;
        v78 = 1LL << (v77 - 1);
        goto LABEL_187;
      }
    }
    else
    {
      if ( *(_BYTE *)v9 != 67 )
        goto LABEL_24;
      v48 = *(_QWORD *)(v9 - 32);
      if ( !v48 || a3 != v48 )
        goto LABEL_24;
    }
    v93 = v20;
    v117 = v21;
    v155 = (unsigned __int8 *)&v134;
    LOBYTE(v156) = 0;
    v33 = sub_991580((__int64)&v155, v10);
    v21 = v117;
    v20 = v93;
    if ( !v33 )
      goto LABEL_24;
    sub_AB1A50((__int64)&v146, v117, v134);
    if ( sub_AAF7D0((__int64)&v146) )
    {
      sub_969240((__int64 *)&v148);
      sub_969240((__int64 *)&v146);
      v20 = v93;
      v21 = v117;
      goto LABEL_24;
    }
    v143 = v103;
    if ( v103 > 0x40 )
      sub_C43690((__int64)&v142, 0, 0);
    else
      v142 = 0;
    sub_AB0A00((__int64)&v138, (__int64)&v146);
    sub_C449B0((__int64)&v140, (const void **)&v138, v103);
    v34 = v141;
    if ( v141 <= 0x40 )
    {
      if ( v140 == (unsigned __int8 *)v142 )
        goto LABEL_58;
    }
    else if ( sub_C43C50((__int64)&v140, (const void **)&v142) )
    {
LABEL_58:
      sub_AADB10((__int64)&v150, v34, 1);
LABEL_59:
      sub_22C06B0((__int64)&v155, (__int64)&v150, 0);
      sub_22C0650(a1, (unsigned __int8 *)&v155);
      *(_BYTE *)(a1 + 40) = 1;
      sub_22C0090((unsigned __int8 *)&v155);
      sub_969240((__int64 *)&v152);
      sub_969240((__int64 *)&v150);
      sub_969240((__int64 *)&v140);
      sub_969240((__int64 *)&v138);
      sub_969240((__int64 *)&v142);
      sub_969240((__int64 *)&v148);
      sub_969240((__int64 *)&v146);
      goto LABEL_14;
    }
    v76 = v143;
    v143 = 0;
    v145 = v34;
    LODWORD(v156) = v76;
    v141 = 0;
    v155 = (unsigned __int8 *)v142;
    v144 = (unsigned __int64)v140;
    sub_AADC30((__int64)&v150, (__int64)&v144, (__int64 *)&v155);
    if ( v145 > 0x40 && v144 )
      j_j___libc_free_0_0(v144);
    if ( (unsigned int)v156 > 0x40 && v155 )
      j_j___libc_free_0_0((unsigned __int64)v155);
    goto LABEL_59;
  }
  *(_BYTE *)(a1 + 40) = 1;
  *(_WORD *)a1 = 6;
  LOWORD(v155) = 0;
  sub_22C0090((unsigned __int8 *)&v155);
  return a1;
}
