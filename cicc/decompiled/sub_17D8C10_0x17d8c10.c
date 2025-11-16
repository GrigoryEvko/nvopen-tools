// Function: sub_17D8C10
// Address: 0x17d8c10
//
__int64 __fastcall sub_17D8C10(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  char v5; // al
  _QWORD *v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // r15
  __int64 v9; // rax
  __int64 *v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rbx
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  __int128 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rsi
  __int64 *v19; // rax
  __int64 **v20; // rdx
  unsigned __int64 v21; // rbx
  __int64 v22; // r15
  _QWORD *v23; // rdi
  __int64 v24; // rax
  int v25; // ebx
  _QWORD *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rax
  unsigned int v37; // r9d
  unsigned int v38; // r15d
  unsigned int v39; // eax
  __int64 *v40; // rax
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdx
  unsigned __int8 *v50; // rsi
  __int64 v51; // rax
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rdx
  unsigned __int8 *v56; // rsi
  unsigned __int64 v57; // rdx
  unsigned __int64 v58; // r15
  __int64 v59; // rsi
  __int64 *v60; // rax
  __int64 v61; // rax
  __int64 v62; // r15
  __int64 v63; // rax
  _QWORD *v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rsi
  __int64 *v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  _QWORD *v70; // r15
  __int64 v71; // rax
  __int64 result; // rax
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // r15
  _QWORD *v80; // rax
  __int64 v81; // rax
  __int64 v82; // rbx
  __int64 v83; // r14
  unsigned __int64 i; // r13
  __int64 *v85; // r15
  unsigned __int64 v86; // rax
  __int128 v87; // rdi
  __int64 v88; // rsi
  __int64 *v89; // rax
  __int64 v90; // rcx
  __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rdx
  unsigned __int8 *v97; // rsi
  const char *v98; // rsi
  __int64 v99; // r15
  __int64 v100; // r14
  __int64 v101; // r10
  __int64 v102; // r14
  __int64 v103; // rdx
  __int64 v104; // rcx
  __int64 *v105; // rdi
  __int64 **v106; // rax
  __int64 v107; // rsi
  __int64 v108; // rdx
  __int64 v109; // rsi
  __int64 v110; // rax
  __int64 v111; // rsi
  __int64 v112; // rdx
  unsigned __int8 *v113; // rsi
  unsigned __int64 v114; // [rsp+18h] [rbp-198h]
  unsigned __int64 v115; // [rsp+20h] [rbp-190h]
  unsigned int v116; // [rsp+20h] [rbp-190h]
  __int64 *v117; // [rsp+20h] [rbp-190h]
  __int64 *v118; // [rsp+20h] [rbp-190h]
  unsigned __int64 v119; // [rsp+30h] [rbp-180h]
  __int64 *v120; // [rsp+38h] [rbp-178h]
  __int64 *v121; // [rsp+40h] [rbp-170h]
  _QWORD *v122; // [rsp+40h] [rbp-170h]
  int v123; // [rsp+48h] [rbp-168h]
  __int64 v124; // [rsp+48h] [rbp-168h]
  __int64 v125; // [rsp+48h] [rbp-168h]
  __int64 v126; // [rsp+48h] [rbp-168h]
  __int64 **v127; // [rsp+50h] [rbp-160h]
  __int64 v128; // [rsp+50h] [rbp-160h]
  __int64 v129; // [rsp+50h] [rbp-160h]
  __int64 v130; // [rsp+58h] [rbp-158h] BYREF
  _QWORD v131[2]; // [rsp+60h] [rbp-150h] BYREF
  __int16 v132; // [rsp+70h] [rbp-140h]
  __int64 v133; // [rsp+80h] [rbp-130h] BYREF
  __int64 v134; // [rsp+88h] [rbp-128h]
  __int64 *v135; // [rsp+90h] [rbp-120h]
  _QWORD *v136; // [rsp+98h] [rbp-118h]
  unsigned __int8 *v137[2]; // [rsp+D0h] [rbp-E0h] BYREF
  __int16 v138; // [rsp+E0h] [rbp-D0h]
  const char *v139; // [rsp+120h] [rbp-90h] BYREF
  __int64 v140; // [rsp+128h] [rbp-88h]
  __int64 *v141; // [rsp+130h] [rbp-80h] BYREF
  _QWORD *v142; // [rsp+138h] [rbp-78h]
  __int64 **v143; // [rsp+140h] [rbp-70h]
  __int64 **v144; // [rsp+148h] [rbp-68h]
  __int64 v145; // [rsp+150h] [rbp-60h]
  __int64 v146; // [rsp+158h] [rbp-58h]
  __int64 v147; // [rsp+160h] [rbp-50h]
  __int64 v148; // [rsp+168h] [rbp-48h]
  __int64 v149; // [rsp+170h] [rbp-40h]
  __int64 v150; // [rsp+178h] [rbp-38h]

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v130 = a2;
  v114 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (a2 & 4) == 0 )
    goto LABEL_6;
  v4 = *(_QWORD *)(v2 - 24);
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 != 20 )
  {
    if ( !v5 )
    {
      v139 = 0;
      LODWORD(v141) = 0;
      v142 = 0;
      v143 = &v141;
      v144 = &v141;
      v145 = 0;
      v146 = 0;
      v147 = 0;
      v148 = 0;
      v149 = 0;
      v150 = 0;
      v6 = sub_15606E0(&v139, 37);
      sub_15606E0(v6, 36);
      sub_15E0EF0(v4, -1, &v139);
      sub_17CCFA0(v142);
    }
    sub_1AED190(v114, *(_QWORD *)(a1 + 472));
LABEL_6:
    sub_17CE510((__int64)&v133, v114, 0, 0, 0);
    v127 = (__int64 **)((v130 & 0xFFFFFFFFFFFFFFF8LL)
                      - 24LL * (*(_DWORD *)((v130 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
    v119 = sub_1389B50(&v130);
    if ( (__int64 **)v119 != v127 )
    {
      v7 = v130;
      v123 = 0;
      while ( 1 )
      {
        v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
        v10 = *v127;
        v11 = **v127;
        v12 = *(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF;
        v13 = *(unsigned __int8 *)(v11 + 8);
        if ( (unsigned __int8)v13 <= 0xFu )
        {
          v14 = 35454;
          if ( _bittest64(&v14, v13) )
            break;
        }
        if ( (unsigned int)(v13 - 13) > 1 && (_DWORD)v13 != 16 )
        {
          v9 = (v7 >> 2) & 1;
          goto LABEL_12;
        }
        if ( sub_16435F0(v11, 0) )
          break;
LABEL_11:
        v7 = v130;
        v8 = v130 & 0xFFFFFFFFFFFFFFF8LL;
        v9 = (v130 >> 2) & 1;
LABEL_12:
        v127 += 3;
        if ( (__int64 **)v119 == v127 )
          goto LABEL_59;
      }
      *((_QWORD *)&v15 + 1) = v10;
      *(_QWORD *)&v15 = a1;
      v121 = sub_17D4DA0(v15);
      v138 = 257;
      v16 = *(_QWORD *)(a1 + 8);
      v17 = *(_QWORD *)(v16 + 192);
      v18 = *(_QWORD *)(v16 + 176);
      if ( v18 != *(_QWORD *)v17 )
      {
        if ( *(_BYTE *)(v17 + 16) > 0x10u )
        {
          v44 = *(_QWORD *)(v16 + 192);
          LOWORD(v141) = 257;
          v45 = sub_15FDFF0(v44, v18, (__int64)&v139, 0);
          v17 = v45;
          if ( v134 )
          {
            v117 = v135;
            sub_157E9D0(v134 + 40, v45);
            v46 = *v117;
            v47 = *(_QWORD *)(v17 + 24) & 7LL;
            *(_QWORD *)(v17 + 32) = v117;
            v46 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v17 + 24) = v46 | v47;
            *(_QWORD *)(v46 + 8) = v17 + 24;
            *v117 = *v117 & 7 | (v17 + 24);
          }
          sub_164B780(v17, (__int64 *)v137);
          if ( v133 )
          {
            v131[0] = v133;
            sub_1623A60((__int64)v131, v133, 2);
            v48 = *(_QWORD *)(v17 + 48);
            v49 = v17 + 48;
            if ( v48 )
            {
              sub_161E7C0(v17 + 48, v48);
              v49 = v17 + 48;
            }
            v50 = (unsigned __int8 *)v131[0];
            *(_QWORD *)(v17 + 48) = v131[0];
            if ( v50 )
              sub_1623210((__int64)v131, v50, v49);
          }
        }
        else
        {
          v17 = sub_15A4A70(*(__int64 ****)(v16 + 192), v18);
        }
      }
      if ( v123 )
      {
        LOWORD(v141) = 257;
        v43 = sub_15A0680(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 176LL), v123, 0);
        v17 = sub_12899C0(&v133, v17, v43, (__int64)&v139, 0, 0);
      }
      v137[0] = "_msarg";
      v138 = 259;
      v19 = sub_17CD8D0((_QWORD *)a1, *v10);
      v20 = (__int64 **)sub_1646BA0(v19, 0);
      if ( v20 != *(__int64 ***)v17 )
      {
        if ( *(_BYTE *)(v17 + 16) > 0x10u )
        {
          LOWORD(v141) = 257;
          v51 = sub_15FDBD0(46, v17, (__int64)v20, (__int64)&v139, 0);
          v17 = v51;
          if ( v134 )
          {
            v118 = v135;
            sub_157E9D0(v134 + 40, v51);
            v52 = *v118;
            v53 = *(_QWORD *)(v17 + 24) & 7LL;
            *(_QWORD *)(v17 + 32) = v118;
            v52 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v17 + 24) = v52 | v53;
            *(_QWORD *)(v52 + 8) = v17 + 24;
            *v118 = *v118 & 7 | (v17 + 24);
          }
          sub_164B780(v17, (__int64 *)v137);
          if ( v133 )
          {
            v131[0] = v133;
            sub_1623A60((__int64)v131, v133, 2);
            v54 = *(_QWORD *)(v17 + 48);
            v55 = v17 + 48;
            if ( v54 )
            {
              sub_161E7C0(v17 + 48, v54);
              v55 = v17 + 48;
            }
            v56 = (unsigned __int8 *)v131[0];
            *(_QWORD *)(v17 + 48) = v131[0];
            if ( v56 )
              sub_1623210((__int64)v131, v56, v55);
          }
        }
        else
        {
          v17 = sub_15A46C0(46, (__int64 ***)v17, v20, 0);
        }
      }
      v21 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)&v127[3 * v12] - v8) >> 3);
      v22 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
      v115 = v130 & 0xFFFFFFFFFFFFFFF8LL;
      v23 = (_QWORD *)((v130 & 0xFFFFFFFFFFFFFFF8LL) + 56);
      if ( (v130 & 4) != 0 )
      {
        if ( (unsigned __int8)sub_1560290(v23, v21, 6) )
          goto LABEL_35;
        v24 = *(_QWORD *)(v115 - 24);
        if ( *(_BYTE *)(v24 + 16) )
          goto LABEL_27;
      }
      else
      {
        if ( (unsigned __int8)sub_1560290(v23, v21, 6) )
          goto LABEL_35;
        v24 = *(_QWORD *)(v115 - 72);
        if ( *(_BYTE *)(v24 + 16) )
        {
LABEL_27:
          v25 = sub_12BE0A0(v22, *v10);
          if ( (unsigned int)(v25 + v123) > 0x320 )
            goto LABEL_58;
          v26 = sub_12A8F50(&v133, (__int64)v121, v17, 0);
          sub_15F9450((__int64)v26, 8u);
          if ( *((_BYTE *)v121 + 16) <= 0x10u && sub_1593BB0((__int64)v121, 8, v27, v28) )
            goto LABEL_34;
          v29 = *(_QWORD *)(a1 + 8);
          if ( !*(_DWORD *)(v29 + 156) )
            goto LABEL_34;
          goto LABEL_31;
        }
      }
      v139 = *(const char **)(v24 + 112);
      if ( !(unsigned __int8)sub_1560290(&v139, v21, 6) )
        goto LABEL_27;
LABEL_35:
      v37 = sub_12BE0A0(v22, **(_QWORD **)(*v10 + 16));
      if ( v37 + v123 > 0x320 )
        goto LABEL_58;
      v116 = v37;
      v38 = 8;
      v39 = sub_15603A0((_QWORD *)((v130 & 0xFFFFFFFFFFFFFFF8LL) + 56), v21);
      if ( v39 <= 8 )
        v38 = v39;
      v40 = (__int64 *)sub_1643330(v136);
      v122 = (_QWORD *)sub_17CFB40(a1, (__int64)v10, &v133, v40, v38);
      v25 = v116;
      v41 = sub_1643360(v136);
      v42 = (__int64 *)sub_159C470(v41, v116, 0);
      sub_15E7430(&v133, (_QWORD *)v17, v38, v122, v38, v42, 0, 0, 0, 0, 0);
      v29 = *(_QWORD *)(a1 + 8);
      if ( !*(_DWORD *)(v29 + 156) )
        goto LABEL_34;
LABEL_31:
      LOWORD(v141) = 257;
      v30 = sub_12A95D0(&v133, *(_QWORD *)(v29 + 200), *(_QWORD *)(v29 + 176), (__int64)&v139);
      if ( v123 )
      {
        v77 = *(_QWORD *)(a1 + 8);
        LOWORD(v141) = 257;
        v78 = sub_15A0680(*(_QWORD *)(v77 + 176), v123, 0);
        v30 = sub_12899C0(&v133, v30, v78, (__int64)&v139, 0, 0);
      }
      v139 = "_msarg_o";
      v31 = *(_QWORD *)(a1 + 8);
      LOWORD(v141) = 259;
      v32 = sub_1646BA0(*(__int64 **)(v31 + 184), 0);
      v33 = sub_12AA3B0(&v133, 0x2Eu, v30, v32, (__int64)&v139);
      v36 = sub_17D4880(a1, (const char *)v10, v34, v35);
      sub_12A8F50(&v133, v36, v33, 0);
LABEL_34:
      v123 += (v25 + 7) & 0xFFFFFFF8;
      goto LABEL_11;
    }
LABEL_58:
    v8 = v130 & 0xFFFFFFFFFFFFFFF8LL;
    v9 = (v130 >> 2) & 1;
LABEL_59:
    v57 = v8 - 24;
    v58 = v8 - 72;
    if ( (_BYTE)v9 )
      v58 = v57;
    if ( *(_DWORD *)(**(_QWORD **)(**(_QWORD **)v58 + 16LL) + 8LL) >> 8 )
      (*(void (__fastcall **)(_QWORD, __int64 *, __int64 *))(**(_QWORD **)(a1 + 464) + 16LL))(
        *(_QWORD *)(a1 + 464),
        &v130,
        &v133);
    if ( sub_1704BC0(*(_QWORD *)v114, 0) && ((v130 & 4) == 0 || (*(_WORD *)(v114 + 18) & 3) != 2) )
    {
      sub_17CE510((__int64)v137, v114, 0, 0, 0);
      v59 = *(_QWORD *)v114;
      v139 = "_msret";
      LOWORD(v141) = 259;
      v60 = sub_17CD8D0((_QWORD *)a1, v59);
      v61 = sub_1646BA0(v60, 0);
      v62 = sub_12A95D0((__int64 *)v137, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 208LL), v61, (__int64)&v139);
      v63 = sub_17CDAE0((_QWORD *)a1, *(_QWORD *)v114);
      v64 = sub_12A8F50((__int64 *)v137, v63, v62, 0);
      sub_15F9450((__int64)v64, 8u);
      if ( (v130 & 4) != 0 )
      {
        v65 = *(_QWORD *)(v114 + 32);
        if ( !v65 )
        {
LABEL_69:
          sub_17CE510((__int64)&v139, v65, 0, 0, 0);
          v66 = *(_QWORD *)v114;
          v131[0] = "_msret";
          v132 = 259;
          v67 = sub_17CD8D0((_QWORD *)a1, v66);
          v68 = sub_1646BA0(v67, 0);
          v69 = sub_12A95D0((__int64 *)&v139, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 208LL), v68, (__int64)v131);
          v70 = sub_17D3810((__int64 *)&v139, v69, "_msret");
          sub_15F8F50((__int64)v70, 8u);
          sub_17D4920(a1, (__int64 *)v114, (__int64)v70);
          v71 = *(_QWORD *)(a1 + 8);
          if ( *(_DWORD *)(v71 + 156) )
          {
            v132 = 257;
            v80 = sub_156E5B0((__int64 *)&v139, *(_QWORD *)(v71 + 216), (__int64)v131);
            sub_17D4B80(a1, v114, (__int64)v80);
          }
          sub_17CD270((__int64 *)&v139);
          sub_17CD270((__int64 *)v137);
          return sub_17CD270(&v133);
        }
LABEL_68:
        v65 -= 24;
        goto LABEL_69;
      }
      v79 = *(_QWORD *)(v114 - 48);
      if ( sub_157F0B0(v79) )
      {
        v65 = sub_157EE30(v79);
        if ( !v65 )
          goto LABEL_69;
        goto LABEL_68;
      }
      v73 = sub_17CDAE0((_QWORD *)a1, *(_QWORD *)v114);
      sub_17D4920(a1, (__int64 *)v114, v73);
      v76 = sub_15A06D0(*(__int64 ***)(*(_QWORD *)(a1 + 8) + 184LL), v114, v74, v75);
      sub_17D4B80(a1, v114, v76);
      sub_17CD270((__int64 *)v137);
    }
    return sub_17CD270(&v133);
  }
  if ( !byte_4FA46E0 )
    return sub_17D7760((_QWORD *)a1, v114);
  if ( *(_BYTE *)(v2 + 16) != 78 )
    BUG();
  v81 = *(_DWORD *)(v2 + 20) & 0xFFFFFFF;
  if ( (*(_DWORD *)(v2 + 20) & 0xFFFFFFF) != 0 )
  {
    v128 = *(_DWORD *)(v2 + 20) & 0xFFFFFFF;
    v82 = 0;
    v83 = 35454;
    for ( i = a2 & 0xFFFFFFFFFFFFFFF8LL; ; v81 = *(_DWORD *)(i + 20) & 0xFFFFFFF )
    {
      v85 = *(__int64 **)(i + 24 * (v82 - v81));
      v86 = *(unsigned __int8 *)(*v85 + 8);
      if ( (unsigned __int8)v86 <= 0xFu && _bittest64(&v83, v86)
        || ((unsigned int)(v86 - 13) <= 1 || (_DWORD)v86 == 16) && sub_16435F0(*v85, 0) )
      {
        *((_QWORD *)&v87 + 1) = v85;
        *(_QWORD *)&v87 = a1;
        sub_17D5820(v87, i);
      }
      if ( v128 == ++v82 )
        break;
    }
  }
  v88 = *(_QWORD *)v114;
  v89 = sub_17CD8D0((_QWORD *)a1, *(_QWORD *)v114);
  v91 = (__int64)v89;
  if ( v89 )
    v91 = sub_15A06D0((__int64 **)v89, v88, (__int64)v89, v90);
  sub_17D4920(a1, (__int64 *)v114, v91);
  v94 = sub_15A06D0(*(__int64 ***)(*(_QWORD *)(a1 + 8) + 184LL), v114, v92, v93);
  sub_17D4B80(a1, v114, v94);
  sub_17CE510((__int64)&v139, v114, 0, 0, 0);
  v95 = *(_QWORD *)(v114 + 32);
  if ( v95 == *(_QWORD *)(v114 + 40) + 40LL || !v95 )
    BUG();
  v96 = *(_QWORD *)(v95 + 16);
  v141 = *(__int64 **)(v114 + 32);
  v140 = v96;
  v97 = *(unsigned __int8 **)(v95 + 24);
  v137[0] = v97;
  if ( v97 )
  {
    sub_1623A60((__int64)v137, (__int64)v97, 2);
    v98 = v139;
    if ( !v139 )
      goto LABEL_97;
    goto LABEL_96;
  }
  v98 = v139;
  if ( v139 )
  {
LABEL_96:
    sub_161E7C0((__int64)&v139, (__int64)v98);
LABEL_97:
    v139 = (const char *)v137[0];
    if ( v137[0] )
      sub_1623210((__int64)v137, v137[0], (__int64)&v139);
  }
  result = *(_DWORD *)(v114 + 20) & 0xFFFFFFF;
  if ( (*(_DWORD *)(v114 + 20) & 0xFFFFFFF) != 0 )
  {
    v129 = *(_DWORD *)(v114 + 20) & 0xFFFFFFF;
    v99 = 0;
    while ( 1 )
    {
      v100 = *(_QWORD *)(v114 + 24 * (v99 - result));
      result = *(_QWORD *)v100;
      if ( *(_BYTE *)(*(_QWORD *)v100 + 8LL) == 15 )
      {
        if ( (v101 = **(_QWORD **)(result + 16), result = *(unsigned __int8 *)(v101 + 8),
                                                 (unsigned __int8)result <= 0xFu)
          && (v108 = 35454, _bittest64(&v108, result))
          || ((unsigned int)(result - 13) <= 1 || (_DWORD)result == 16)
          && (v124 = v101, result = sub_16435F0(v101, 0), v101 = v124, (_BYTE)result) )
        {
          v125 = v101;
          v102 = sub_17CFB40(a1, v100, (__int64 *)&v139, (__int64 *)v101, 1u);
          v105 = sub_17CD8D0((_QWORD *)a1, v125);
          if ( !v105 )
          {
            LOWORD(v135) = 257;
            BUG();
          }
          v106 = (__int64 **)sub_15A06D0((__int64 **)v105, v125, v103, v104);
          LOWORD(v135) = 257;
          v126 = (__int64)v106;
          v107 = sub_1647190(*v106, 0);
          if ( v107 != *(_QWORD *)v102 )
          {
            if ( *(_BYTE *)(v102 + 16) > 0x10u )
            {
              v138 = 257;
              v102 = sub_15FDFF0(v102, v107, (__int64)v137, 0);
              if ( v140 )
              {
                v120 = v141;
                sub_157E9D0(v140 + 40, v102);
                v109 = *v120;
                v110 = *(_QWORD *)(v102 + 24) & 7LL;
                *(_QWORD *)(v102 + 32) = v120;
                v109 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v102 + 24) = v109 | v110;
                *(_QWORD *)(v109 + 8) = v102 + 24;
                *v120 = *v120 & 7 | (v102 + 24);
              }
              sub_164B780(v102, &v133);
              if ( v139 )
              {
                v131[0] = v139;
                sub_1623A60((__int64)v131, (__int64)v139, 2);
                v111 = *(_QWORD *)(v102 + 48);
                v112 = v102 + 48;
                if ( v111 )
                {
                  sub_161E7C0(v102 + 48, v111);
                  v112 = v102 + 48;
                }
                v113 = (unsigned __int8 *)v131[0];
                *(_QWORD *)(v102 + 48) = v131[0];
                if ( v113 )
                  sub_1623210((__int64)v131, v113, v112);
              }
            }
            else
            {
              v102 = sub_15A4A70((__int64 ***)v102, v107);
            }
          }
          result = (__int64)sub_12A8F50((__int64 *)&v139, v126, v102, 0);
        }
      }
      if ( v129 == ++v99 )
        break;
      result = *(_DWORD *)(v114 + 20) & 0xFFFFFFF;
    }
  }
  if ( v139 )
    return sub_161E7C0((__int64)&v139, (__int64)v139);
  return result;
}
