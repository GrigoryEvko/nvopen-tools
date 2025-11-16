// Function: sub_13E1350
// Address: 0x13e1350
//
__int64 __fastcall sub_13E1350(__int64 a1, __int64 **a2, __int64 a3, __int64 *a4)
{
  __int64 **v5; // rbx
  __int64 v6; // rax
  unsigned int v8; // r15d
  __int64 v9; // r14
  __int64 **v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  unsigned __int64 v13; // rbx
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 **v16; // rcx
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 **v19; // rax
  __int64 **v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 *v23; // rbx
  __int64 v24; // rax
  __int64 *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // r9
  __int64 *v29; // r12
  __int64 *v30; // r14
  __int64 v31; // r15
  char v32; // al
  _QWORD *v33; // r9
  __int64 *v34; // r15
  char v35; // al
  __int64 *v36; // r9
  char v37; // al
  __int64 *v38; // r9
  char v39; // al
  __int64 *v40; // r9
  __int64 v41; // rax
  unsigned int v42; // r14d
  __int64 **v43; // rbx
  __int64 v44; // rax
  __int64 *v45; // rcx
  char v46; // al
  __int64 *v47; // r14
  int v48; // eax
  bool v49; // al
  __int64 v50; // rdi
  int v51; // eax
  bool v52; // al
  __int64 v53; // rdi
  int v54; // eax
  __int64 v55; // rdi
  int v56; // eax
  bool v57; // al
  __int64 v58; // rax
  __int64 v59; // rdx
  int v60; // eax
  __int64 v61; // rax
  __int64 v62; // rdx
  int v63; // eax
  bool v64; // al
  __int64 v65; // rax
  __int64 v66; // rdx
  int v67; // eax
  __int64 v68; // rdx
  __int64 v69; // r12
  unsigned __int64 v70; // rbx
  unsigned __int64 v71; // rax
  int v72; // ebx
  int v73; // eax
  __int64 v74; // rdx
  __int64 **v75; // rax
  __int64 v76; // r12
  char v77; // cl
  __int64 v78; // rax
  unsigned int v79; // r15d
  int v80; // eax
  int v81; // ebx
  int v82; // eax
  __int64 **v83; // rax
  __int64 v84; // rsi
  bool v85; // al
  __int64 **v86; // rax
  __int64 v87; // rbx
  unsigned int v88; // r12d
  __int64 v89; // rax
  char v90; // si
  bool v91; // al
  unsigned int v92; // r12d
  __int64 v93; // rax
  char v94; // cl
  bool v95; // al
  __int64 v96; // rbx
  unsigned int v97; // r12d
  __int64 v98; // rax
  char v99; // si
  bool v100; // al
  __int64 v101; // rbx
  unsigned int v102; // r12d
  __int64 v103; // rax
  char v104; // si
  bool v105; // al
  bool v106; // al
  __int64 v107; // rbx
  bool v108; // al
  char v109; // al
  int v110; // eax
  __int64 v111; // r12
  __int64 v112; // r15
  __int64 v113; // rax
  bool v114; // al
  bool v115; // al
  __int64 v116; // rbx
  _QWORD *v117; // rax
  __int64 v118; // rax
  unsigned int v119; // r15d
  unsigned int v120; // r14d
  __int64 v121; // rax
  char v122; // cl
  bool v123; // al
  int v124; // [rsp+Ch] [rbp-E4h]
  int v125; // [rsp+Ch] [rbp-E4h]
  int v126; // [rsp+Ch] [rbp-E4h]
  __int64 *v127; // [rsp+10h] [rbp-E0h]
  int v128; // [rsp+10h] [rbp-E0h]
  __int64 *v129; // [rsp+10h] [rbp-E0h]
  __int64 *v130; // [rsp+10h] [rbp-E0h]
  __int64 *v131; // [rsp+18h] [rbp-D8h]
  _QWORD *v132; // [rsp+18h] [rbp-D8h]
  __int64 *v133; // [rsp+18h] [rbp-D8h]
  __int64 *v134; // [rsp+18h] [rbp-D8h]
  __int64 *v135; // [rsp+20h] [rbp-D0h]
  int v136; // [rsp+20h] [rbp-D0h]
  int v137; // [rsp+28h] [rbp-C8h]
  _QWORD *v138; // [rsp+30h] [rbp-C0h]
  __int64 *v139; // [rsp+38h] [rbp-B8h]
  __int64 *v140; // [rsp+38h] [rbp-B8h]
  __int64 *v141; // [rsp+38h] [rbp-B8h]
  _QWORD *v142; // [rsp+38h] [rbp-B8h]
  __int64 *v143; // [rsp+38h] [rbp-B8h]
  __int64 *v144; // [rsp+38h] [rbp-B8h]
  __int64 *v145; // [rsp+38h] [rbp-B8h]
  __int64 *v146; // [rsp+38h] [rbp-B8h]
  __int64 *v147; // [rsp+38h] [rbp-B8h]
  _QWORD *v148; // [rsp+40h] [rbp-B0h]
  __int64 v149; // [rsp+40h] [rbp-B0h]
  __int64 v150; // [rsp+40h] [rbp-B0h]
  __int64 v151; // [rsp+40h] [rbp-B0h]
  int v152; // [rsp+40h] [rbp-B0h]
  int v153; // [rsp+40h] [rbp-B0h]
  int v154; // [rsp+40h] [rbp-B0h]
  int v155; // [rsp+40h] [rbp-B0h]
  int v156; // [rsp+40h] [rbp-B0h]
  int v157; // [rsp+40h] [rbp-B0h]
  int v158; // [rsp+40h] [rbp-B0h]
  int v159; // [rsp+40h] [rbp-B0h]
  int v160; // [rsp+40h] [rbp-B0h]
  int v161; // [rsp+40h] [rbp-B0h]
  int v162; // [rsp+40h] [rbp-B0h]
  __int64 v163; // [rsp+40h] [rbp-B0h]
  __int64 v164; // [rsp+40h] [rbp-B0h]
  __int64 *v165; // [rsp+40h] [rbp-B0h]
  int v166; // [rsp+40h] [rbp-B0h]
  __int64 v167; // [rsp+40h] [rbp-B0h]
  int v168; // [rsp+40h] [rbp-B0h]
  __int64 *v169; // [rsp+48h] [rbp-A8h]
  int v171; // [rsp+58h] [rbp-98h]
  __int64 v173; // [rsp+60h] [rbp-90h]
  __int64 *v174; // [rsp+60h] [rbp-90h]
  __int64 *v175; // [rsp+60h] [rbp-90h]
  __int64 v176; // [rsp+68h] [rbp-88h]
  unsigned __int64 v177; // [rsp+68h] [rbp-88h]
  __int64 v178; // [rsp+78h] [rbp-78h] BYREF
  __int64 v179; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v180; // [rsp+88h] [rbp-68h]
  __int64 *v181; // [rsp+90h] [rbp-60h] BYREF
  int v182; // [rsp+98h] [rbp-58h]
  __int64 *v183; // [rsp+A0h] [rbp-50h] BYREF
  _QWORD *v184; // [rsp+A8h] [rbp-48h] BYREF
  __int64 *v185; // [rsp+B0h] [rbp-40h]

  v5 = (__int64 **)*a2;
  v171 = a1;
  v6 = **a2;
  if ( *(_BYTE *)(v6 + 8) == 16 )
    v6 = **(_QWORD **)(v6 + 16);
  if ( a3 == 1 )
    return (__int64)v5;
  v137 = a3 - 1;
  v8 = *(_DWORD *)(v6 + 8) >> 8;
  v138 = a2 + 1;
  v9 = sub_15F9F50(a1, a2 + 1, a3 - 1);
  v169 = (__int64 *)sub_1646BA0(v9, v8);
  v10 = (__int64 **)*a2;
  v11 = **a2;
  if ( *(_BYTE *)(v11 + 8) == 16 || (v11 = *a2[1], *(_BYTE *)(v11 + 8) == 16) )
  {
    v169 = (__int64 *)sub_16463B0(v169, *(_QWORD *)(v11 + 32));
    v10 = (__int64 **)*a2;
  }
  if ( *((_BYTE *)v10 + 16) != 9 )
  {
    if ( a3 != 2 )
      goto LABEL_9;
    if ( sub_13CD190((__int64)a2[1]) )
    {
      v5 = (__int64 **)*a2;
      if ( v169 == (__int64 *)**a2 )
        return (__int64)v5;
    }
    v22 = *(unsigned __int8 *)(a1 + 8);
    if ( (unsigned __int8)v22 > 0xFu || (v68 = 35454, !_bittest64(&v68, v22)) )
    {
      if ( (unsigned int)(v22 - 13) > 1 && (_DWORD)v22 != 16 || !(unsigned __int8)sub_16435F0(a1, 0) )
        goto LABEL_9;
    }
    v69 = *a4;
    v70 = (unsigned int)sub_15A9FE0(*a4, a1);
    v71 = v70 * ((v70 + ((unsigned __int64)(sub_127FA20(v69, a1) + 7) >> 3) - 1) / v70);
    if ( v71 )
    {
      v177 = v71;
      v81 = sub_16431D0(*a2[1]);
      v82 = sub_15A95A0(*a4, v8);
      v74 = v177;
      if ( v81 != 8 * v82 )
        goto LABEL_9;
      v181 = v169;
      if ( v177 == 1 )
      {
        v83 = (__int64 **)*a2;
        v84 = (__int64)a2[1];
        v183 = &v178;
        v184 = v83;
        v85 = sub_13D62D0((__int64)&v183, v84);
        v74 = 1;
        if ( v85 )
        {
          v86 = (__int64 **)sub_13CD570(&v181, v178);
          v74 = 1;
          v5 = v86;
          if ( v86 )
            return (__int64)v5;
        }
      }
    }
    else
    {
      v5 = (__int64 **)*a2;
      if ( v169 == (__int64 *)**a2 )
        return (__int64)v5;
      v72 = sub_16431D0(*a2[1]);
      v73 = sub_15A95A0(*a4, v8);
      v74 = 0;
      if ( v72 != 8 * v73 )
      {
LABEL_9:
        v12 = *a4;
        v13 = (unsigned int)sub_15A9FE0(*a4, v9);
        if ( v13 * ((v13 + ((unsigned __int64)(sub_127FA20(v12, v9) + 7) >> 3) - 1) / v13) != 1 )
        {
LABEL_10:
          v14 = a3;
          goto LABEL_11;
        }
        v23 = (__int64 *)(a2 + 1);
        v24 = 8 * a3 - 16;
        v25 = (_QWORD *)((char *)v138 + v24);
        v26 = v24 >> 5;
        v135 = v25;
        v27 = (8 * a3 - 16) >> 3;
        if ( v26 > 0 )
        {
          v28 = a2 + 4;
          v29 = (__int64 *)(a2 + 3);
          v30 = (__int64 *)(a2 + 2);
          v176 = (__int64)&a2[4 * v26 + 1];
          while ( 1 )
          {
            v31 = *v23;
            v148 = v28;
            if ( *(_BYTE *)(*v23 + 16) > 0x10u )
              goto LABEL_59;
            v32 = sub_1593BB0(v31);
            v33 = v148;
            if ( !v32 )
            {
              if ( *(_BYTE *)(v31 + 16) == 13 )
              {
                if ( *(_DWORD *)(v31 + 32) <= 0x40u )
                {
                  v49 = *(_QWORD *)(v31 + 24) == 0;
                }
                else
                {
                  v142 = v148;
                  v152 = *(_DWORD *)(v31 + 32);
                  v48 = sub_16A57B0(v31 + 24);
                  v33 = v142;
                  v49 = v152 == v48;
                }
                goto LABEL_65;
              }
              if ( *(_BYTE *)(*(_QWORD *)v31 + 8LL) != 16 )
                goto LABEL_59;
              v78 = sub_15A1020(v31);
              v33 = v148;
              if ( v78 && *(_BYTE *)(v78 + 16) == 13 )
              {
                v79 = *(_DWORD *)(v78 + 32);
                if ( v79 <= 0x40 )
                {
                  v49 = *(_QWORD *)(v78 + 24) == 0;
                }
                else
                {
                  v80 = sub_16A57B0(v78 + 24);
                  v33 = v148;
                  v49 = v79 == v80;
                }
LABEL_65:
                if ( !v49 )
                  goto LABEL_59;
                goto LABEL_36;
              }
              v160 = *(_QWORD *)(*(_QWORD *)v31 + 32LL);
              if ( v160 )
              {
                v132 = v33;
                v144 = v29;
                v92 = 0;
                while ( 1 )
                {
                  v93 = sub_15A0A60(v31, v92);
                  if ( !v93 )
                    goto LABEL_59;
                  v94 = *(_BYTE *)(v93 + 16);
                  if ( v94 != 9 )
                  {
                    if ( v94 != 13 )
                      goto LABEL_59;
                    if ( *(_DWORD *)(v93 + 32) <= 0x40u )
                    {
                      v95 = *(_QWORD *)(v93 + 24) == 0;
                    }
                    else
                    {
                      v128 = *(_DWORD *)(v93 + 32);
                      v95 = v128 == (unsigned int)sub_16A57B0(v93 + 24);
                    }
                    if ( !v95 )
                      goto LABEL_59;
                  }
                  if ( v160 == ++v92 )
                  {
                    v29 = v144;
                    v33 = v132;
                    break;
                  }
                }
              }
            }
LABEL_36:
            v34 = v30;
            if ( *(_BYTE *)(v23[1] + 16) > 0x10u )
              goto LABEL_60;
            v139 = v33;
            v149 = v23[1];
            v35 = sub_1593BB0(v149);
            v36 = v139;
            if ( v35 )
              goto LABEL_38;
            if ( *(_BYTE *)(v149 + 16) == 13 )
            {
              if ( *(_DWORD *)(v149 + 32) <= 0x40u )
              {
                v52 = *(_QWORD *)(v149 + 24) == 0;
              }
              else
              {
                v50 = v149 + 24;
                v153 = *(_DWORD *)(v149 + 32);
                v51 = sub_16A57B0(v50);
                v36 = v139;
                v52 = v153 == v51;
              }
              goto LABEL_70;
            }
            if ( *(_BYTE *)(*(_QWORD *)v149 + 8LL) != 16 )
              goto LABEL_60;
            v58 = sub_15A1020(v149);
            v59 = v149;
            v36 = v139;
            if ( v58 && *(_BYTE *)(v58 + 16) == 13 )
            {
              if ( *(_DWORD *)(v58 + 32) <= 0x40u )
              {
                v52 = *(_QWORD *)(v58 + 24) == 0;
              }
              else
              {
                v156 = *(_DWORD *)(v58 + 32);
                v60 = sub_16A57B0(v58 + 24);
                v36 = v139;
                v52 = v156 == v60;
              }
LABEL_70:
              if ( !v52 )
                goto LABEL_60;
              goto LABEL_38;
            }
            v159 = *(_QWORD *)(*(_QWORD *)v149 + 32LL);
            if ( v159 )
            {
              v127 = v139;
              v143 = v23;
              v87 = v59;
              v131 = v29;
              v88 = 0;
              while ( 1 )
              {
                v89 = sub_15A0A60(v87, v88);
                if ( !v89 )
                  goto LABEL_60;
                v90 = *(_BYTE *)(v89 + 16);
                if ( v90 != 9 )
                {
                  if ( v90 != 13 )
                    goto LABEL_60;
                  if ( *(_DWORD *)(v89 + 32) <= 0x40u )
                  {
                    v91 = *(_QWORD *)(v89 + 24) == 0;
                  }
                  else
                  {
                    v124 = *(_DWORD *)(v89 + 32);
                    v91 = v124 == (unsigned int)sub_16A57B0(v89 + 24);
                  }
                  if ( !v91 )
                    goto LABEL_60;
                }
                if ( v159 == ++v88 )
                {
                  v23 = v143;
                  v29 = v131;
                  v36 = v127;
                  break;
                }
              }
            }
LABEL_38:
            v34 = v29;
            if ( *(_BYTE *)(v23[2] + 16) > 0x10u )
              goto LABEL_60;
            v140 = v36;
            v150 = v23[2];
            v37 = sub_1593BB0(v150);
            v38 = v140;
            if ( !v37 )
            {
              if ( *(_BYTE *)(v150 + 16) == 13 )
              {
                if ( *(_DWORD *)(v150 + 32) <= 0x40u )
                {
                  if ( *(_QWORD *)(v150 + 24) )
                    goto LABEL_60;
                }
                else
                {
                  v53 = v150 + 24;
                  v154 = *(_DWORD *)(v150 + 32);
                  v54 = sub_16A57B0(v53);
                  v38 = v140;
                  if ( v154 != v54 )
                    goto LABEL_60;
                }
              }
              else
              {
                if ( *(_BYTE *)(*(_QWORD *)v150 + 8LL) != 16 )
                  goto LABEL_60;
                v61 = sub_15A1020(v150);
                v62 = v150;
                v38 = v140;
                if ( v61 && *(_BYTE *)(v61 + 16) == 13 )
                {
                  if ( *(_DWORD *)(v61 + 32) <= 0x40u )
                  {
                    v64 = *(_QWORD *)(v61 + 24) == 0;
                  }
                  else
                  {
                    v157 = *(_DWORD *)(v61 + 32);
                    v63 = sub_16A57B0(v61 + 24);
                    v38 = v140;
                    v64 = v157 == v63;
                  }
                  if ( !v64 )
                    goto LABEL_60;
                }
                else
                {
                  v161 = *(_QWORD *)(*(_QWORD *)v150 + 32LL);
                  if ( v161 )
                  {
                    v129 = v140;
                    v145 = v23;
                    v96 = v62;
                    v133 = v29;
                    v97 = 0;
                    while ( 1 )
                    {
                      v98 = sub_15A0A60(v96, v97);
                      if ( !v98 )
                        goto LABEL_60;
                      v99 = *(_BYTE *)(v98 + 16);
                      if ( v99 != 9 )
                      {
                        if ( v99 != 13 )
                          goto LABEL_60;
                        if ( *(_DWORD *)(v98 + 32) <= 0x40u )
                        {
                          v100 = *(_QWORD *)(v98 + 24) == 0;
                        }
                        else
                        {
                          v125 = *(_DWORD *)(v98 + 32);
                          v100 = v125 == (unsigned int)sub_16A57B0(v98 + 24);
                        }
                        if ( !v100 )
                          goto LABEL_60;
                      }
                      if ( v161 == ++v97 )
                      {
                        v23 = v145;
                        v29 = v133;
                        v38 = v129;
                        break;
                      }
                    }
                  }
                }
              }
            }
            v34 = v38;
            if ( *(_BYTE *)(v23[3] + 16) > 0x10u )
              goto LABEL_60;
            v141 = v38;
            v151 = v23[3];
            v39 = sub_1593BB0(v151);
            v40 = v141;
            if ( !v39 )
            {
              if ( *(_BYTE *)(v151 + 16) == 13 )
              {
                if ( *(_DWORD *)(v151 + 32) <= 0x40u )
                {
                  v57 = *(_QWORD *)(v151 + 24) == 0;
                }
                else
                {
                  v55 = v151 + 24;
                  v155 = *(_DWORD *)(v151 + 32);
                  v56 = sub_16A57B0(v55);
                  v40 = v141;
                  v57 = v155 == v56;
                }
                goto LABEL_79;
              }
              if ( *(_BYTE *)(*(_QWORD *)v151 + 8LL) != 16 )
                goto LABEL_60;
              v65 = sub_15A1020(v151);
              v66 = v151;
              v40 = v141;
              if ( v65 && *(_BYTE *)(v65 + 16) == 13 )
              {
                if ( *(_DWORD *)(v65 + 32) <= 0x40u )
                {
                  v57 = *(_QWORD *)(v65 + 24) == 0;
                }
                else
                {
                  v158 = *(_DWORD *)(v65 + 32);
                  v67 = sub_16A57B0(v65 + 24);
                  v40 = v141;
                  v57 = v158 == v67;
                }
LABEL_79:
                if ( !v57 )
                  goto LABEL_60;
                goto LABEL_42;
              }
              v162 = *(_QWORD *)(*(_QWORD *)v151 + 32LL);
              if ( v162 )
              {
                v130 = v141;
                v146 = v23;
                v101 = v66;
                v134 = v29;
                v102 = 0;
                while ( 1 )
                {
                  v103 = sub_15A0A60(v101, v102);
                  if ( !v103 )
                    goto LABEL_60;
                  v104 = *(_BYTE *)(v103 + 16);
                  if ( v104 != 9 )
                  {
                    if ( v104 != 13 )
                      goto LABEL_60;
                    if ( *(_DWORD *)(v103 + 32) <= 0x40u )
                    {
                      v105 = *(_QWORD *)(v103 + 24) == 0;
                    }
                    else
                    {
                      v126 = *(_DWORD *)(v103 + 32);
                      v105 = v126 == (unsigned int)sub_16A57B0(v103 + 24);
                    }
                    if ( !v105 )
                      goto LABEL_60;
                  }
                  if ( v162 == ++v102 )
                  {
                    v23 = v146;
                    v29 = v134;
                    v40 = v130;
                    break;
                  }
                }
              }
            }
LABEL_42:
            v23 += 4;
            v28 = v40 + 4;
            v29 += 4;
            v30 += 4;
            if ( (__int64 *)v176 == v23 )
            {
              v27 = v135 - v23;
              goto LABEL_44;
            }
          }
        }
        v23 = (__int64 *)(a2 + 1);
LABEL_44:
        if ( v27 != 2 )
        {
          if ( v27 != 3 )
          {
            if ( v27 != 1 )
              goto LABEL_47;
LABEL_140:
            if ( sub_13CD190(*v23) )
              goto LABEL_47;
            goto LABEL_59;
          }
          if ( !sub_13CD190(*v23) )
          {
LABEL_59:
            v34 = v23;
LABEL_60:
            if ( v135 != v34 )
              goto LABEL_10;
LABEL_47:
            v41 = **a2;
            if ( *(_BYTE *)(v41 + 8) == 16 )
              v41 = **(_QWORD **)(v41 + 16);
            v42 = 8 * sub_15A95A0(*a4, *(_DWORD *)(v41 + 8) >> 8);
            v14 = a3;
            v43 = &a2[a3 - 1];
            if ( sub_127FA20(*a4, **v43) != v42 )
            {
LABEL_11:
              v15 = v14 * 8;
              v16 = &a2[v14];
              v17 = (v14 * 8) >> 5;
              v18 = v15 >> 3;
              if ( v17 > 0 )
              {
                v19 = a2;
                v20 = &a2[4 * v17];
                while ( *((_BYTE *)*v19 + 16) <= 0x10u )
                {
                  if ( *((_BYTE *)v19[1] + 16) > 0x10u )
                  {
                    ++v19;
                    break;
                  }
                  if ( *((_BYTE *)v19[2] + 16) > 0x10u )
                  {
                    v19 += 2;
                    break;
                  }
                  if ( *((_BYTE *)v19[3] + 16) > 0x10u )
                  {
                    v19 += 3;
                    break;
                  }
                  v19 += 4;
                  if ( v19 == v20 )
                  {
                    v18 = v16 - v19;
                    goto LABEL_97;
                  }
                }
LABEL_18:
                v5 = 0;
                if ( v16 == v19 )
                  goto LABEL_19;
                return (__int64)v5;
              }
              v19 = a2;
LABEL_97:
              if ( v18 != 2 )
              {
                if ( v18 != 3 )
                {
                  if ( v18 != 1 )
                    goto LABEL_19;
                  goto LABEL_100;
                }
                if ( *((_BYTE *)*v19 + 16) > 0x10u )
                  goto LABEL_18;
                ++v19;
              }
              if ( *((_BYTE *)*v19 + 16) > 0x10u )
                goto LABEL_18;
              ++v19;
LABEL_100:
              if ( *((_BYTE *)*v19 + 16) > 0x10u )
                goto LABEL_18;
LABEL_19:
              BYTE4(v183) = 0;
              v5 = (__int64 **)sub_15A2E80(v171, (unsigned int)*a2, (_DWORD)v138, v137, 0, (unsigned int)&v183, 0);
              v21 = sub_14DBA30(v5, *a4, 0);
              if ( v21 )
                return v21;
              return (__int64)v5;
            }
            v180 = v42;
            if ( v42 > 0x40 )
              sub_16A4EF0(&v179, 0, 0);
            else
              v179 = 0;
            v44 = sub_164A410(*a2, *a4, &v179);
            v45 = *v43;
            v173 = v44;
            v184 = (_QWORD *)v44;
            v46 = *((_BYTE *)v45 + 16);
            v47 = v45;
            if ( v46 == 37 )
            {
              v165 = v45;
              if ( sub_13CD190(*(v45 - 6)) && (unsigned __int8)sub_13D76A0(&v184, *(v165 - 3)) )
                goto LABEL_202;
LABEL_193:
              v45 = *v43;
              v109 = *((_BYTE *)*v43 + 16);
LABEL_194:
              v183 = (__int64 *)v173;
              if ( v109 == 52 )
              {
LABEL_195:
                v174 = v45;
                if ( !(unsigned __int8)sub_13D76A0(&v183, *(v45 - 6)) || !(unsigned __int8)sub_13CC520(*(v174 - 3)) )
                  goto LABEL_57;
LABEL_197:
                sub_13A38D0((__int64)&v181, (__int64)&v179);
                sub_16A7800(&v181, 1);
                v110 = v182;
                v182 = 0;
                LODWORD(v184) = v110;
                v183 = v181;
                v111 = sub_159C0E0(*v169, &v183);
                sub_135E100((__int64 *)&v183);
                sub_135E100((__int64 *)&v181);
                v5 = (__int64 **)sub_15A3BA0(v111, v169, 0);
LABEL_198:
                sub_135E100(&v179);
                return (__int64)v5;
              }
              if ( v109 == 5 )
                goto LABEL_56;
LABEL_57:
              sub_135E100(&v179);
              goto LABEL_11;
            }
            if ( v46 != 5 )
            {
              v183 = (__int64 *)v173;
              if ( v46 == 52 )
                goto LABEL_195;
              goto LABEL_57;
            }
            if ( *((_WORD *)v45 + 9) != 13 )
            {
              v183 = (__int64 *)v173;
LABEL_56:
              if ( *((_WORD *)v45 + 9) != 28 )
                goto LABEL_57;
              v175 = v45;
              v116 = *((_DWORD *)v45 + 5) & 0xFFFFFFF;
              if ( !(unsigned __int8)sub_13D7710(&v183, v45[-3 * v116]) || !sub_13CC690(v175[3 * (1 - v116)]) )
                goto LABEL_57;
              goto LABEL_197;
            }
            v112 = v45[-3 * (*((_DWORD *)v45 + 5) & 0xFFFFFFF)];
            if ( !v112 )
            {
LABEL_216:
              v109 = *((_BYTE *)v47 + 16);
              v45 = v47;
              goto LABEL_194;
            }
            if ( (unsigned __int8)sub_1593BB0(v45[-3 * (*((_DWORD *)v45 + 5) & 0xFFFFFFF)]) )
              goto LABEL_201;
            if ( *(_BYTE *)(v112 + 16) == 13 )
            {
              if ( *(_DWORD *)(v112 + 32) <= 0x40u )
              {
                v114 = *(_QWORD *)(v112 + 24) == 0;
              }
              else
              {
                v166 = *(_DWORD *)(v112 + 32);
                v114 = v166 == (unsigned int)sub_16A57B0(v112 + 24);
              }
            }
            else
            {
              if ( *(_BYTE *)(*(_QWORD *)v112 + 8LL) != 16 )
                goto LABEL_215;
              v118 = sub_15A1020(v112);
              if ( !v118 || *(_BYTE *)(v118 + 16) != 13 )
              {
                v147 = v47;
                v120 = 0;
                v168 = *(_DWORD *)(*(_QWORD *)v112 + 32LL);
                while ( v168 != v120 )
                {
                  v121 = sub_15A0A60(v112, v120);
                  if ( !v121 )
                    goto LABEL_215;
                  v122 = *(_BYTE *)(v121 + 16);
                  if ( v122 != 9 )
                  {
                    if ( v122 != 13 )
                      goto LABEL_215;
                    if ( *(_DWORD *)(v121 + 32) <= 0x40u )
                    {
                      v123 = *(_QWORD *)(v121 + 24) == 0;
                    }
                    else
                    {
                      v136 = *(_DWORD *)(v121 + 32);
                      v123 = v136 == (unsigned int)sub_16A57B0(v121 + 24);
                    }
                    if ( !v123 )
                      goto LABEL_215;
                  }
                  ++v120;
                }
                v47 = v147;
                goto LABEL_201;
              }
              v119 = *(_DWORD *)(v118 + 32);
              if ( v119 <= 0x40 )
                v114 = *(_QWORD *)(v118 + 24) == 0;
              else
                v114 = v119 == (unsigned int)sub_16A57B0(v118 + 24);
            }
            if ( !v114 )
            {
LABEL_215:
              v47 = *v43;
              goto LABEL_216;
            }
LABEL_201:
            if ( (unsigned __int8)sub_13D7710(&v184, v47[3 * (1LL - (*((_DWORD *)v47 + 5) & 0xFFFFFFF))]) )
            {
LABEL_202:
              v113 = sub_159C0E0(*v169, &v179);
              v5 = (__int64 **)sub_15A3BA0(v113, v169, 0);
              goto LABEL_198;
            }
            goto LABEL_193;
          }
          ++v23;
        }
        if ( sub_13CD190(*v23) )
        {
          ++v23;
          goto LABEL_140;
        }
        goto LABEL_59;
      }
      v181 = v169;
    }
    v75 = (__int64 **)*a2;
    v76 = (__int64)a2[1];
    v183 = &v178;
    v184 = v75;
    v185 = &v179;
    v77 = *(_BYTE *)(v76 + 16);
    if ( v77 == 49 )
    {
      v163 = v74;
      v106 = sub_13D62D0((__int64)&v183, *(_QWORD *)(v76 - 48));
      v74 = v163;
      if ( !v106 )
        goto LABEL_187;
      v107 = *(_QWORD *)(v76 - 24);
      if ( *(_BYTE *)(v107 + 16) != 13 )
        goto LABEL_187;
    }
    else
    {
      if ( v77 != 5 || *(_WORD *)(v76 + 18) != 25 )
      {
LABEL_115:
        v184 = v75;
        v185 = (__int64 *)v74;
        v183 = &v178;
        if ( sub_13D63A0((__int64)&v183, v76) )
        {
          v5 = (__int64 **)sub_13CD570(&v181, v178);
          if ( v5 )
            return (__int64)v5;
        }
        goto LABEL_9;
      }
      v167 = v74;
      v115 = sub_13E1280((__int64)&v183, *(_QWORD *)(v76 - 24LL * (*(_DWORD *)(v76 + 20) & 0xFFFFFFF)));
      v74 = v167;
      if ( !v115
        || (v107 = *(_QWORD *)(v76 + 24 * (1LL - (*(_DWORD *)(v76 + 20) & 0xFFFFFFF))), *(_BYTE *)(v107 + 16) != 13) )
      {
LABEL_187:
        v75 = (__int64 **)*a2;
        v76 = (__int64)a2[1];
        goto LABEL_115;
      }
    }
    v164 = v74;
    v108 = sub_13D04D0(v107 + 24, 0xFFFFFFFFFFFFFFFFLL);
    v74 = v164;
    if ( !v108 )
    {
      v117 = *(_QWORD **)(v107 + 24);
      if ( *(_DWORD *)(v107 + 32) > 0x40u )
        v117 = (_QWORD *)*v117;
      *v185 = (__int64)v117;
      if ( 1LL << v179 == v164 )
      {
        v5 = (__int64 **)sub_13CD570(&v181, v178);
        if ( v5 )
          return (__int64)v5;
        v75 = (__int64 **)*a2;
        v76 = (__int64)a2[1];
        v74 = v164;
        goto LABEL_115;
      }
    }
    goto LABEL_187;
  }
  return sub_1599EF0(v169);
}
