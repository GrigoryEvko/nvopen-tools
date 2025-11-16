// Function: sub_328C8C0
// Address: 0x328c8c0
//
__int64 __fastcall sub_328C8C0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v6; // rdx
  unsigned int *v7; // r9
  __int64 v8; // r10
  __int64 v9; // rcx
  __int64 (__fastcall *v10)(__int64, unsigned int); // r8
  int v11; // r14d
  char (__fastcall *v12)(__int64, unsigned int); // rax
  __int64 v13; // r15
  int v14; // eax
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  __int16 v18; // si
  __int64 v19; // rdi
  __int64 v20; // r11
  __int64 v21; // rcx
  __int64 v22; // r11
  __int64 v23; // rax
  unsigned __int16 *v24; // rcx
  __int64 v25; // rax
  __int128 v26; // rax
  __int128 v27; // rax
  int v28; // r8d
  int v29; // r9d
  char v30; // al
  __int64 v31; // rsi
  __int64 v32; // r14
  int v33; // r10d
  __int64 v34; // r11
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 (__fastcall *v37)(__int64, unsigned int); // rdx
  char (__fastcall *v38)(__int64, unsigned int); // rax
  __int64 v39; // rax
  unsigned __int16 v40; // dx
  __int64 v41; // rax
  __int16 *v42; // rax
  __int16 v43; // cx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rdx
  char v48; // al
  unsigned int v49; // eax
  unsigned int v50; // r10d
  __int64 v51; // r11
  __int64 v52; // rax
  __int64 v53; // rdx
  unsigned int v54; // eax
  int v55; // edx
  unsigned int v56; // eax
  int v57; // eax
  int v58; // edx
  unsigned int v59; // eax
  __int64 v60; // rdx
  unsigned int v61; // eax
  __int64 v62; // rdx
  unsigned int v63; // r13d
  char v64; // al
  unsigned int v65; // r10d
  __int64 v66; // r11
  char v67; // al
  __int64 v68; // r8
  __int64 v69; // rax
  _QWORD *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // r8
  unsigned int v76; // r10d
  __int64 v77; // rdx
  __int64 v78; // r14
  __int128 v79; // rax
  __int64 v80; // rax
  int v81; // r9d
  unsigned int v82; // r10d
  __int64 v83; // r13
  __int64 v84; // rdx
  __int64 v85; // r14
  __int128 v86; // rax
  int v87; // r9d
  unsigned int v88; // r10d
  __int64 v89; // rax
  __int64 v90; // rdx
  char v91; // al
  char v92; // al
  char v93; // al
  bool v94; // al
  _QWORD *v95; // rax
  __int64 v96; // rax
  void *v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rax
  __int128 v100; // rax
  __int128 v101; // kr00_16
  int v102; // r9d
  __int128 v103; // rax
  int v104; // r9d
  __int128 v105; // rax
  __int64 v106; // rdx
  int v107; // r9d
  __int64 v108; // rax
  __int64 v109; // rdx
  __int128 v110; // [rsp-20h] [rbp-120h]
  __int128 v111; // [rsp-10h] [rbp-110h]
  __int64 v112; // [rsp+0h] [rbp-100h]
  unsigned int v113; // [rsp+0h] [rbp-100h]
  __int64 v114; // [rsp+0h] [rbp-100h]
  __int64 v115; // [rsp+8h] [rbp-F8h]
  __int64 v116; // [rsp+8h] [rbp-F8h]
  unsigned int *v117; // [rsp+10h] [rbp-F0h]
  int v118; // [rsp+10h] [rbp-F0h]
  __int64 v119; // [rsp+10h] [rbp-F0h]
  __int64 v120; // [rsp+10h] [rbp-F0h]
  __int64 v121; // [rsp+18h] [rbp-E8h]
  __int64 v122; // [rsp+18h] [rbp-E8h]
  __int64 v123; // [rsp+18h] [rbp-E8h]
  __int64 v124; // [rsp+18h] [rbp-E8h]
  __int128 v125; // [rsp+20h] [rbp-E0h]
  unsigned int v126; // [rsp+20h] [rbp-E0h]
  __int64 v127; // [rsp+20h] [rbp-E0h]
  __int64 v128; // [rsp+20h] [rbp-E0h]
  unsigned int v129; // [rsp+20h] [rbp-E0h]
  unsigned int v130; // [rsp+20h] [rbp-E0h]
  __int64 v131; // [rsp+20h] [rbp-E0h]
  unsigned int v132; // [rsp+20h] [rbp-E0h]
  __int64 v133; // [rsp+20h] [rbp-E0h]
  __int64 v134; // [rsp+20h] [rbp-E0h]
  unsigned int v135; // [rsp+20h] [rbp-E0h]
  int v136; // [rsp+30h] [rbp-D0h]
  __int128 v137; // [rsp+30h] [rbp-D0h]
  __int64 v138; // [rsp+30h] [rbp-D0h]
  __int64 v139; // [rsp+30h] [rbp-D0h]
  int v140; // [rsp+30h] [rbp-D0h]
  unsigned int v141; // [rsp+30h] [rbp-D0h]
  unsigned int v142; // [rsp+30h] [rbp-D0h]
  unsigned int v143; // [rsp+30h] [rbp-D0h]
  __int64 v144; // [rsp+30h] [rbp-D0h]
  __int64 v145; // [rsp+30h] [rbp-D0h]
  __int64 v146; // [rsp+30h] [rbp-D0h]
  unsigned int v147; // [rsp+40h] [rbp-C0h]
  __int64 v148; // [rsp+40h] [rbp-C0h]
  __int64 v149; // [rsp+40h] [rbp-C0h]
  int v150; // [rsp+40h] [rbp-C0h]
  unsigned int v151; // [rsp+40h] [rbp-C0h]
  unsigned int v152; // [rsp+40h] [rbp-C0h]
  __int64 v153; // [rsp+40h] [rbp-C0h]
  __int128 v154; // [rsp+40h] [rbp-C0h]
  unsigned int v155; // [rsp+40h] [rbp-C0h]
  __int64 v156; // [rsp+40h] [rbp-C0h]
  int v157; // [rsp+40h] [rbp-C0h]
  int v158; // [rsp+40h] [rbp-C0h]
  int v159; // [rsp+40h] [rbp-C0h]
  __int64 v160; // [rsp+40h] [rbp-C0h]
  __int64 v161; // [rsp+40h] [rbp-C0h]
  __int64 v162; // [rsp+40h] [rbp-C0h]
  __int128 v163; // [rsp+40h] [rbp-C0h]
  __int64 v164; // [rsp+40h] [rbp-C0h]
  unsigned int v165; // [rsp+5Ch] [rbp-A4h] BYREF
  __int64 v166; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v167; // [rsp+68h] [rbp-98h]
  unsigned int v168; // [rsp+70h] [rbp-90h] BYREF
  __int64 v169; // [rsp+78h] [rbp-88h]
  unsigned int v170; // [rsp+80h] [rbp-80h] BYREF
  __int64 v171; // [rsp+88h] [rbp-78h]
  unsigned int v172; // [rsp+90h] [rbp-70h] BYREF
  __int64 v173; // [rsp+98h] [rbp-68h]
  __int64 v174; // [rsp+A0h] [rbp-60h] BYREF
  int v175; // [rsp+A8h] [rbp-58h]
  __int64 v176; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v177; // [rsp+B8h] [rbp-48h]
  __int64 v178; // [rsp+C0h] [rbp-40h] BYREF
  __int64 v179; // [rsp+C8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 16);
  v7 = *(unsigned int **)(a1 + 40);
  v8 = *(_QWORD *)v7;
  v9 = v7[2];
  v10 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v6 + 1368LL);
  v11 = *(_DWORD *)(*(_QWORD *)v7 + 24LL);
  if ( v10 == sub_2FE4300 )
  {
    v12 = *(char (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v6 + 1360LL);
    if ( v12 == sub_2FE3400 )
    {
      if ( v11 <= 98 )
      {
        if ( v11 > 55 )
        {
          switch ( v11 )
          {
            case '8':
            case ':':
            case '?':
            case '@':
            case 'D':
            case 'F':
            case 'L':
            case 'M':
            case 'R':
            case 'S':
            case '`':
            case 'b':
              goto LABEL_18;
            default:
              break;
          }
        }
LABEL_5:
        if ( v11 > 56 )
        {
LABEL_6:
          switch ( v11 )
          {
            case '9':
            case ';':
            case '<':
            case '=':
            case '>':
            case 'T':
            case 'U':
            case 'a':
            case 'c':
            case 'd':
              goto LABEL_18;
            default:
              goto LABEL_13;
          }
        }
        goto LABEL_13;
      }
      if ( v11 > 188 )
      {
        if ( (unsigned int)(v11 - 279) <= 7 )
          goto LABEL_18;
      }
      else
      {
        if ( v11 > 185 || (unsigned int)(v11 - 172) <= 0xB )
          goto LABEL_18;
        if ( v11 <= 100 )
          goto LABEL_6;
      }
    }
    else
    {
      v132 = v7[2];
      v145 = *(_QWORD *)v7;
      v156 = *(_QWORD *)(a2 + 16);
      v91 = v12(v6, v11);
      v7 = *(unsigned int **)(a1 + 40);
      v6 = v156;
      v8 = v145;
      v9 = v132;
      if ( v91 )
        goto LABEL_18;
      if ( v11 <= 100 )
        goto LABEL_5;
    }
    if ( (unsigned int)(v11 - 190) > 4 )
    {
LABEL_13:
      v13 = *((_QWORD *)v7 + 5);
      goto LABEL_14;
    }
  }
  else
  {
    v126 = v7[2];
    v138 = *(_QWORD *)v7;
    v149 = *(_QWORD *)(a2 + 16);
    v30 = v10(v6, v11);
    v7 = *(unsigned int **)(a1 + 40);
    v6 = v149;
    v8 = v138;
    v9 = v126;
    if ( !v30 )
      goto LABEL_13;
  }
LABEL_18:
  if ( *(_DWORD *)(v8 + 68) != 1 )
    goto LABEL_13;
  v16 = *(_QWORD *)(v8 + 40);
  v17 = *(_QWORD *)(v8 + 48) + 16 * v9;
  v13 = *((_QWORD *)v7 + 5);
  v18 = *(_WORD *)v17;
  v19 = *(_QWORD *)v16;
  v20 = *(_QWORD *)(*(_QWORD *)v16 + 48LL) + 16LL * *(unsigned int *)(v16 + 8);
  if ( *(_WORD *)v17 != *(_WORD *)v20 )
    goto LABEL_14;
  v21 = *(_QWORD *)(v17 + 8);
  if ( *(_QWORD *)(v20 + 8) != v21 && !v18 )
    goto LABEL_14;
  v22 = *(_QWORD *)(v16 + 40);
  v23 = *(_QWORD *)(v22 + 48) + 16LL * *(unsigned int *)(v16 + 48);
  if ( v18 != *(_WORD *)v23 || *(_QWORD *)(v23 + 8) != v21 && !v18 )
    goto LABEL_14;
  v24 = *(unsigned __int16 **)(a1 + 48);
  v136 = v7[12];
  v25 = *v24;
  v147 = (unsigned __int16)v25;
  if ( a3 )
  {
    if ( ((_WORD)v25 == 1 || (_WORD)v25 && *(_QWORD *)(v6 + 8LL * (unsigned __int16)v25 + 112))
      && (unsigned int)v11 <= 0x1F3
      && !*(_BYTE *)((unsigned int)v11 + 500 * v25 + v6 + 6414) )
    {
      goto LABEL_31;
    }
LABEL_14:
    v14 = *(_DWORD *)(v13 + 24);
    if ( v14 != 11 && v14 != 35 )
      return 0;
    v31 = *((_QWORD *)v7 + 1);
    v32 = *(_QWORD *)(a2 + 16);
    v166 = sub_33CF5B0(*(_QWORD *)v7, v31);
    v33 = *(_DWORD *)(v166 + 24);
    v34 = v166;
    v35 = *(_QWORD *)v32;
    v167 = v36;
    v37 = *(__int64 (__fastcall **)(__int64, unsigned int))(v35 + 1368);
    if ( v37 == sub_2FE4300 )
    {
      v38 = *(char (__fastcall **)(__int64, unsigned int))(v35 + 1360);
      if ( v38 == sub_2FE3400 )
      {
        if ( v33 <= 98 )
        {
          if ( v33 > 55 )
          {
            switch ( v33 )
            {
              case '8':
              case ':':
              case '?':
              case '@':
              case 'D':
              case 'F':
              case 'L':
              case 'M':
              case 'R':
              case 'S':
              case '`':
              case 'b':
                goto LABEL_49;
              default:
                break;
            }
          }
LABEL_45:
          if ( v33 > 56 )
          {
LABEL_46:
            switch ( v33 )
            {
              case '9':
              case ';':
              case '<':
              case '=':
              case '>':
              case 'T':
              case 'U':
              case 'a':
              case 'c':
              case 'd':
                goto LABEL_49;
              default:
                return 0;
            }
          }
          return 0;
        }
        if ( v33 > 188 )
        {
          if ( (unsigned int)(v33 - 279) <= 7 )
            goto LABEL_49;
        }
        else
        {
          if ( v33 > 185 || (unsigned int)(v33 - 172) <= 0xB )
            goto LABEL_49;
          if ( v33 <= 100 )
            goto LABEL_46;
        }
      }
      else
      {
        v31 = (unsigned int)v33;
        v158 = v33;
        v93 = v38(v32, v33);
        v33 = v158;
        v34 = v166;
        if ( v93 )
          goto LABEL_49;
        if ( v158 <= 100 )
          goto LABEL_45;
      }
      if ( (unsigned int)(v33 - 190) > 4 )
        return 0;
    }
    else
    {
      v31 = (unsigned int)v33;
      v157 = v33;
      v92 = v37(v32, v33);
      v33 = v157;
      v34 = v166;
      if ( !v92 )
        return 0;
    }
LABEL_49:
    if ( *(_DWORD *)(v34 + 68) != 1 )
      return 0;
    if ( v33 == 97 )
    {
      v95 = *(_QWORD **)(v34 + 40);
      v160 = v34;
      v31 = v95[1];
      v96 = sub_33E1790(*v95, v31, 1);
      v34 = v160;
      v33 = 97;
      if ( v96 )
      {
        v134 = v160;
        v161 = *(_QWORD *)(v96 + 96);
        v97 = sub_C33340();
        v33 = 97;
        v34 = v134;
        if ( *(void **)(v161 + 24) == v97 )
        {
          v98 = *(_QWORD *)(v161 + 32);
          if ( (*(_BYTE *)(v98 + 20) & 7) != 3 )
            goto LABEL_51;
        }
        else
        {
          v98 = v161 + 24;
          if ( (*(_BYTE *)(v161 + 44) & 7) != 3 )
            goto LABEL_51;
        }
        if ( (*(_BYTE *)(v98 + 20) & 8) != 0 )
          return 0;
      }
    }
LABEL_51:
    v39 = *(_QWORD *)(v34 + 48) + 16LL * (unsigned int)v167;
    v40 = *(_WORD *)v39;
    v41 = *(_QWORD *)(v39 + 8);
    LOWORD(v168) = v40;
    v169 = v41;
    if ( v40 )
    {
      if ( (unsigned __int16)(v40 - 17) > 0x9Eu )
        return 0;
    }
    else
    {
      v133 = v34;
      v159 = v33;
      v94 = sub_30070D0((__int64)&v168);
      v33 = v159;
      v40 = 0;
      v34 = v133;
      if ( !v94 )
        return 0;
    }
    v42 = *(__int16 **)(a1 + 48);
    v43 = *v42;
    v171 = *((_QWORD *)v42 + 1);
    v44 = *(_QWORD *)(v13 + 96);
    LOWORD(v170) = v43;
    if ( *(_DWORD *)(v44 + 32) <= 0x40u )
      v122 = *(_QWORD *)(v44 + 24);
    else
      v122 = **(_QWORD **)(v44 + 24);
    if ( v40 )
    {
      if ( v40 == 1 || (unsigned __int16)(v40 - 504) <= 7u )
        goto LABEL_122;
      v99 = 16LL * (v40 - 1);
      v47 = *(_QWORD *)&byte_444C4A0[v99];
      v48 = byte_444C4A0[v99 + 8];
    }
    else
    {
      v139 = v34;
      v150 = v33;
      v45 = sub_3007260((__int64)&v168);
      v34 = v139;
      v33 = v150;
      v178 = v45;
      v179 = v46;
      v47 = v45;
      v48 = v179;
    }
    v127 = v34;
    v140 = v33;
    v176 = v47;
    LOBYTE(v177) = v48;
    v49 = sub_CA1930(&v176);
    v50 = v140;
    v51 = v127;
    v151 = v49;
    if ( !(_WORD)v170 )
    {
      v52 = sub_3007260((__int64)&v170);
      v51 = v127;
      v50 = v140;
      v176 = v52;
      v177 = v53;
LABEL_59:
      v128 = v51;
      v141 = v50;
      LOBYTE(v175) = v53;
      v174 = v52;
      v54 = sub_CA1930(&v174);
      v55 = v151 % v54;
      v152 = v151 / v54;
      if ( v55 )
        return 0;
      v119 = v128;
      v56 = sub_3281500(&v168, v31);
      v58 = v56 % v152;
      v57 = v56 / v152;
      if ( v58 )
        return 0;
      v129 = v57;
      LOWORD(v59) = sub_3281100((unsigned __int16 *)&v168, v31);
      v61 = sub_327FCF0(*(__int64 **)(a2 + 64), v59, v60, v129, 0);
      v173 = v62;
      v172 = v61;
      if ( !(unsigned __int8)sub_328C7F0(v32, v141, v61, v62, a3) )
        return 0;
      v130 = v141;
      v63 = (unsigned int)v122 / (unsigned int)sub_3281500(&v170, v141);
      v142 = v63 * sub_3281500(&v172, v141);
      v64 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v32 + 1680LL))(
              v32,
              v172,
              v173,
              v168,
              v169,
              v142);
      v65 = v130;
      v66 = v119;
      if ( v64 )
      {
        v67 = sub_3286E00(&v166);
        v65 = v130;
        v66 = v119;
        if ( v67 )
        {
          v69 = *(_QWORD *)(**(_QWORD **)(a1 + 40) + 56LL);
          if ( v69 )
          {
            if ( !*(_QWORD *)(v69 + 32) )
            {
              v174 = *(_QWORD *)(a1 + 80);
              if ( v174 )
              {
                sub_325F5D0(&v174);
                v66 = v119;
                v65 = v130;
              }
              v135 = v65;
              v162 = v66;
              v175 = *(_DWORD *)(a1 + 72);
              *(_QWORD *)&v100 = sub_3400EE0(a2, v142, &v174, 0, v68);
              v101 = v100;
              v146 = v162;
              *(_QWORD *)&v103 = sub_3406EB0(
                                   a2,
                                   161,
                                   (unsigned int)&v174,
                                   v172,
                                   v173,
                                   v102,
                                   *(_OWORD *)*(_QWORD *)(v162 + 40),
                                   v100);
              v163 = v103;
              *(_QWORD *)&v105 = sub_3406EB0(
                                   a2,
                                   161,
                                   (unsigned int)&v174,
                                   v172,
                                   v173,
                                   v104,
                                   *(_OWORD *)(*(_QWORD *)(v146 + 40) + 40LL),
                                   v101);
              v89 = sub_3405C90(a2, v135, (unsigned int)&v174, v172, v173, *(_DWORD *)(v146 + 28), v163, v105);
              goto LABEL_73;
            }
          }
        }
      }
      if ( v152 != 2 || v65 - 186 > 2 )
        return 0;
      v70 = *(_QWORD **)(v66 + 40);
      v165 = v63;
      v113 = v65;
      v153 = v66;
      v71 = sub_33CF5B0(*v70, v70[1]);
      v123 = sub_325F200(&v165, v71);
      v120 = v72;
      v73 = sub_33CF5B0(*(_QWORD *)(*(_QWORD *)(v153 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v153 + 40) + 48LL));
      v74 = sub_325F200(&v165, v73);
      v76 = v113;
      v131 = v74;
      v116 = v77;
      if ( v123 )
      {
        v78 = v142;
        v174 = *(_QWORD *)(a1 + 80);
        v114 = v153;
        v143 = v76;
        if ( v174 )
          sub_325F5D0(&v174);
        v175 = *(_DWORD *)(a1 + 72);
        *(_QWORD *)&v79 = sub_3400EE0(a2, v78, &v174, 0, v75);
        v154 = v79;
        v80 = sub_33FB890(a2, v172, v173, v123, v120);
        v82 = v143;
        v83 = v80;
        v85 = v84;
        if ( !v131 )
        {
          *(_QWORD *)&v86 = sub_3406EB0(
                              a2,
                              161,
                              (unsigned int)&v174,
                              v172,
                              v173,
                              v81,
                              *(_OWORD *)(*(_QWORD *)(v114 + 40) + 40LL),
                              v154);
          v88 = v143;
          goto LABEL_72;
        }
      }
      else
      {
        if ( !v74 )
          return 0;
        v174 = *(_QWORD *)(a1 + 80);
        v124 = v153;
        if ( v174 )
          sub_325F5D0(&v174);
        v175 = *(_DWORD *)(a1 + 72);
        v164 = sub_3400EE0(a2, v142, &v174, 0, v75);
        *((_QWORD *)&v111 + 1) = v106;
        *(_QWORD *)&v111 = v164;
        v108 = sub_3406EB0(a2, 161, (unsigned int)&v174, v172, v173, v107, *(_OWORD *)*(_QWORD *)(v124 + 40), v111);
        v82 = v113;
        v83 = v108;
        v85 = v109;
      }
      v155 = v82;
      *(_QWORD *)&v86 = sub_33FB890(a2, v172, v173, v131, v116);
      v88 = v155;
LABEL_72:
      *((_QWORD *)&v110 + 1) = v85;
      *(_QWORD *)&v110 = v83;
      v89 = sub_3406EB0(a2, v88, (unsigned int)&v174, v172, v173, v87, v110, v86);
LABEL_73:
      v144 = sub_33FB890(a2, v170, v171, v89, v90);
      sub_9C6650(&v174);
      return v144;
    }
    if ( (_WORD)v170 != 1 && (unsigned __int16)(v170 - 504) > 7u )
    {
      v53 = 16LL * ((unsigned __int16)v170 - 1);
      v52 = *(_QWORD *)&byte_444C4A0[v53];
      LOBYTE(v53) = byte_444C4A0[v53 + 8];
      goto LABEL_59;
    }
LABEL_122:
    BUG();
  }
  if ( (_WORD)v25 != 1 && (!(_WORD)v25 || !*(_QWORD *)(v6 + 8LL * (unsigned __int16)v25 + 112))
    || (unsigned int)v11 <= 0x1F3 && (*(_BYTE *)((unsigned int)v11 + 500 * v25 + v6 + 6414) & 0xFB) != 0 )
  {
    goto LABEL_14;
  }
LABEL_31:
  v117 = v7;
  v115 = v22;
  v112 = v8;
  v121 = *((_QWORD *)v24 + 1);
  *(_QWORD *)&v26 = sub_326B110(v19, v13, v136, (unsigned __int16)v25, v121);
  v125 = v26;
  *(_QWORD *)&v27 = sub_326B110(v115, v13, v136, v147, v121);
  v7 = v117;
  v137 = v27;
  if ( !(_QWORD)v125 || !(_QWORD)v27 )
    goto LABEL_14;
  v28 = v121;
  v29 = *(_DWORD *)(v112 + 28);
  v178 = *(_QWORD *)(a1 + 80);
  if ( v178 )
  {
    v118 = v29;
    sub_B96E90((__int64)&v178, v178, 1);
    v29 = v118;
    v28 = v121;
  }
  LODWORD(v179) = *(_DWORD *)(a1 + 72);
  result = sub_3405C90(a2, v11, (unsigned int)&v178, v147, v28, v29, v125, v137);
  if ( v178 )
  {
    v148 = result;
    sub_B91220((__int64)&v178, v178);
    result = v148;
  }
  if ( !result )
  {
    v7 = *(unsigned int **)(a1 + 40);
    goto LABEL_13;
  }
  return result;
}
