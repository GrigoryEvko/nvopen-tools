// Function: sub_12992B0
// Address: 0x12992b0
//
__int64 __fastcall sub_12992B0(_QWORD *a1, __int64 a2, __int64 **a3, _DWORD *a4)
{
  __int64 v7; // rax
  _QWORD *v8; // r12
  _BYTE *v9; // rsi
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  bool v15; // cc
  unsigned __int8 v16; // al
  _QWORD *v17; // rcx
  _BYTE *v18; // r12
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  signed int v22; // r15d
  _QWORD *v23; // r12
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rbx
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r11
  unsigned __int64 v31; // rbx
  unsigned int v32; // ecx
  __int64 v33; // rax
  _QWORD *v34; // r13
  __int64 v35; // rdi
  unsigned __int64 *v36; // r12
  __int64 v37; // rax
  unsigned __int64 v38; // rcx
  __int64 v39; // rsi
  __int64 v40; // rsi
  __int64 v41; // r12
  unsigned int v42; // ebx
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rcx
  unsigned __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // r8
  __int64 v50; // rsi
  __int64 v51; // r12
  int v53; // eax
  int v54; // eax
  __int64 v55; // r13
  unsigned int v56; // eax
  __int64 v57; // rcx
  unsigned __int64 v58; // r9
  _QWORD *v59; // rax
  __int64 v60; // rax
  unsigned int v61; // ecx
  unsigned int v62; // edx
  _BYTE *v63; // rsi
  bool v64; // zf
  _BYTE *v65; // rax
  unsigned int v66; // r13d
  _QWORD *v67; // rbx
  __int64 v68; // rdi
  unsigned __int64 *v69; // r12
  __int64 v70; // rax
  unsigned __int64 v71; // rcx
  __int64 v72; // rsi
  __int64 v73; // rsi
  __int64 v74; // rdi
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // r11
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rdi
  unsigned __int64 *v81; // r13
  __int64 v82; // rax
  unsigned __int64 v83; // rcx
  __int64 v84; // rsi
  unsigned __int64 v85; // rdx
  __int64 v86; // rsi
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // r11
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // r11
  __int64 v95; // r10
  __int64 v96; // rax
  __int64 v97; // rdi
  __int64 v98; // r11
  unsigned __int64 *v99; // r13
  __int64 v100; // rax
  unsigned __int64 v101; // rcx
  __int64 v102; // rsi
  unsigned __int64 v103; // rdx
  __int64 v104; // rsi
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // r12
  __int64 v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rax
  __int64 v111; // rax
  unsigned __int64 v112; // rax
  __int64 v113; // rsi
  int v114; // eax
  unsigned int v115; // eax
  __int64 v116; // r11
  __int64 v117; // rsi
  unsigned __int64 v118; // r13
  __int64 v119; // rax
  _BYTE *v120; // rsi
  _QWORD *v121; // rax
  _BYTE *v122; // rsi
  __int64 v123; // rax
  __int64 v124; // rax
  __int64 v125; // rsi
  int v126; // eax
  __int64 v127; // rax
  _QWORD *v128; // rax
  __int64 v129; // [rsp+8h] [rbp-1D8h]
  __int64 v130; // [rsp+10h] [rbp-1D0h]
  __int64 v131; // [rsp+20h] [rbp-1C0h]
  __int64 v132; // [rsp+20h] [rbp-1C0h]
  __int64 v133; // [rsp+28h] [rbp-1B8h]
  __int64 v134; // [rsp+28h] [rbp-1B8h]
  __int64 v135; // [rsp+28h] [rbp-1B8h]
  __int64 v136; // [rsp+28h] [rbp-1B8h]
  unsigned int v137; // [rsp+30h] [rbp-1B0h]
  __int64 v138; // [rsp+30h] [rbp-1B0h]
  __int64 v139; // [rsp+30h] [rbp-1B0h]
  unsigned __int64 v140; // [rsp+30h] [rbp-1B0h]
  __int64 v141; // [rsp+38h] [rbp-1A8h]
  __int64 v142; // [rsp+38h] [rbp-1A8h]
  __int64 v143; // [rsp+38h] [rbp-1A8h]
  __int64 v144; // [rsp+38h] [rbp-1A8h]
  unsigned __int64 v145; // [rsp+38h] [rbp-1A8h]
  __int64 v146; // [rsp+40h] [rbp-1A0h]
  __int64 v147; // [rsp+40h] [rbp-1A0h]
  __int64 v148; // [rsp+40h] [rbp-1A0h]
  unsigned __int64 v149; // [rsp+40h] [rbp-1A0h]
  __int64 v150; // [rsp+40h] [rbp-1A0h]
  __int64 v151; // [rsp+40h] [rbp-1A0h]
  __int64 v152; // [rsp+40h] [rbp-1A0h]
  __int64 v153; // [rsp+48h] [rbp-198h]
  __int64 v155; // [rsp+60h] [rbp-180h]
  _BYTE *v156; // [rsp+68h] [rbp-178h]
  __int64 v157; // [rsp+70h] [rbp-170h]
  _BYTE *v158; // [rsp+80h] [rbp-160h]
  unsigned int v159; // [rsp+90h] [rbp-150h]
  __int64 v160; // [rsp+90h] [rbp-150h]
  __int64 v161; // [rsp+90h] [rbp-150h]
  __int64 v162; // [rsp+90h] [rbp-150h]
  __int64 v163; // [rsp+90h] [rbp-150h]
  __int64 v164; // [rsp+90h] [rbp-150h]
  __int64 v165; // [rsp+90h] [rbp-150h]
  __int64 v166; // [rsp+90h] [rbp-150h]
  unsigned __int64 v167; // [rsp+98h] [rbp-148h]
  __int64 v168; // [rsp+98h] [rbp-148h]
  __int64 v169; // [rsp+98h] [rbp-148h]
  __int64 v170; // [rsp+98h] [rbp-148h]
  __int64 v171; // [rsp+98h] [rbp-148h]
  __int64 v172; // [rsp+A0h] [rbp-140h]
  __int64 v173; // [rsp+A8h] [rbp-138h]
  _QWORD **v174; // [rsp+A8h] [rbp-138h]
  __int64 v175; // [rsp+B0h] [rbp-130h] BYREF
  __int64 v176; // [rsp+B8h] [rbp-128h] BYREF
  __int64 v177; // [rsp+C0h] [rbp-120h] BYREF
  _BYTE *v178; // [rsp+C8h] [rbp-118h]
  _BYTE *v179; // [rsp+D0h] [rbp-110h]
  _QWORD v180[2]; // [rsp+E0h] [rbp-100h] BYREF
  char v181; // [rsp+F0h] [rbp-F0h]
  char v182; // [rsp+F1h] [rbp-EFh]
  _QWORD v183[2]; // [rsp+100h] [rbp-E0h] BYREF
  __int16 v184; // [rsp+110h] [rbp-D0h]
  _QWORD *v185; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v186; // [rsp+128h] [rbp-B8h]
  _QWORD v187[22]; // [rsp+130h] [rbp-B0h] BYREF

  v7 = sub_1643330(a1[5]);
  v177 = 0;
  v155 = v7;
  v178 = 0;
  v179 = 0;
  v185 = (_QWORD *)sub_1646BA0(v7, 0);
  v8 = v185;
  sub_1278040((__int64)&v177, 0, &v185);
  v185 = v8;
  v120 = v178;
  if ( v178 == v179 )
  {
    sub_1278040((__int64)&v177, v178, &v185);
    v9 = v178;
  }
  else
  {
    if ( v178 )
    {
      *(_QWORD *)v178 = v8;
      v120 = v178;
    }
    v9 = v120 + 8;
    v178 = v9;
  }
  v173 = v177;
  v10 = (__int64)&v9[-v177] >> 3;
  v11 = sub_1643350(a1[5]);
  v12 = sub_1644EA0(v11, v173, v10, 0);
  v153 = sub_1632190(*(_QWORD *)a1[4], "vprintf", 7, v12);
  v185 = v187;
  v186 = 0x1000000000LL;
  v13 = **a3;
  v14 = (__int64)(*a3 + 1);
  v15 = *((_DWORD *)a3 + 2) <= 1u;
  LODWORD(v186) = 1;
  v174 = (_QWORD **)v14;
  v187[0] = v13;
  if ( v15 )
  {
    v107 = sub_15A06D0(v8);
    v108 = (unsigned int)v186;
    if ( (unsigned int)v186 >= HIDWORD(v186) )
    {
      sub_16CD150(&v185, v187, 0, 8);
      v108 = (unsigned int)v186;
    }
    v185[v108] = v107;
    v109 = *((unsigned int *)a3 + 2);
    v49 = (unsigned int)(v186 + 1);
    v110 = *(_QWORD *)(a2 + 16);
    LODWORD(v186) = v186 + 1;
    v172 = v110 + 80;
    v157 = (__int64)&(*a3)[v109];
    if ( v174 != (_QWORD **)v157 )
    {
      v158 = 0;
      v156 = 0;
      goto LABEL_8;
    }
    goto LABEL_46;
  }
  v158 = (_BYTE *)a1[19];
  if ( v158 )
  {
    v16 = v158[16];
    if ( v16 <= 0x17u )
    {
      v17 = v187;
      v18 = 0;
      v19 = 1;
      v156 = 0;
      goto LABEL_7;
    }
    v61 = 16;
    v62 = 1;
  }
  else
  {
    v183[0] = "tmp";
    v184 = 259;
    v121 = sub_127FC40(a1, v155, (__int64)v183, 8u, 0);
    v62 = v186;
    v61 = HIDWORD(v186);
    a1[19] = v121;
    v122 = v121;
    v16 = *((_BYTE *)v121 + 16);
    if ( v16 <= 0x17u )
    {
      v156 = 0;
      v18 = 0;
      v158 = v122;
      goto LABEL_69;
    }
    v158 = v122;
  }
  v63 = v158;
  while ( 1 )
  {
    if ( v16 == 71 )
    {
      v63 = (_BYTE *)*((_QWORD *)v63 - 3);
      goto LABEL_94;
    }
    if ( v16 != 78 )
      break;
    v106 = *((_QWORD *)v63 - 3);
    if ( *(_BYTE *)(v106 + 16) || (*(_BYTE *)(v106 + 33) & 0x20) == 0 )
    {
LABEL_95:
      v156 = 0;
      v18 = 0;
      goto LABEL_69;
    }
    if ( *(_DWORD *)(v106 + 36) == 4237 )
      v63 = *(_BYTE **)&v63[-24 * (*((_DWORD *)v63 + 5) & 0xFFFFFFF)];
LABEL_94:
    v16 = v63[16];
    if ( v16 <= 0x17u )
      goto LABEL_95;
  }
  v64 = v16 == 53;
  v65 = 0;
  if ( v64 )
    v65 = v63;
  v156 = v65;
  v18 = v65;
LABEL_69:
  if ( v62 >= v61 )
  {
    sub_16CD150(&v185, v187, 0, 8);
    v17 = v185;
    v19 = (unsigned int)v186;
  }
  else
  {
    v17 = v185;
    v19 = v62;
  }
LABEL_7:
  v17[v19] = v18;
  v20 = *(_QWORD *)(a2 + 16);
  v21 = *((unsigned int *)a3 + 2);
  LODWORD(v186) = v186 + 1;
  v172 = v20 + 80;
  v157 = (__int64)&(*a3)[v21];
  if ( v174 == (_QWORD **)v157 )
  {
    v22 = 0;
  }
  else
  {
LABEL_8:
    v22 = 0;
    do
    {
      v23 = *v174;
      if ( *(_DWORD *)(v172 + 12) == 2 && *(_BYTE *)(v172 + 16) )
      {
        v184 = 257;
        v66 = unk_4D0463C;
        if ( unk_4D0463C )
          v66 = sub_126A420(a1[4], (unsigned __int64)v23);
        v67 = (_QWORD *)sub_1648A60(64, 1);
        if ( v67 )
          sub_15F9210(v67, *(_QWORD *)(*v23 + 24LL), v23, 0, v66, 0);
        v68 = a1[7];
        if ( v68 )
        {
          v69 = (unsigned __int64 *)a1[8];
          sub_157E9D0(v68 + 40, v67);
          v70 = v67[3];
          v71 = *v69;
          v67[4] = v69;
          v71 &= 0xFFFFFFFFFFFFFFF8LL;
          v67[3] = v71 | v70 & 7;
          *(_QWORD *)(v71 + 8) = v67 + 3;
          *v69 = *v69 & 7 | (unsigned __int64)(v67 + 3);
        }
        sub_164B780(v67, v183);
        v72 = a1[6];
        if ( v72 )
        {
          v180[0] = a1[6];
          sub_1623A60(v180, v72, 2);
          if ( v67[6] )
            sub_161E7C0(v67 + 6);
          v73 = v180[0];
          v67[6] = v180[0];
          if ( v73 )
            sub_1623210(v180, v73, v67 + 6);
        }
        v23 = v67;
      }
      v24 = *v23;
      v25 = *(unsigned __int8 *)(*v23 + 8LL);
      if ( (_BYTE)v25 == 15 && *(_DWORD *)(v24 + 8) >> 8 )
      {
        v87 = sub_1646BA0(*(_QWORD *)(v24 + 24), 0);
        v88 = sub_128B420((__int64)a1, v23, 0, v87, 0, 0, a4);
        v24 = *(_QWORD *)v88;
        v23 = (_QWORD *)v88;
        v25 = *(unsigned __int8 *)(*(_QWORD *)v88 + 8LL);
      }
      v26 = v24;
      v27 = 1;
      v28 = *(_QWORD *)(a1[4] + 368LL);
      while ( 2 )
      {
        switch ( v25 )
        {
          case 0LL:
          case 8LL:
          case 10LL:
          case 12LL:
          case 16LL:
            v60 = *(_QWORD *)(v26 + 32);
            v26 = *(_QWORD *)(v26 + 24);
            v27 *= v60;
            v25 = *(unsigned __int8 *)(v26 + 8);
            continue;
          case 1LL:
            v29 = 16;
            break;
          case 2LL:
            v29 = 32;
            break;
          case 3LL:
          case 9LL:
            v29 = 64;
            break;
          case 4LL:
            v29 = 80;
            break;
          case 5LL:
          case 6LL:
            v29 = 128;
            break;
          case 7LL:
            v168 = v24;
            v53 = sub_15A9520(v28, 0);
            v24 = v168;
            v29 = (unsigned int)(8 * v53);
            break;
          case 11LL:
            v29 = *(_DWORD *)(v26 + 8) >> 8;
            break;
          case 13LL:
            v171 = v24;
            v59 = (_QWORD *)sub_15A9930(v28, v26);
            v24 = v171;
            v29 = 8LL * *v59;
            break;
          case 14LL:
            v55 = *(_QWORD *)(v26 + 24);
            v146 = v24;
            v170 = *(_QWORD *)(v26 + 32);
            v56 = sub_15A9FE0(v28, v55);
            v24 = v146;
            v57 = 1;
            v58 = v56;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v55 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v119 = *(_QWORD *)(v55 + 32);
                  v55 = *(_QWORD *)(v55 + 24);
                  v57 *= v119;
                  continue;
                case 1:
                  v112 = 16;
                  goto LABEL_125;
                case 2:
                  v112 = 32;
                  goto LABEL_125;
                case 3:
                case 9:
                  v112 = 64;
                  goto LABEL_125;
                case 4:
                  v112 = 80;
                  goto LABEL_125;
                case 5:
                case 6:
                  v112 = 128;
                  goto LABEL_125;
                case 7:
                  v142 = v146;
                  v113 = 0;
                  v149 = v58;
                  v165 = v57;
                  goto LABEL_128;
                case 0xB:
                  v112 = *(_DWORD *)(v55 + 8) >> 8;
                  goto LABEL_125;
                case 0xD:
                  sub_15A9930(v28, v55);
                  JUMPOUT(0x129A22F);
                case 0xE:
                  v131 = v146;
                  v134 = v58;
                  v139 = v57;
                  v143 = *(_QWORD *)(v55 + 24);
                  v166 = *(_QWORD *)(v55 + 32);
                  v115 = sub_15A9FE0(v28, v143);
                  v24 = v146;
                  v116 = 1;
                  v58 = v134;
                  v117 = v143;
                  v118 = v115;
                  v57 = v139;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v117 + 8) )
                    {
                      case 0:
                      case 8:
                      case 0xA:
                      case 0xC:
                      case 0x10:
                        v124 = *(_QWORD *)(v117 + 32);
                        v117 = *(_QWORD *)(v117 + 24);
                        v116 *= v124;
                        continue;
                      case 1:
                        v123 = 16;
                        goto LABEL_149;
                      case 2:
                        v123 = 32;
                        goto LABEL_149;
                      case 3:
                      case 9:
                        v123 = 64;
                        goto LABEL_149;
                      case 4:
                        v123 = 80;
                        goto LABEL_149;
                      case 5:
                      case 6:
                        v123 = 128;
                        goto LABEL_149;
                      case 7:
                        v135 = v146;
                        v125 = 0;
                        v140 = v58;
                        v144 = v57;
                        v150 = v116;
                        goto LABEL_156;
                      case 0xB:
                        v123 = *(_DWORD *)(v117 + 8) >> 8;
                        goto LABEL_149;
                      case 0xD:
                        v152 = v116;
                        v128 = (_QWORD *)sub_15A9930(v28, v117);
                        v116 = v152;
                        v57 = v139;
                        v58 = v134;
                        v24 = v131;
                        v123 = 8LL * *v128;
                        goto LABEL_149;
                      case 0xE:
                        v129 = v146;
                        v130 = v134;
                        v132 = v116;
                        v136 = *(_QWORD *)(v117 + 24);
                        v151 = *(_QWORD *)(v117 + 32);
                        v145 = (unsigned int)sub_15A9FE0(v28, v136);
                        v127 = sub_127FA20(v28, v136);
                        v116 = v132;
                        v57 = v139;
                        v58 = v130;
                        v24 = v129;
                        v123 = 8 * v145 * v151 * ((v145 + ((unsigned __int64)(v127 + 7) >> 3) - 1) / v145);
                        goto LABEL_149;
                      case 0xF:
                        v135 = v146;
                        v140 = v58;
                        v144 = v57;
                        v125 = *(_DWORD *)(v117 + 8) >> 8;
                        v150 = v116;
LABEL_156:
                        v126 = sub_15A9520(v28, v125);
                        v116 = v150;
                        v57 = v144;
                        v58 = v140;
                        v24 = v135;
                        v123 = (unsigned int)(8 * v126);
LABEL_149:
                        v112 = 8 * v118 * v166 * ((v118 + ((unsigned __int64)(v123 * v116 + 7) >> 3) - 1) / v118);
                        break;
                    }
                    goto LABEL_125;
                  }
                case 0xF:
                  v142 = v146;
                  v149 = v58;
                  v165 = v57;
                  v113 = *(_DWORD *)(v55 + 8) >> 8;
LABEL_128:
                  v114 = sub_15A9520(v28, v113);
                  v57 = v165;
                  v58 = v149;
                  v24 = v142;
                  v112 = (unsigned int)(8 * v114);
LABEL_125:
                  v29 = 8 * v170 * v58 * ((v58 + ((v112 * v57 + 7) >> 3) - 1) / v58);
                  break;
              }
              break;
            }
            break;
          case 15LL:
            v169 = v24;
            v54 = sub_15A9520(v28, *(_DWORD *)(v26 + 8) >> 8);
            v24 = v169;
            v29 = (unsigned int)(8 * v54);
            break;
        }
        break;
      }
      v167 = (unsigned __int64)(v29 * v27 + 7) >> 3;
      v30 = sub_1646BA0(v24, 0);
      v31 = (unsigned __int64)v158;
      if ( v22 % (int)v167 )
        v22 = v22 + v167 - v22 % (int)v167;
      if ( v22 )
      {
        v74 = a1[9];
        v160 = v30;
        v182 = 1;
        v180[0] = "buf.indexed";
        v181 = 3;
        v75 = sub_1643350(v74);
        v76 = sub_159C470(v75, (unsigned int)v22, 0);
        v77 = v160;
        v175 = v76;
        if ( v158[16] > 0x10u )
        {
          v184 = 257;
          v161 = v155;
          if ( !v155 )
          {
            v111 = *(_QWORD *)v158;
            if ( *(_BYTE *)(*(_QWORD *)v158 + 8LL) == 16 )
              v111 = **(_QWORD **)(v111 + 16);
            v161 = *(_QWORD *)(v111 + 24);
          }
          v147 = v77;
          v89 = sub_1648A60(72, 2);
          v90 = v147;
          v31 = v89;
          if ( v89 )
          {
            v148 = v89;
            v141 = v89 - 48;
            v91 = *(_QWORD *)v158;
            if ( *(_BYTE *)(*(_QWORD *)v158 + 8LL) == 16 )
              v91 = **(_QWORD **)(v91 + 16);
            v133 = v90;
            v137 = *(_DWORD *)(v91 + 8) >> 8;
            v92 = sub_15F9F50(v161, &v175, 1);
            v93 = sub_1646BA0(v92, v137);
            v94 = v133;
            v95 = v93;
            v96 = *(_QWORD *)v158;
            if ( *(_BYTE *)(*(_QWORD *)v158 + 8LL) == 16
              || (v96 = *(_QWORD *)v175, *(_BYTE *)(*(_QWORD *)v175 + 8LL) == 16) )
            {
              v105 = sub_16463B0(v95, *(_QWORD *)(v96 + 32));
              v94 = v133;
              v95 = v105;
            }
            v138 = v94;
            sub_15F1EA0(v31, v95, 32, v141, 2, 0);
            *(_QWORD *)(v31 + 56) = v161;
            *(_QWORD *)(v31 + 64) = sub_15F9F50(v161, &v175, 1);
            sub_15F9CE0(v31, v158, &v175, 1, v183);
            v90 = v138;
          }
          else
          {
            v148 = 0;
          }
          v162 = v90;
          sub_15FA2E0(v31, 1);
          v97 = a1[7];
          v98 = v162;
          if ( v97 )
          {
            v99 = (unsigned __int64 *)a1[8];
            sub_157E9D0(v97 + 40, v31);
            v100 = *(_QWORD *)(v31 + 24);
            v98 = v162;
            v101 = *v99;
            *(_QWORD *)(v31 + 32) = v99;
            v101 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v31 + 24) = v101 | v100 & 7;
            *(_QWORD *)(v101 + 8) = v31 + 24;
            *v99 = *v99 & 7 | (v31 + 24);
          }
          v163 = v98;
          sub_164B780(v148, v180);
          v102 = a1[6];
          v30 = v163;
          if ( v102 )
          {
            v176 = a1[6];
            sub_1623A60(&v176, v102, 2);
            v30 = v163;
            v103 = v31 + 48;
            if ( *(_QWORD *)(v31 + 48) )
            {
              sub_161E7C0(v31 + 48);
              v30 = v163;
              v103 = v31 + 48;
            }
            v104 = v176;
            *(_QWORD *)(v31 + 48) = v176;
            if ( v104 )
            {
              v164 = v30;
              sub_1623210(&v176, v104, v103);
              v30 = v164;
            }
          }
        }
        else
        {
          BYTE4(v183[0]) = 0;
          v78 = sub_15A2E80(v155, (_DWORD)v158, (unsigned int)&v175, 1, 1, (unsigned int)v183, 0);
          v30 = v160;
          v31 = v78;
        }
      }
      v182 = 1;
      v180[0] = "casted";
      v181 = 3;
      if ( v30 != *(_QWORD *)v31 )
      {
        if ( *(_BYTE *)(v31 + 16) > 0x10u )
        {
          v184 = 257;
          v79 = sub_15FDBD0(47, v31, v30, v183, 0);
          v80 = a1[7];
          v31 = v79;
          if ( v80 )
          {
            v81 = (unsigned __int64 *)a1[8];
            sub_157E9D0(v80 + 40, v79);
            v82 = *(_QWORD *)(v31 + 24);
            v83 = *v81;
            *(_QWORD *)(v31 + 32) = v81;
            v83 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v31 + 24) = v83 | v82 & 7;
            *(_QWORD *)(v83 + 8) = v31 + 24;
            *v81 = *v81 & 7 | (v31 + 24);
          }
          sub_164B780(v31, v180);
          v84 = a1[6];
          if ( v84 )
          {
            v176 = a1[6];
            sub_1623A60(&v176, v84, 2);
            v85 = v31 + 48;
            if ( *(_QWORD *)(v31 + 48) )
            {
              sub_161E7C0(v31 + 48);
              v85 = v31 + 48;
            }
            v86 = v176;
            *(_QWORD *)(v31 + 48) = v176;
            if ( v86 )
              sub_1623210(&v176, v86, v85);
          }
        }
        else
        {
          v31 = sub_15A46C0(47, v31, v30, 0);
        }
      }
      v32 = unk_4D0463C;
      if ( unk_4D0463C )
        v32 = sub_126A420(a1[4], v31);
      v159 = v32;
      v184 = 257;
      v33 = sub_1648A60(64, 2);
      v34 = (_QWORD *)v33;
      if ( v33 )
        sub_15F9650(v33, v23, v31, v159, 0);
      v35 = a1[7];
      if ( v35 )
      {
        v36 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v35 + 40, v34);
        v37 = v34[3];
        v38 = *v36;
        v34[4] = v36;
        v38 &= 0xFFFFFFFFFFFFFFF8LL;
        v34[3] = v38 | v37 & 7;
        *(_QWORD *)(v38 + 8) = v34 + 3;
        *v36 = *v36 & 7 | (unsigned __int64)(v34 + 3);
      }
      sub_164B780(v34, v183);
      v39 = a1[6];
      if ( v39 )
      {
        v180[0] = a1[6];
        sub_1623A60(v180, v39, 2);
        if ( v34[6] )
          sub_161E7C0(v34 + 6);
        v40 = v180[0];
        v34[6] = v180[0];
        if ( v40 )
          sub_1623210(v180, v40, v34 + 6);
      }
      ++v174;
      v22 += v167;
      v172 += 40;
    }
    while ( v174 != (_QWORD **)v157 );
  }
  if ( v156 )
  {
    v41 = *((_QWORD *)v156 - 3);
    v42 = *(_DWORD *)(v41 + 32);
    if ( v42 > 0x40 )
    {
      if ( v42 - (unsigned int)sub_16A57B0(v41 + 24) <= 0x40 )
      {
        v43 = **(_QWORD **)(v41 + 24);
        goto LABEL_37;
      }
    }
    else
    {
      v43 = *(_QWORD *)(v41 + 24);
LABEL_37:
      if ( v22 > v43 )
      {
        v44 = sub_1643350(a1[5]);
        v45 = sub_159C470(v44, v22, 0);
        if ( *((_QWORD *)v156 - 3) )
        {
          v46 = *((_QWORD *)v156 - 2);
          v47 = *((_QWORD *)v156 - 1) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v47 = v46;
          if ( v46 )
            *(_QWORD *)(v46 + 16) = *(_QWORD *)(v46 + 16) & 3LL | v47;
        }
        *((_QWORD *)v156 - 3) = v45;
        if ( v45 )
        {
          v48 = *(_QWORD *)(v45 + 8);
          *((_QWORD *)v156 - 2) = v48;
          if ( v48 )
            *(_QWORD *)(v48 + 16) = (unsigned __int64)(v156 - 16) | *(_QWORD *)(v48 + 16) & 3LL;
          *((_QWORD *)v156 - 1) = (v45 + 8) | *((_QWORD *)v156 - 1) & 3LL;
          *(_QWORD *)(v45 + 8) = v156 - 24;
        }
      }
    }
  }
  v49 = (unsigned int)v186;
LABEL_46:
  v184 = 257;
  v50 = *(_QWORD *)(v153 + 24);
  v51 = sub_1285290(a1 + 6, v50, v153, (int)v185, v49, (__int64)v183, 0);
  if ( v185 != v187 )
    _libc_free(v185, v50);
  if ( v177 )
    j_j___libc_free_0(v177, &v179[-v177]);
  return v51;
}
