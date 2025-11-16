// Function: sub_1B710B0
// Address: 0x1b710b0
//
__int64 __fastcall sub_1B710B0(__int64 a1, __int64 a2, __int64 *a3, _BYTE *a4)
{
  __int64 v6; // r14
  __int64 **v7; // r15
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // r11
  __int64 v11; // r11
  __int64 v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rbx
  char v16; // al
  __int64 **v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r8
  __int64 **v20; // rax
  __int64 v21; // rax
  __int64 **v22; // rax
  __int64 **v23; // rbx
  char v24; // al
  __int64 v25; // rax
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rcx
  unsigned __int64 v30; // r8
  __int64 v31; // rax
  _QWORD *v32; // rax
  int v33; // eax
  __int64 v34; // rax
  int v35; // eax
  unsigned int v36; // eax
  __int64 v37; // rsi
  __int64 v38; // rcx
  unsigned __int64 v39; // r8
  char v40; // al
  __int64 **v41; // rax
  char v42; // al
  __int64 **v43; // rdx
  char v44; // al
  __int64 v45; // rdi
  __int64 *v46; // r15
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rsi
  __int64 v50; // rsi
  unsigned __int8 *v51; // rsi
  __int64 v52; // rdi
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rsi
  __int64 v57; // rdx
  unsigned __int8 *v58; // rsi
  __int64 v59; // rdx
  __int64 v60; // rsi
  int v61; // edi
  __int64 v62; // rdi
  __int64 *v63; // rbx
  __int64 v64; // rax
  __int64 v65; // rcx
  __int64 v66; // rsi
  __int64 v67; // rsi
  unsigned __int8 *v68; // rsi
  __int64 v69; // rcx
  __int64 v70; // rcx
  __int64 v71; // rsi
  __int64 v72; // r8
  unsigned __int64 v73; // rcx
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  unsigned int v77; // esi
  int v78; // eax
  __int64 v79; // rax
  __int64 v80; // rax
  _QWORD *v81; // rax
  __int64 v82; // rax
  int v83; // eax
  __int64 v84; // rax
  _QWORD *v85; // rax
  int v86; // eax
  __int64 v87; // rax
  __int64 v88; // rax
  unsigned int v89; // esi
  int v90; // eax
  _QWORD *v91; // rax
  __int64 v92; // rax
  unsigned int v93; // esi
  int v94; // eax
  __int64 v95; // rax
  _QWORD *v96; // rax
  __int64 v97; // rax
  __int64 v98; // rax
  int v99; // r11d
  __int64 v100; // rdi
  __int64 v101; // rcx
  __int64 v102; // rax
  __int64 v103; // rsi
  __int64 v104; // rsi
  __int64 v105; // rdx
  unsigned __int8 *v106; // rsi
  __int64 v107; // rax
  __int64 *v108; // rbx
  __int64 v109; // rax
  __int64 v110; // rcx
  __int64 v111; // rsi
  __int64 v112; // rsi
  unsigned __int8 *v113; // rsi
  __int64 v114; // rax
  __int64 *v115; // rbx
  __int64 v116; // rax
  __int64 v117; // rcx
  __int64 v118; // rsi
  __int64 v119; // rsi
  unsigned __int8 *v120; // rsi
  __int64 v121; // rdx
  __int64 v122; // rsi
  int v123; // edi
  __int64 v124; // rax
  __int64 *v125; // rbx
  __int64 v126; // rax
  __int64 v127; // rcx
  __int64 v128; // rsi
  __int64 v129; // rsi
  unsigned __int8 *v130; // rsi
  __int64 v131; // [rsp+0h] [rbp-C0h]
  __int64 v132; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v133; // [rsp+8h] [rbp-B8h]
  __int64 v134; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v135; // [rsp+10h] [rbp-B0h]
  __int64 v136; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v137; // [rsp+18h] [rbp-A8h]
  __int64 v138; // [rsp+18h] [rbp-A8h]
  __int64 v139; // [rsp+18h] [rbp-A8h]
  __int64 v140; // [rsp+18h] [rbp-A8h]
  __int64 v141; // [rsp+18h] [rbp-A8h]
  __int64 v142; // [rsp+18h] [rbp-A8h]
  __int64 v143; // [rsp+20h] [rbp-A0h]
  __int64 v144; // [rsp+20h] [rbp-A0h]
  __int64 v145; // [rsp+20h] [rbp-A0h]
  __int64 v146; // [rsp+20h] [rbp-A0h]
  __int64 v147; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v148; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v149; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v150; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v151; // [rsp+20h] [rbp-A0h]
  __int64 v152; // [rsp+28h] [rbp-98h]
  __int64 v153; // [rsp+28h] [rbp-98h]
  __int64 v154; // [rsp+28h] [rbp-98h]
  unsigned __int64 v155; // [rsp+28h] [rbp-98h]
  __int64 v156; // [rsp+28h] [rbp-98h]
  unsigned __int64 v157; // [rsp+28h] [rbp-98h]
  unsigned __int64 v158; // [rsp+28h] [rbp-98h]
  unsigned __int64 v159; // [rsp+28h] [rbp-98h]
  __int64 v160; // [rsp+28h] [rbp-98h]
  __int64 v161; // [rsp+28h] [rbp-98h]
  __int64 v162; // [rsp+28h] [rbp-98h]
  __int64 v163; // [rsp+28h] [rbp-98h]
  int v164; // [rsp+28h] [rbp-98h]
  int v165; // [rsp+30h] [rbp-90h]
  __int64 v166; // [rsp+30h] [rbp-90h]
  __int64 v167; // [rsp+30h] [rbp-90h]
  __int64 v168; // [rsp+30h] [rbp-90h]
  __int64 v169; // [rsp+30h] [rbp-90h]
  __int64 v170; // [rsp+30h] [rbp-90h]
  __int64 *v171; // [rsp+30h] [rbp-90h]
  unsigned __int64 v172; // [rsp+30h] [rbp-90h]
  __int64 v173; // [rsp+30h] [rbp-90h]
  unsigned __int64 v174; // [rsp+30h] [rbp-90h]
  __int64 v175; // [rsp+30h] [rbp-90h]
  __int64 v176; // [rsp+30h] [rbp-90h]
  __int64 v177; // [rsp+30h] [rbp-90h]
  __int64 v178; // [rsp+30h] [rbp-90h]
  __int64 *v179; // [rsp+30h] [rbp-90h]
  int v180; // [rsp+30h] [rbp-90h]
  int v181; // [rsp+30h] [rbp-90h]
  __int64 v183; // [rsp+48h] [rbp-78h] BYREF
  __int64 v184[2]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v185; // [rsp+60h] [rbp-60h]
  _BYTE v186[16]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v187; // [rsp+80h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 16) > 0x10u || (v6 = sub_14DBA30(a1, (__int64)a4, 0)) == 0 )
    v6 = a1;
  v7 = *(__int64 ***)v6;
  v8 = 1;
  v9 = *(_QWORD *)v6;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v9 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v31 = *(_QWORD *)(v9 + 32);
        v9 = *(_QWORD *)(v9 + 24);
        v8 *= v31;
        continue;
      case 1:
        v10 = 16;
        break;
      case 2:
        v10 = 32;
        break;
      case 3:
      case 9:
        v10 = 64;
        break;
      case 4:
        v10 = 80;
        break;
      case 5:
      case 6:
        v10 = 128;
        break;
      case 7:
        v10 = 8 * (unsigned int)sub_15A9520((__int64)a4, 0);
        break;
      case 0xB:
        v10 = *(_DWORD *)(v9 + 8) >> 8;
        break;
      case 0xD:
        v10 = 8LL * *(_QWORD *)sub_15A9930((__int64)a4, v9);
        break;
      case 0xE:
        v152 = *(_QWORD *)(v9 + 24);
        v166 = *(_QWORD *)(v9 + 32);
        v27 = sub_15A9FE0((__int64)a4, v152);
        v28 = v152;
        v29 = 1;
        v30 = v27;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v28 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v88 = *(_QWORD *)(v28 + 32);
              v28 = *(_QWORD *)(v28 + 24);
              v29 *= v88;
              continue;
            case 1:
              v76 = 16;
              goto LABEL_125;
            case 2:
              v76 = 32;
              goto LABEL_125;
            case 3:
            case 9:
              v76 = 64;
              goto LABEL_125;
            case 4:
              v76 = 80;
              goto LABEL_125;
            case 5:
            case 6:
              v76 = 128;
              goto LABEL_125;
            case 7:
              v146 = v29;
              v89 = 0;
              v158 = v30;
              goto LABEL_152;
            case 0xB:
              v76 = *(_DWORD *)(v28 + 8) >> 8;
              goto LABEL_125;
            case 0xD:
              v147 = v29;
              v159 = v30;
              v91 = (_QWORD *)sub_15A9930((__int64)a4, v28);
              v30 = v159;
              v29 = v147;
              v76 = 8LL * *v91;
              goto LABEL_125;
            case 0xE:
              v132 = v29;
              v135 = v30;
              v139 = *(_QWORD *)(v28 + 24);
              v160 = *(_QWORD *)(v28 + 32);
              v148 = (unsigned int)sub_15A9FE0((__int64)a4, v139);
              v92 = sub_127FA20((__int64)a4, v139);
              v30 = v135;
              v29 = v132;
              v76 = 8 * v160 * v148 * ((v148 + ((unsigned __int64)(v92 + 7) >> 3) - 1) / v148);
              goto LABEL_125;
            case 0xF:
              v146 = v29;
              v158 = v30;
              v89 = *(_DWORD *)(v28 + 8) >> 8;
LABEL_152:
              v90 = sub_15A9520((__int64)a4, v89);
              v30 = v158;
              v29 = v146;
              v76 = (unsigned int)(8 * v90);
LABEL_125:
              v10 = 8 * v30 * v166 * ((v30 + ((unsigned __int64)(v76 * v29 + 7) >> 3) - 1) / v30);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v10 = 8 * (unsigned int)sub_15A9520((__int64)a4, *(_DWORD *)(v9 + 8) >> 8);
        break;
    }
    break;
  }
  v11 = v8 * v10;
  v12 = a2;
  v13 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v12 + 8) )
    {
      case 1:
        v14 = 16;
        goto LABEL_12;
      case 2:
        v14 = 32;
        goto LABEL_12;
      case 3:
      case 9:
        v14 = 64;
        goto LABEL_12;
      case 4:
        v14 = 80;
        goto LABEL_12;
      case 5:
      case 6:
        v15 = v13 << 7;
        v16 = *((_BYTE *)v7 + 8);
        if ( v16 != 13 )
          goto LABEL_22;
        goto LABEL_13;
      case 7:
        v168 = v11;
        v33 = sub_15A9520((__int64)a4, 0);
        v11 = v168;
        v14 = (unsigned int)(8 * v33);
        goto LABEL_12;
      case 0xB:
        v14 = *(_DWORD *)(v12 + 8) >> 8;
        goto LABEL_12;
      case 0xD:
        v167 = v11;
        v32 = (_QWORD *)sub_15A9930((__int64)a4, v12);
        v11 = v167;
        v14 = 8LL * *v32;
        goto LABEL_12;
      case 0xE:
        v143 = v11;
        v153 = *(_QWORD *)(v12 + 24);
        v170 = *(_QWORD *)(v12 + 32);
        v36 = sub_15A9FE0((__int64)a4, v153);
        v11 = v143;
        v37 = v153;
        v38 = 1;
        v39 = v36;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v37 + 8) )
          {
            case 1:
              v87 = 16;
              goto LABEL_146;
            case 2:
              v87 = 32;
              goto LABEL_146;
            case 3:
            case 9:
              v87 = 64;
              goto LABEL_146;
            case 4:
              v87 = 80;
              goto LABEL_146;
            case 5:
            case 6:
              v87 = 128;
              goto LABEL_146;
            case 7:
              v140 = v38;
              v93 = 0;
              v149 = v39;
              v161 = v11;
              goto LABEL_161;
            case 0xB:
              v87 = *(_DWORD *)(v37 + 8) >> 8;
              goto LABEL_146;
            case 0xD:
              v142 = v38;
              v151 = v39;
              v163 = v11;
              v96 = (_QWORD *)sub_15A9930((__int64)a4, v37);
              v11 = v163;
              v39 = v151;
              v38 = v142;
              v87 = 8LL * *v96;
              goto LABEL_146;
            case 0xE:
              v131 = v38;
              v133 = v39;
              v136 = v143;
              v141 = *(_QWORD *)(v37 + 24);
              v162 = *(_QWORD *)(v37 + 32);
              v150 = (unsigned int)sub_15A9FE0((__int64)a4, v141);
              v95 = sub_127FA20((__int64)a4, v141);
              v11 = v136;
              v39 = v133;
              v38 = v131;
              v87 = 8 * v150 * v162 * ((v150 + ((unsigned __int64)(v95 + 7) >> 3) - 1) / v150);
              goto LABEL_146;
            case 0xF:
              v140 = v38;
              v149 = v39;
              v161 = v11;
              v93 = *(_DWORD *)(v37 + 8) >> 8;
LABEL_161:
              v94 = sub_15A9520((__int64)a4, v93);
              v11 = v161;
              v39 = v149;
              v38 = v140;
              v87 = (unsigned int)(8 * v94);
LABEL_146:
              v14 = 8 * v170 * v39 * ((v39 + ((unsigned __int64)(v87 * v38 + 7) >> 3) - 1) / v39);
              goto LABEL_12;
            case 0x10:
              v97 = *(_QWORD *)(v37 + 32);
              v37 = *(_QWORD *)(v37 + 24);
              v38 *= v97;
              continue;
            default:
              goto LABEL_6;
          }
        }
      case 0xF:
        v169 = v11;
        v35 = sub_15A9520((__int64)a4, *(_DWORD *)(v12 + 8) >> 8);
        v11 = v169;
        v14 = (unsigned int)(8 * v35);
LABEL_12:
        v15 = v14 * v13;
        v16 = *((_BYTE *)v7 + 8);
        if ( v16 == 13 )
        {
LABEL_13:
          if ( a2 == *v7[2] )
          {
            LODWORD(v184[0]) = 0;
            v187 = 257;
            return sub_12A9E60(a3, v6, (__int64)v184, 1, (__int64)v186);
          }
          if ( v15 != v11 )
            goto LABEL_15;
          goto LABEL_74;
        }
LABEL_22:
        if ( v15 != v11 )
        {
          if ( v16 == 16 )
          {
            if ( *(_BYTE *)(*v7[2] + 8) != 15 )
              goto LABEL_15;
          }
          else if ( v16 != 15 )
          {
            goto LABEL_29;
          }
          v165 = v11;
          v20 = (__int64 **)sub_15A9650((__int64)a4, (__int64)v7);
          LODWORD(v11) = v165;
          v7 = v20;
          v185 = 257;
          if ( v20 != *(__int64 ***)v6 )
          {
            if ( *(_BYTE *)(v6 + 16) > 0x10u )
            {
              v187 = 257;
              v98 = sub_15FDBD0(45, v6, (__int64)v20, (__int64)v186, 0);
              v99 = v165;
              v6 = v98;
              v100 = a3[1];
              if ( v100 )
              {
                v164 = v165;
                v179 = (__int64 *)a3[2];
                sub_157E9D0(v100 + 40, v98);
                v99 = v164;
                v101 = *v179;
                v102 = *(_QWORD *)(v6 + 24) & 7LL;
                *(_QWORD *)(v6 + 32) = v179;
                v101 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v6 + 24) = v101 | v102;
                *(_QWORD *)(v101 + 8) = v6 + 24;
                *v179 = *v179 & 7 | (v6 + 24);
              }
              v180 = v99;
              sub_164B780(v6, v184);
              LODWORD(v11) = v180;
              v103 = *a3;
              if ( *a3 )
              {
                v183 = *a3;
                sub_1623A60((__int64)&v183, v103, 2);
                v104 = *(_QWORD *)(v6 + 48);
                v105 = v6 + 48;
                LODWORD(v11) = v180;
                if ( v104 )
                {
                  sub_161E7C0(v6 + 48, v104);
                  LODWORD(v11) = v180;
                  v105 = v6 + 48;
                }
                v106 = (unsigned __int8 *)v183;
                *(_QWORD *)(v6 + 48) = v183;
                if ( v106 )
                {
                  v181 = v11;
                  sub_1623210((__int64)&v183, v106, v105);
                  LODWORD(v11) = v181;
                }
              }
            }
            else
            {
              v21 = sub_15A46C0(45, (__int64 ***)v6, v20, 0);
              LODWORD(v11) = v165;
              v6 = v21;
            }
          }
          v16 = *((_BYTE *)v7 + 8);
LABEL_29:
          if ( v16 == 11 )
          {
            if ( *a4 )
            {
              v18 = (__int64)v7;
              v19 = 1;
LABEL_123:
              v69 = *(_DWORD *)(v18 + 8) >> 8;
LABEL_116:
              v70 = v19 * v69;
              v71 = a2;
              v72 = 1;
              v73 = (unsigned __int64)(v70 + 7) >> 3;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v71 + 8) )
                {
                  case 1:
                    v74 = 16;
                    goto LABEL_119;
                  case 2:
                    v74 = 32;
                    goto LABEL_119;
                  case 3:
                  case 9:
                    v74 = 64;
                    goto LABEL_119;
                  case 4:
                    v74 = 80;
                    goto LABEL_119;
                  case 5:
                  case 6:
                    v74 = 128;
                    goto LABEL_119;
                  case 7:
                    v154 = v72;
                    v77 = 0;
                    v172 = v73;
                    goto LABEL_129;
                  case 0xB:
                    v74 = *(_DWORD *)(v71 + 8) >> 8;
                    goto LABEL_119;
                  case 0xD:
                    v156 = v72;
                    v174 = v73;
                    v81 = (_QWORD *)sub_15A9930((__int64)a4, v71);
                    v73 = v174;
                    v72 = v156;
                    v74 = 8LL * *v81;
                    goto LABEL_119;
                  case 0xE:
                    v134 = v72;
                    v137 = v73;
                    v144 = *(_QWORD *)(v71 + 24);
                    v173 = *(_QWORD *)(v71 + 32);
                    v155 = (unsigned int)sub_15A9FE0((__int64)a4, v144);
                    v80 = sub_127FA20((__int64)a4, v144);
                    v73 = v137;
                    v72 = v134;
                    v74 = 8 * v173 * v155 * ((v155 + ((unsigned __int64)(v80 + 7) >> 3) - 1) / v155);
                    goto LABEL_119;
                  case 0xF:
                    v154 = v72;
                    v172 = v73;
                    v77 = *(_DWORD *)(v71 + 8) >> 8;
LABEL_129:
                    v78 = sub_15A9520((__int64)a4, v77);
                    v73 = v172;
                    v72 = v154;
                    v74 = (unsigned int)(8 * v78);
LABEL_119:
                    v187 = 257;
                    v75 = sub_15A0680(*(_QWORD *)v6, 8 * (v73 - ((unsigned __int64)(v72 * v74 + 7) >> 3)), 0);
                    v6 = sub_156E320(a3, v6, v75, (__int64)v186, 0);
                    break;
                  case 0x10:
                    v79 = *(_QWORD *)(v71 + 32);
                    v71 = *(_QWORD *)(v71 + 24);
                    v72 *= v79;
                    continue;
                  default:
                    goto LABEL_6;
                }
                break;
              }
            }
LABEL_31:
            v22 = (__int64 **)sub_1644900(*v7, v15);
            v185 = 257;
            v23 = v22;
            if ( v22 != *(__int64 ***)v6 )
            {
              if ( *(_BYTE *)(v6 + 16) > 0x10u )
              {
                v187 = 257;
                v6 = sub_15FDF30((_QWORD *)v6, (__int64)v22, (__int64)v186, 0);
                v45 = a3[1];
                if ( v45 )
                {
                  v46 = (__int64 *)a3[2];
                  sub_157E9D0(v45 + 40, v6);
                  v47 = *(_QWORD *)(v6 + 24);
                  v48 = *v46;
                  *(_QWORD *)(v6 + 32) = v46;
                  v48 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v6 + 24) = v48 | v47 & 7;
                  *(_QWORD *)(v48 + 8) = v6 + 24;
                  *v46 = *v46 & 7 | (v6 + 24);
                }
                sub_164B780(v6, v184);
                v49 = *a3;
                if ( *a3 )
                {
                  v183 = *a3;
                  sub_1623A60((__int64)&v183, v49, 2);
                  v50 = *(_QWORD *)(v6 + 48);
                  if ( v50 )
                    sub_161E7C0(v6 + 48, v50);
                  v51 = (unsigned __int8 *)v183;
                  *(_QWORD *)(v6 + 48) = v183;
                  if ( v51 )
                    sub_1623210((__int64)&v183, v51, v6 + 48);
                }
              }
              else
              {
                v6 = sub_15A4670((__int64 ***)v6, v22);
              }
            }
            if ( (__int64 **)a2 != v23 )
            {
              v24 = *(_BYTE *)(a2 + 8);
              if ( v24 == 16 )
                v24 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
              v185 = 257;
              if ( v24 == 15 )
              {
                if ( a2 == *(_QWORD *)v6 )
                  goto LABEL_41;
                if ( *(_BYTE *)(v6 + 16) <= 0x10u )
                {
                  v6 = sub_15A46C0(46, (__int64 ***)v6, (__int64 **)a2, 0);
                  goto LABEL_41;
                }
                v60 = v6;
                v187 = 257;
                v61 = 46;
                v59 = a2;
              }
              else
              {
                if ( a2 == *(_QWORD *)v6 )
                  goto LABEL_41;
                if ( *(_BYTE *)(v6 + 16) <= 0x10u )
                {
                  v6 = sub_15A46C0(47, (__int64 ***)v6, (__int64 **)a2, 0);
                  goto LABEL_41;
                }
                v59 = a2;
                v187 = 257;
                v60 = v6;
                v61 = 47;
              }
              v6 = sub_15FDBD0(v61, v60, v59, (__int64)v186, 0);
              v62 = a3[1];
              if ( v62 )
              {
                v63 = (__int64 *)a3[2];
                sub_157E9D0(v62 + 40, v6);
                v64 = *(_QWORD *)(v6 + 24);
                v65 = *v63;
                *(_QWORD *)(v6 + 32) = v63;
                v65 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v6 + 24) = v65 | v64 & 7;
                *(_QWORD *)(v65 + 8) = v6 + 24;
                *v63 = *v63 & 7 | (v6 + 24);
              }
              sub_164B780(v6, v184);
              v66 = *a3;
              if ( *a3 )
              {
                v183 = *a3;
                sub_1623A60((__int64)&v183, v66, 2);
                v67 = *(_QWORD *)(v6 + 48);
                if ( v67 )
                  sub_161E7C0(v6 + 48, v67);
                v68 = (unsigned __int8 *)v183;
                *(_QWORD *)(v6 + 48) = v183;
                if ( v68 )
                  sub_1623210((__int64)&v183, v68, v6 + 48);
              }
            }
LABEL_41:
            if ( *(_BYTE *)(v6 + 16) > 0x10u )
              return v6;
            goto LABEL_42;
          }
LABEL_15:
          v17 = (__int64 **)sub_1644900(*v7, v11);
          v185 = 257;
          v7 = v17;
          if ( v17 != *(__int64 ***)v6 )
          {
            if ( *(_BYTE *)(v6 + 16) > 0x10u )
            {
              v187 = 257;
              v6 = sub_15FDBD0(47, v6, (__int64)v17, (__int64)v186, 0);
              v52 = a3[1];
              if ( v52 )
              {
                v171 = (__int64 *)a3[2];
                sub_157E9D0(v52 + 40, v6);
                v53 = *v171;
                v54 = *(_QWORD *)(v6 + 24) & 7LL;
                *(_QWORD *)(v6 + 32) = v171;
                v53 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v6 + 24) = v53 | v54;
                *(_QWORD *)(v53 + 8) = v6 + 24;
                *v171 = *v171 & 7 | (v6 + 24);
              }
              sub_164B780(v6, v184);
              v55 = *a3;
              if ( *a3 )
              {
                v183 = *a3;
                sub_1623A60((__int64)&v183, v55, 2);
                v56 = *(_QWORD *)(v6 + 48);
                v57 = v6 + 48;
                if ( v56 )
                {
                  sub_161E7C0(v6 + 48, v56);
                  v57 = v6 + 48;
                }
                v58 = (unsigned __int8 *)v183;
                *(_QWORD *)(v6 + 48) = v183;
                if ( v58 )
                  sub_1623210((__int64)&v183, v58, v57);
              }
            }
            else
            {
              v6 = sub_15A46C0(47, (__int64 ***)v6, v17, 0);
            }
          }
          if ( *a4 )
          {
            v18 = (__int64)v7;
            v19 = 1;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v18 + 8) )
              {
                case 1:
                  v69 = 16;
                  goto LABEL_116;
                case 2:
                  v69 = 32;
                  goto LABEL_116;
                case 3:
                case 9:
                  v69 = 64;
                  goto LABEL_116;
                case 4:
                  v69 = 80;
                  goto LABEL_116;
                case 5:
                case 6:
                  v69 = 128;
                  goto LABEL_116;
                case 7:
                  v178 = v19;
                  v86 = sub_15A9520((__int64)a4, 0);
                  v19 = v178;
                  v69 = (unsigned int)(8 * v86);
                  goto LABEL_116;
                case 0xB:
                  goto LABEL_123;
                case 0xD:
                  v177 = v19;
                  v85 = (_QWORD *)sub_15A9930((__int64)a4, v18);
                  v19 = v177;
                  v69 = 8LL * *v85;
                  goto LABEL_116;
                case 0xE:
                  v138 = v19;
                  v145 = *(_QWORD *)(v18 + 24);
                  v176 = *(_QWORD *)(v18 + 32);
                  v157 = (unsigned int)sub_15A9FE0((__int64)a4, v145);
                  v84 = sub_127FA20((__int64)a4, v145);
                  v19 = v138;
                  v69 = 8 * v176 * v157 * ((v157 + ((unsigned __int64)(v84 + 7) >> 3) - 1) / v157);
                  goto LABEL_116;
                case 0xF:
                  v175 = v19;
                  v83 = sub_15A9520((__int64)a4, *(_DWORD *)(v18 + 8) >> 8);
                  v19 = v175;
                  v69 = (unsigned int)(8 * v83);
                  goto LABEL_116;
                case 0x10:
                  v82 = *(_QWORD *)(v18 + 32);
                  v18 = *(_QWORD *)(v18 + 24);
                  v19 *= v82;
                  continue;
                default:
                  goto LABEL_6;
              }
            }
          }
          goto LABEL_31;
        }
        if ( v16 == 16 )
          v16 = *(_BYTE *)(*v7[2] + 8);
        if ( v16 == 15 )
        {
          v40 = *(_BYTE *)(a2 + 8);
          if ( v40 == 16 )
            v40 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
          if ( v40 == 15 )
          {
            v185 = 257;
            if ( a2 == *(_QWORD *)v6 )
              goto LABEL_85;
            if ( *(_BYTE *)(v6 + 16) <= 0x10u )
            {
              v6 = sub_15A46C0(47, (__int64 ***)v6, (__int64 **)a2, 0);
              goto LABEL_85;
            }
            v121 = a2;
            v122 = v6;
            v187 = 257;
            v123 = 47;
            goto LABEL_197;
          }
          v41 = (__int64 **)sub_15A9650((__int64)a4, (__int64)v7);
          v185 = 257;
          v7 = v41;
          if ( v41 != *(__int64 ***)v6 )
          {
            if ( *(_BYTE *)(v6 + 16) > 0x10u )
            {
              v187 = 257;
              v6 = sub_15FDBD0(45, v6, (__int64)v41, (__int64)v186, 0);
              v114 = a3[1];
              if ( v114 )
              {
                v115 = (__int64 *)a3[2];
                sub_157E9D0(v114 + 40, v6);
                v116 = *(_QWORD *)(v6 + 24);
                v117 = *v115;
                *(_QWORD *)(v6 + 32) = v115;
                v117 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v6 + 24) = v117 | v116 & 7;
                *(_QWORD *)(v117 + 8) = v6 + 24;
                *v115 = *v115 & 7 | (v6 + 24);
              }
              sub_164B780(v6, v184);
              v118 = *a3;
              if ( *a3 )
              {
                v183 = *a3;
                sub_1623A60((__int64)&v183, v118, 2);
                v119 = *(_QWORD *)(v6 + 48);
                if ( v119 )
                  sub_161E7C0(v6 + 48, v119);
                v120 = (unsigned __int8 *)v183;
                *(_QWORD *)(v6 + 48) = v183;
                if ( v120 )
                  sub_1623210((__int64)&v183, v120, v6 + 48);
              }
            }
            else
            {
              v6 = sub_15A46C0(45, (__int64 ***)v6, v41, 0);
            }
          }
        }
LABEL_74:
        v42 = *(_BYTE *)(a2 + 8);
        if ( v42 == 16 )
          v42 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
        v43 = (__int64 **)a2;
        if ( v42 == 15 )
          v43 = (__int64 **)sub_15A9650((__int64)a4, a2);
        if ( v7 != v43 )
        {
          v185 = 257;
          if ( v43 != *(__int64 ***)v6 )
          {
            if ( *(_BYTE *)(v6 + 16) > 0x10u )
            {
              v187 = 257;
              v6 = sub_15FDBD0(47, v6, (__int64)v43, (__int64)v186, 0);
              v107 = a3[1];
              if ( v107 )
              {
                v108 = (__int64 *)a3[2];
                sub_157E9D0(v107 + 40, v6);
                v109 = *(_QWORD *)(v6 + 24);
                v110 = *v108;
                *(_QWORD *)(v6 + 32) = v108;
                v110 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v6 + 24) = v110 | v109 & 7;
                *(_QWORD *)(v110 + 8) = v6 + 24;
                *v108 = *v108 & 7 | (v6 + 24);
              }
              sub_164B780(v6, v184);
              v111 = *a3;
              if ( *a3 )
              {
                v183 = *a3;
                sub_1623A60((__int64)&v183, v111, 2);
                v112 = *(_QWORD *)(v6 + 48);
                if ( v112 )
                  sub_161E7C0(v6 + 48, v112);
                v113 = (unsigned __int8 *)v183;
                *(_QWORD *)(v6 + 48) = v183;
                if ( v113 )
                  sub_1623210((__int64)&v183, v113, v6 + 48);
              }
            }
            else
            {
              v6 = sub_15A46C0(47, (__int64 ***)v6, v43, 0);
            }
          }
        }
        v44 = *(_BYTE *)(a2 + 8);
        if ( v44 == 16 )
          v44 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
        if ( v44 == 15 )
        {
          v185 = 257;
          if ( a2 != *(_QWORD *)v6 )
          {
            if ( *(_BYTE *)(v6 + 16) <= 0x10u )
            {
              v6 = sub_15A46C0(46, (__int64 ***)v6, (__int64 **)a2, 0);
              goto LABEL_85;
            }
            v121 = a2;
            v187 = 257;
            v122 = v6;
            v123 = 46;
LABEL_197:
            v6 = sub_15FDBD0(v123, v122, v121, (__int64)v186, 0);
            v124 = a3[1];
            if ( v124 )
            {
              v125 = (__int64 *)a3[2];
              sub_157E9D0(v124 + 40, v6);
              v126 = *(_QWORD *)(v6 + 24);
              v127 = *v125;
              *(_QWORD *)(v6 + 32) = v125;
              v127 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v6 + 24) = v127 | v126 & 7;
              *(_QWORD *)(v127 + 8) = v6 + 24;
              *v125 = *v125 & 7 | (v6 + 24);
            }
            sub_164B780(v6, v184);
            v128 = *a3;
            if ( *a3 )
            {
              v183 = *a3;
              sub_1623A60((__int64)&v183, v128, 2);
              v129 = *(_QWORD *)(v6 + 48);
              if ( v129 )
                sub_161E7C0(v6 + 48, v129);
              v130 = (unsigned __int8 *)v183;
              *(_QWORD *)(v6 + 48) = v183;
              if ( v130 )
                sub_1623210((__int64)&v183, v130, v6 + 48);
            }
          }
        }
LABEL_85:
        if ( *(_BYTE *)(v6 + 16) != 5 )
          return v6;
LABEL_42:
        v25 = sub_14DBA30(v6, (__int64)a4, 0);
        if ( v25 )
          return v25;
        return v6;
      case 0x10:
        v34 = *(_QWORD *)(v12 + 32);
        v12 = *(_QWORD *)(v12 + 24);
        v13 *= v34;
        continue;
      default:
LABEL_6:
        BUG();
    }
  }
}
