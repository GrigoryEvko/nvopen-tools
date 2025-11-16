// Function: sub_174CCF0
// Address: 0x174ccf0
//
__int64 __fastcall sub_174CCF0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v13; // r14
  __int64 **v14; // r13
  unsigned __int8 *v15; // rsi
  __m128 v16; // xmm0
  int v17; // eax
  void (__fastcall *v18)(_BYTE *, __int64, __int64); // rax
  unsigned __int8 *v19; // rsi
  __int64 v20; // rax
  unsigned __int8 *v21; // rsi
  __int64 v22; // r15
  __int64 v23; // r14
  unsigned __int64 v24; // rax
  __int64 v25; // r14
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 v29; // rdx
  unsigned int v30; // r13d
  unsigned int v31; // eax
  __int64 v32; // rdx
  unsigned int v33; // eax
  __int64 v34; // rdi
  __int64 v35; // rsi
  __int64 v36; // rcx
  unsigned __int64 v37; // r13
  __int64 v38; // rax
  unsigned int v39; // eax
  __int64 v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // rcx
  unsigned __int64 v43; // r13
  __int64 v44; // rax
  unsigned __int64 v45; // r13
  __int64 v46; // rdi
  __int64 v47; // rcx
  __int64 v48; // r15
  __int64 v49; // rsi
  __int64 v50; // r8
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r15
  unsigned __int64 v54; // rsi
  unsigned __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  _QWORD *v60; // rax
  __int64 v61; // r13
  __int64 *v62; // r14
  __int64 v63; // rax
  __int64 v64; // rcx
  __int64 v65; // rdx
  __int64 v66; // rsi
  unsigned __int8 *v67; // rsi
  __int16 v68; // dx
  __int64 v69; // rax
  __int64 **v70; // rcx
  unsigned __int8 *v71; // rax
  __int64 v72; // r15
  __int64 v73; // r14
  _QWORD *v74; // rax
  double v75; // xmm4_8
  double v76; // xmm5_8
  __int64 v77; // r12
  __int64 v78; // rbx
  _QWORD *v79; // rax
  double v80; // xmm4_8
  double v81; // xmm5_8
  __int64 v82; // rax
  int v83; // eax
  int v84; // eax
  _QWORD *v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  int v88; // eax
  _QWORD *v89; // rax
  int v90; // eax
  unsigned int v91; // eax
  __int64 v92; // rdi
  __int64 v93; // rsi
  __int64 v94; // r10
  unsigned __int64 v95; // r11
  unsigned int v96; // esi
  int v97; // eax
  unsigned __int64 v98; // r15
  __int64 v99; // rax
  _QWORD *v100; // rax
  __int64 v101; // rax
  int v102; // eax
  __int64 v103; // rax
  _QWORD *v104; // rax
  __int64 v105; // rax
  int v106; // eax
  __int64 v107; // rax
  unsigned int v108; // esi
  int v109; // eax
  _QWORD *v110; // rax
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // [rsp+0h] [rbp-150h]
  unsigned __int64 v114; // [rsp+8h] [rbp-148h]
  __int64 v115; // [rsp+10h] [rbp-140h]
  __int64 v116; // [rsp+10h] [rbp-140h]
  __int64 v117; // [rsp+18h] [rbp-138h]
  __int64 v118; // [rsp+18h] [rbp-138h]
  __int64 v119; // [rsp+20h] [rbp-130h]
  __int64 v120; // [rsp+20h] [rbp-130h]
  __int64 v121; // [rsp+20h] [rbp-130h]
  __int64 v122; // [rsp+20h] [rbp-130h]
  __int64 v123; // [rsp+20h] [rbp-130h]
  __int64 v124; // [rsp+28h] [rbp-128h]
  __int64 v125; // [rsp+28h] [rbp-128h]
  __int64 v126; // [rsp+28h] [rbp-128h]
  unsigned __int64 v127; // [rsp+28h] [rbp-128h]
  unsigned __int64 v128; // [rsp+28h] [rbp-128h]
  unsigned __int64 v129; // [rsp+28h] [rbp-128h]
  __int64 v130; // [rsp+30h] [rbp-120h]
  __int64 v131; // [rsp+30h] [rbp-120h]
  __int64 v132; // [rsp+30h] [rbp-120h]
  unsigned __int64 v133; // [rsp+30h] [rbp-120h]
  __int64 v134; // [rsp+30h] [rbp-120h]
  __int64 v135; // [rsp+30h] [rbp-120h]
  __int64 v136; // [rsp+30h] [rbp-120h]
  __int64 v137; // [rsp+30h] [rbp-120h]
  __int64 v138; // [rsp+30h] [rbp-120h]
  __int64 v139; // [rsp+38h] [rbp-118h]
  unsigned __int64 v140; // [rsp+38h] [rbp-118h]
  __int64 v141; // [rsp+38h] [rbp-118h]
  __int64 v142; // [rsp+38h] [rbp-118h]
  __int64 v143; // [rsp+38h] [rbp-118h]
  __int64 v144; // [rsp+38h] [rbp-118h]
  __int64 v145; // [rsp+38h] [rbp-118h]
  __int64 v146; // [rsp+38h] [rbp-118h]
  __int64 v147; // [rsp+38h] [rbp-118h]
  __int64 v148; // [rsp+38h] [rbp-118h]
  __int64 v149; // [rsp+38h] [rbp-118h]
  __int64 v150; // [rsp+38h] [rbp-118h]
  __int64 v151; // [rsp+38h] [rbp-118h]
  __int64 v152; // [rsp+40h] [rbp-110h]
  unsigned __int64 v153; // [rsp+40h] [rbp-110h]
  unsigned int v154; // [rsp+40h] [rbp-110h]
  __int64 v155; // [rsp+40h] [rbp-110h]
  __int64 v156; // [rsp+40h] [rbp-110h]
  __int64 v157; // [rsp+40h] [rbp-110h]
  __int64 v158; // [rsp+40h] [rbp-110h]
  __int64 v159; // [rsp+40h] [rbp-110h]
  unsigned int v161; // [rsp+5Ch] [rbp-F4h] BYREF
  __int64 v162; // [rsp+60h] [rbp-F0h] BYREF
  unsigned __int8 *v163; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v164[2]; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v165; // [rsp+80h] [rbp-D0h]
  unsigned __int8 *v166[2]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v167; // [rsp+A0h] [rbp-B0h]
  unsigned __int8 *v168; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v169; // [rsp+B8h] [rbp-98h]
  __int64 *v170; // [rsp+C0h] [rbp-90h]
  __int64 v171; // [rsp+C8h] [rbp-88h]
  __int64 v172; // [rsp+D0h] [rbp-80h]
  int v173; // [rsp+D8h] [rbp-78h]
  __m128 v174; // [rsp+E0h] [rbp-70h]
  _BYTE v175[16]; // [rsp+F0h] [rbp-60h] BYREF
  void (__fastcall *v176)(_BYTE *, _BYTE *, __int64); // [rsp+100h] [rbp-50h]
  void (__fastcall *v177)(_BYTE *, unsigned __int8 **); // [rsp+108h] [rbp-48h]
  __int64 v178; // [rsp+110h] [rbp-40h]

  v13 = a1[1];
  v14 = *(__int64 ***)a2;
  v15 = *(unsigned __int8 **)v13;
  v168 = v15;
  if ( v15 )
    sub_1623A60((__int64)&v168, (__int64)v15, 2);
  v16 = (__m128)_mm_loadu_si128((const __m128i *)(v13 + 48));
  v169 = *(_QWORD *)(v13 + 8);
  v170 = *(__int64 **)(v13 + 16);
  v171 = *(_QWORD *)(v13 + 24);
  v172 = *(_QWORD *)(v13 + 32);
  v17 = *(_DWORD *)(v13 + 40);
  v176 = 0;
  v173 = v17;
  v18 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(v13 + 80);
  v174 = v16;
  if ( v18 )
  {
    v18(v175, v13 + 64, 2);
    v177 = *(void (__fastcall **)(_BYTE *, unsigned __int8 **))(v13 + 88);
    v176 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(v13 + 80);
  }
  v19 = *(unsigned __int8 **)(a3 + 48);
  v178 = *(_QWORD *)(v13 + 96);
  v20 = *(_QWORD *)(a3 + 40);
  v166[0] = v19;
  v169 = v20;
  v170 = (__int64 *)(a3 + 24);
  if ( v19 )
  {
    sub_1623A60((__int64)v166, (__int64)v19, 2);
    v21 = v168;
    if ( !v168 )
      goto LABEL_8;
  }
  else
  {
    v21 = v168;
    if ( !v168 )
      goto LABEL_10;
  }
  sub_161E7C0((__int64)&v168, (__int64)v21);
LABEL_8:
  v168 = v166[0];
  if ( v166[0] )
    sub_1623210((__int64)v166, v166[0], (__int64)&v168);
LABEL_10:
  v22 = *(_QWORD *)(a3 + 56);
  v23 = (__int64)v14[3];
  v24 = *(unsigned __int8 *)(v22 + 8);
  if ( (unsigned __int8)v24 <= 0xFu && (v27 = 35454, _bittest64(&v27, v24))
    || ((unsigned int)(v24 - 13) <= 1 || (_DWORD)v24 == 16) && sub_16435F0(*(_QWORD *)(a3 + 56), 0) )
  {
    if ( (v28 = *(unsigned __int8 *)(v23 + 8), (unsigned __int8)v28 <= 0xFu) && (v29 = 35454, _bittest64(&v29, v28))
      || ((unsigned int)(v28 - 13) <= 1 || (_DWORD)v28 == 16) && sub_16435F0(v23, 0) )
    {
      v30 = sub_15A9FE0(a1[333], v22);
      v31 = sub_15A9FE0(a1[333], v23);
      if ( v30 <= v31 && ((v32 = *(_QWORD *)(a3 + 8)) != 0 && !*(_QWORD *)(v32 + 8) || v30 != v31) )
      {
        v152 = a1[333];
        v33 = sub_15A9FE0(v152, v22);
        v34 = v152;
        v35 = v22;
        v36 = 1;
        v37 = v33;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v35 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v82 = *(_QWORD *)(v35 + 32);
              v35 = *(_QWORD *)(v35 + 24);
              v36 *= v82;
              continue;
            case 1:
              v38 = 16;
              goto LABEL_38;
            case 2:
              v38 = 32;
              goto LABEL_38;
            case 3:
            case 9:
              v38 = 64;
              goto LABEL_38;
            case 4:
              v38 = 80;
              goto LABEL_38;
            case 5:
            case 6:
              v38 = 128;
              goto LABEL_38;
            case 7:
              v157 = v36;
              v84 = sub_15A9520(v34, 0);
              v36 = v157;
              v38 = (unsigned int)(8 * v84);
              goto LABEL_38;
            case 0xB:
              v38 = *(_DWORD *)(v35 + 8) >> 8;
              goto LABEL_38;
            case 0xD:
              v158 = v36;
              v85 = (_QWORD *)sub_15A9930(v34, v35);
              v36 = v158;
              v38 = 8LL * *v85;
              goto LABEL_38;
            case 0xE:
              v119 = v36;
              v130 = v152;
              v124 = *(_QWORD *)(v35 + 24);
              v159 = *(_QWORD *)(v35 + 32);
              v140 = (unsigned int)sub_15A9FE0(v34, v124);
              v86 = sub_127FA20(v130, v124);
              v36 = v119;
              v38 = 8 * v159 * v140 * ((v140 + ((unsigned __int64)(v86 + 7) >> 3) - 1) / v140);
              goto LABEL_38;
            case 0xF:
              v156 = v36;
              v83 = sub_15A9520(v34, *(_DWORD *)(v35 + 8) >> 8);
              v36 = v156;
              v38 = (unsigned int)(8 * v83);
LABEL_38:
              v139 = a1[333];
              v153 = v37 * ((v37 + ((unsigned __int64)(v36 * v38 + 7) >> 3) - 1) / v37);
              v39 = sub_15A9FE0(v139, v23);
              v40 = v139;
              v41 = v23;
              v42 = 1;
              v43 = v39;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v41 + 8) )
                {
                  case 1:
                    v44 = 16;
                    goto LABEL_41;
                  case 2:
                    v44 = 32;
                    goto LABEL_41;
                  case 3:
                  case 9:
                    v44 = 64;
                    goto LABEL_41;
                  case 4:
                    v44 = 80;
                    goto LABEL_41;
                  case 5:
                  case 6:
                    v44 = 128;
                    goto LABEL_41;
                  case 7:
                    v141 = v42;
                    v88 = sub_15A9520(v40, 0);
                    v42 = v141;
                    v44 = (unsigned int)(8 * v88);
                    goto LABEL_41;
                  case 0xB:
                    v44 = *(_DWORD *)(v41 + 8) >> 8;
                    goto LABEL_41;
                  case 0xD:
                    v142 = v42;
                    v89 = (_QWORD *)sub_15A9930(v40, v41);
                    v42 = v142;
                    v44 = 8LL * *v89;
                    goto LABEL_41;
                  case 0xE:
                    v120 = v42;
                    v131 = v139;
                    v125 = *(_QWORD *)(v41 + 24);
                    v144 = *(_QWORD *)(v41 + 32);
                    v91 = sub_15A9FE0(v40, v125);
                    v92 = v131;
                    v93 = v125;
                    v94 = 1;
                    v42 = v120;
                    v95 = v91;
                    while ( 2 )
                    {
                      switch ( *(_BYTE *)(v93 + 8) )
                      {
                        case 1:
                          v107 = 16;
                          goto LABEL_134;
                        case 2:
                          v107 = 32;
                          goto LABEL_134;
                        case 3:
                        case 9:
                          v107 = 64;
                          goto LABEL_134;
                        case 4:
                          v107 = 80;
                          goto LABEL_134;
                        case 5:
                        case 6:
                          v107 = 128;
                          goto LABEL_134;
                        case 7:
                          v108 = 0;
                          v127 = v95;
                          v136 = v94;
                          goto LABEL_137;
                        case 0xB:
                          v107 = *(_DWORD *)(v93 + 8) >> 8;
                          goto LABEL_134;
                        case 0xD:
                          v128 = v95;
                          v137 = v94;
                          v110 = (_QWORD *)sub_15A9930(v92, v93);
                          v94 = v137;
                          v95 = v128;
                          v42 = v120;
                          v107 = 8LL * *v110;
                          goto LABEL_134;
                        case 0xE:
                          v113 = v120;
                          v114 = v95;
                          v116 = v94;
                          v118 = *(_QWORD *)(v93 + 24);
                          v123 = v131;
                          v138 = *(_QWORD *)(v93 + 32);
                          v129 = (unsigned int)sub_15A9FE0(v92, v118);
                          v112 = sub_127FA20(v123, v118);
                          v94 = v116;
                          v95 = v114;
                          v42 = v113;
                          v107 = 8 * v129 * v138 * ((v129 + ((unsigned __int64)(v112 + 7) >> 3) - 1) / v129);
                          goto LABEL_134;
                        case 0xF:
                          v127 = v95;
                          v136 = v94;
                          v108 = *(_DWORD *)(v93 + 8) >> 8;
LABEL_137:
                          v109 = sub_15A9520(v92, v108);
                          v94 = v136;
                          v95 = v127;
                          v42 = v120;
                          v107 = (unsigned int)(8 * v109);
LABEL_134:
                          v44 = 8 * v144 * v95 * ((v95 + ((unsigned __int64)(v107 * v94 + 7) >> 3) - 1) / v95);
                          goto LABEL_41;
                        case 0x10:
                          v111 = *(_QWORD *)(v93 + 32);
                          v93 = *(_QWORD *)(v93 + 24);
                          v94 *= v111;
                          continue;
                        default:
                          goto LABEL_36;
                      }
                    }
                  case 0xF:
                    v143 = v42;
                    v90 = sub_15A9520(v40, *(_DWORD *)(v41 + 8) >> 8);
                    v42 = v143;
                    v44 = (unsigned int)(8 * v90);
LABEL_41:
                    v45 = (v43 + ((unsigned __int64)(v44 * v42 + 7) >> 3) - 1) / v43 * v43;
                    if ( v45 && v153 )
                    {
                      v46 = a1[333];
                      v47 = 1;
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v22 + 8) )
                        {
                          case 1:
                            v48 = 16;
                            goto LABEL_48;
                          case 2:
                            v48 = 32;
                            goto LABEL_48;
                          case 3:
                          case 9:
                            v48 = 64;
                            goto LABEL_48;
                          case 4:
                            v48 = 80;
                            goto LABEL_48;
                          case 5:
                          case 6:
                            v48 = 128;
                            goto LABEL_48;
                          case 7:
                            v145 = v47;
                            v96 = 0;
                            goto LABEL_116;
                          case 0xB:
                            v48 = *(_DWORD *)(v22 + 8) >> 8;
                            goto LABEL_48;
                          case 0xD:
                            v147 = v47;
                            v100 = (_QWORD *)sub_15A9930(v46, v22);
                            v46 = a1[333];
                            v47 = v147;
                            v48 = 8LL * *v100;
                            goto LABEL_48;
                          case 0xE:
                            v121 = v47;
                            v126 = *(_QWORD *)(v22 + 24);
                            v146 = *(_QWORD *)(v22 + 32);
                            v98 = (unsigned int)sub_15A9FE0(v46, v126);
                            v99 = sub_127FA20(v46, v126);
                            v46 = a1[333];
                            v47 = v121;
                            v48 = 8 * v98 * v146 * ((v98 + ((unsigned __int64)(v99 + 7) >> 3) - 1) / v98);
                            goto LABEL_48;
                          case 0xF:
                            v145 = v47;
                            v96 = *(_DWORD *)(v22 + 8) >> 8;
LABEL_116:
                            v97 = sub_15A9520(v46, v96);
                            v46 = a1[333];
                            v47 = v145;
                            v48 = (unsigned int)(8 * v97);
LABEL_48:
                            v49 = v23;
                            v50 = 1;
                            while ( 2 )
                            {
                              switch ( *(_BYTE *)(v49 + 8) )
                              {
                                case 1:
                                  v51 = 16;
                                  goto LABEL_51;
                                case 2:
                                  v51 = 32;
                                  goto LABEL_51;
                                case 3:
                                case 9:
                                  v51 = 64;
                                  goto LABEL_51;
                                case 4:
                                  v51 = 80;
                                  goto LABEL_51;
                                case 5:
                                case 6:
                                  v51 = 128;
                                  goto LABEL_51;
                                case 7:
                                  v135 = v50;
                                  v151 = v47;
                                  v106 = sub_15A9520(v46, 0);
                                  v47 = v151;
                                  v50 = v135;
                                  v51 = (unsigned int)(8 * v106);
                                  goto LABEL_51;
                                case 0xB:
                                  v51 = *(_DWORD *)(v49 + 8) >> 8;
                                  goto LABEL_51;
                                case 0xD:
                                  v134 = v50;
                                  v150 = v47;
                                  v104 = (_QWORD *)sub_15A9930(v46, v49);
                                  v47 = v150;
                                  v50 = v134;
                                  v51 = 8LL * *v104;
                                  goto LABEL_51;
                                case 0xE:
                                  v115 = v50;
                                  v117 = v47;
                                  v122 = *(_QWORD *)(v49 + 24);
                                  v149 = *(_QWORD *)(v49 + 32);
                                  v133 = (unsigned int)sub_15A9FE0(v46, v122);
                                  v103 = sub_127FA20(v46, v122);
                                  v47 = v117;
                                  v50 = v115;
                                  v51 = 8 * v149 * v133 * ((v133 + ((unsigned __int64)(v103 + 7) >> 3) - 1) / v133);
                                  goto LABEL_51;
                                case 0xF:
                                  v132 = v50;
                                  v148 = v47;
                                  v102 = sub_15A9520(v46, *(_DWORD *)(v49 + 8) >> 8);
                                  v47 = v148;
                                  v50 = v132;
                                  v51 = (unsigned int)(8 * v102);
LABEL_51:
                                  v52 = *(_QWORD *)(a3 + 8);
                                  if ( (!v52 || *(_QWORD *)(v52 + 8))
                                    && (unsigned __int64)(v51 * v50 + 7) >> 3 < (unsigned __int64)(v48 * v47 + 7) >> 3 )
                                  {
                                    goto LABEL_14;
                                  }
                                  v53 = sub_1749A40(*(_QWORD *)(a3 - 24), &v161, &v162);
                                  v54 = v153 * v161 / v45;
                                  if ( v153 * v161 % v45 )
                                    goto LABEL_14;
                                  v55 = v162 * v153;
                                  if ( v162 * v153 % v45 )
                                    goto LABEL_14;
                                  if ( (_DWORD)v54 != 1 )
                                  {
                                    v56 = sub_15A0680(**(_QWORD **)(a3 - 24), (unsigned int)v54, 0);
                                    v167 = 257;
                                    if ( *(_BYTE *)(v56 + 16) > 0x10u || *(_BYTE *)(v53 + 16) > 0x10u )
                                    {
                                      v53 = (__int64)sub_170A2B0(
                                                       (__int64)&v168,
                                                       15,
                                                       (__int64 *)v56,
                                                       v53,
                                                       (__int64 *)v166,
                                                       0,
                                                       0);
                                    }
                                    else
                                    {
                                      v53 = sub_15A2C20((__int64 *)v56, v53, 0, 0, *(double *)v16.m128_u64, a5, a6);
                                      v57 = sub_14DBA30(v53, v178, 0);
                                      if ( v57 )
                                        v53 = v57;
                                    }
                                    v55 = v162 * v153;
                                  }
                                  if ( v45 <= v55 )
                                  {
                                    v58 = sub_15A0680(**(_QWORD **)(a3 - 24), v55 / v45, 1u);
                                    v167 = 257;
                                    if ( *(_BYTE *)(v53 + 16) > 0x10u || *(_BYTE *)(v58 + 16) > 0x10u )
                                    {
                                      v53 = (__int64)sub_170A2B0(
                                                       (__int64)&v168,
                                                       11,
                                                       (__int64 *)v53,
                                                       v58,
                                                       (__int64 *)v166,
                                                       0,
                                                       0);
                                    }
                                    else
                                    {
                                      v53 = sub_15A2B30((__int64 *)v53, v58, 0, 0, *(double *)v16.m128_u64, a5, a6);
                                      v59 = sub_14DBA30(v53, v178, 0);
                                      if ( v59 )
                                        v53 = v59;
                                    }
                                  }
                                  v165 = 257;
                                  v154 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v169 + 56) + 40LL)) + 4);
                                  v167 = 257;
                                  v60 = sub_1648A60(64, 1u);
                                  v61 = (__int64)v60;
                                  if ( v60 )
                                    sub_15F8BC0((__int64)v60, (_QWORD *)v23, v154, v53, (__int64)v166, 0);
                                  if ( v169 )
                                  {
                                    v62 = v170;
                                    sub_157E9D0(v169 + 40, v61);
                                    v63 = *(_QWORD *)(v61 + 24);
                                    v64 = *v62;
                                    *(_QWORD *)(v61 + 32) = v62;
                                    v64 &= 0xFFFFFFFFFFFFFFF8LL;
                                    *(_QWORD *)(v61 + 24) = v64 | v63 & 7;
                                    *(_QWORD *)(v64 + 8) = v61 + 24;
                                    *v62 = *v62 & 7 | (v61 + 24);
                                  }
                                  sub_164B780(v61, v164);
                                  v163 = (unsigned __int8 *)v61;
                                  if ( !v176 )
                                    sub_4263D6(v61, v164, v65);
                                  v177(v175, &v163);
                                  if ( v168 )
                                  {
                                    v163 = v168;
                                    sub_1623A60((__int64)&v163, (__int64)v168, 2);
                                    v66 = *(_QWORD *)(v61 + 48);
                                    if ( v66 )
                                      sub_161E7C0(v61 + 48, v66);
                                    v67 = v163;
                                    *(_QWORD *)(v61 + 48) = v163;
                                    if ( v67 )
                                      sub_1623210((__int64)&v163, v67, v61 + 48);
                                  }
                                  sub_15F8A20(v61, (unsigned int)(1 << *(_WORD *)(a3 + 18)) >> 1);
                                  sub_164B7C0(v61, a3);
                                  v68 = *(_WORD *)(v61 + 18) & 0x7FDF;
                                  if ( (*(_BYTE *)(a3 + 18) & 0x20) != 0 )
                                    v68 = *(_WORD *)(v61 + 18) & 0x7FDF | 0x20;
                                  *(_WORD *)(v61 + 18) = v68 | *(_WORD *)(v61 + 18) & 0x8000;
                                  v69 = *(_QWORD *)(a3 + 8);
                                  if ( !v69 || *(_QWORD *)(v69 + 8) )
                                  {
                                    v70 = *(__int64 ***)a3;
                                    v166[0] = "tmpcast";
                                    v167 = 259;
                                    v71 = sub_1708970((__int64)&v168, 47, v61, v70, (__int64 *)v166);
                                    v72 = *(_QWORD *)(a3 + 8);
                                    v155 = (__int64)v71;
                                    if ( v72 )
                                    {
                                      v73 = *a1;
                                      do
                                      {
                                        v74 = sub_1648700(v72);
                                        sub_170B990(v73, (__int64)v74);
                                        v72 = *(_QWORD *)(v72 + 8);
                                      }
                                      while ( v72 );
                                      if ( a3 == v155 )
                                        v155 = sub_1599EF0(*(__int64 ***)a3);
                                      sub_164D160(a3, v155, v16, a5, a6, a7, v75, v76, a10, a11);
                                    }
                                  }
                                  v77 = *(_QWORD *)(a2 + 8);
                                  v25 = a2;
                                  if ( !v77 )
                                    goto LABEL_14;
                                  v78 = *a1;
                                  do
                                  {
                                    v79 = sub_1648700(v77);
                                    sub_170B990(v78, (__int64)v79);
                                    v77 = *(_QWORD *)(v77 + 8);
                                  }
                                  while ( v77 );
                                  if ( a2 == v61 )
                                    v61 = sub_1599EF0(*(__int64 ***)a2);
                                  sub_164D160(a2, v61, v16, a5, a6, a7, v80, v81, a10, a11);
                                  break;
                                case 0x10:
                                  v101 = *(_QWORD *)(v49 + 32);
                                  v49 = *(_QWORD *)(v49 + 24);
                                  v50 *= v101;
                                  continue;
                                default:
                                  goto LABEL_36;
                              }
                              goto LABEL_15;
                            }
                          case 0x10:
                            v105 = *(_QWORD *)(v22 + 32);
                            v22 = *(_QWORD *)(v22 + 24);
                            v47 *= v105;
                            continue;
                          default:
                            goto LABEL_36;
                        }
                      }
                    }
                    goto LABEL_14;
                  case 0x10:
                    v87 = *(_QWORD *)(v41 + 32);
                    v41 = *(_QWORD *)(v41 + 24);
                    v42 *= v87;
                    continue;
                  default:
LABEL_36:
                    BUG();
                }
              }
          }
        }
      }
    }
  }
LABEL_14:
  v25 = 0;
LABEL_15:
  if ( v176 )
    v176(v175, v175, 3);
  if ( v168 )
    sub_161E7C0((__int64)&v168, (__int64)v168);
  return v25;
}
