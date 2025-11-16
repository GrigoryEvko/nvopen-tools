// Function: sub_287D630
// Address: 0x287d630
//
__int64 __fastcall sub_287D630(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  unsigned int v6; // ebx
  unsigned int v7; // r15d
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 *v19; // rax
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // eax
  unsigned int v26; // r14d
  __int64 v27; // rbx
  unsigned __int64 v28; // rsi
  __int64 v29; // r12
  __int64 *v30; // r8
  int v31; // edi
  unsigned int v32; // ecx
  __int64 *v33; // rdx
  __int64 v34; // r10
  __int64 v35; // rax
  __int64 *v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rdi
  _QWORD *v39; // rdi
  size_t v40; // r13
  const char *v41; // r8
  __int64 v42; // rcx
  __int64 v43; // rax
  unsigned int v44; // eax
  unsigned int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  char v50; // al
  unsigned __int64 v51; // rsi
  char *v52; // rax
  char *v53; // r8
  char *v54; // rdi
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // rax
  _DWORD *v58; // rdi
  __int64 v59; // rcx
  __int64 v60; // rdx
  char *v61; // rax
  char *v62; // r8
  char *v63; // rdi
  __int64 v64; // rcx
  __int64 v65; // rdx
  char *v66; // rax
  __int64 v67; // rax
  _QWORD *v68; // rdi
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  void *v72; // r13
  void *v73; // r12
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 **v76; // rdx
  __int64 v77; // rcx
  void **v78; // rax
  int v79; // eax
  void **v80; // rax
  const char *v81; // rax
  unsigned __int64 v82; // rdx
  __int64 v83; // rax
  int v84; // edx
  int v85; // r9d
  __int64 *v86; // r12
  __int64 v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  unsigned __int64 v98; // rcx
  _QWORD *v99; // rax
  _QWORD *v100; // r13
  _QWORD *v101; // rdx
  __int64 **v102; // rsi
  __int64 *v103; // rax
  unsigned __int64 v104; // rax
  unsigned __int64 v105; // rax
  __int64 v106; // rdi
  __int64 v107; // rax
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // rax
  __int64 v111; // r13
  unsigned int v112; // r14d
  __int64 v113; // r15
  int v114; // r13d
  __int64 v115; // rax
  __int64 v116; // rdx
  __int64 v117; // rcx
  __int64 v118; // r8
  __int64 v119; // r9
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // r8
  __int64 v124; // r9
  __int64 v125; // rax
  __int64 v126; // r8
  __int64 v127; // rax
  unsigned __int64 v128; // rax
  int v129; // [rsp+0h] [rbp-460h]
  __int64 v130; // [rsp+8h] [rbp-458h]
  int v131; // [rsp+10h] [rbp-450h]
  __int64 v132; // [rsp+18h] [rbp-448h]
  __int64 v133; // [rsp+20h] [rbp-440h]
  unsigned int v134; // [rsp+20h] [rbp-440h]
  __int64 v135; // [rsp+28h] [rbp-438h]
  unsigned int v136; // [rsp+28h] [rbp-438h]
  __int64 v137; // [rsp+38h] [rbp-428h]
  unsigned __int16 v138; // [rsp+40h] [rbp-420h]
  __int64 v139; // [rsp+48h] [rbp-418h]
  unsigned __int8 v140; // [rsp+50h] [rbp-410h]
  __int64 v141; // [rsp+58h] [rbp-408h]
  __int64 v142; // [rsp+68h] [rbp-3F8h]
  __int64 v143; // [rsp+70h] [rbp-3F0h]
  __int64 v146; // [rsp+90h] [rbp-3D0h]
  int v147; // [rsp+98h] [rbp-3C8h]
  char v148; // [rsp+9Fh] [rbp-3C1h]
  unsigned __int16 v149; // [rsp+B8h] [rbp-3A8h]
  _QWORD *v150; // [rsp+B8h] [rbp-3A8h]
  _QWORD *v151; // [rsp+B8h] [rbp-3A8h]
  _QWORD *v152; // [rsp+B8h] [rbp-3A8h]
  char *v153; // [rsp+B8h] [rbp-3A8h]
  const char *v154; // [rsp+B8h] [rbp-3A8h]
  _QWORD *v155; // [rsp+B8h] [rbp-3A8h]
  unsigned __int8 v156; // [rsp+B8h] [rbp-3A8h]
  unsigned __int16 v157; // [rsp+B8h] [rbp-3A8h]
  unsigned int v158; // [rsp+C0h] [rbp-3A0h]
  unsigned __int16 v159; // [rsp+C4h] [rbp-39Ch]
  unsigned __int16 v160; // [rsp+C6h] [rbp-39Ah]
  __int64 v161; // [rsp+C8h] [rbp-398h]
  __int64 *v162; // [rsp+D0h] [rbp-390h]
  __int64 v163; // [rsp+E0h] [rbp-380h] BYREF
  __int64 v164; // [rsp+E8h] [rbp-378h] BYREF
  _QWORD v165[2]; // [rsp+F0h] [rbp-370h] BYREF
  __int64 v166; // [rsp+100h] [rbp-360h]
  __int64 v167; // [rsp+108h] [rbp-358h]
  __int64 v168; // [rsp+110h] [rbp-350h]
  __int64 v169; // [rsp+118h] [rbp-348h]
  __int64 v170; // [rsp+120h] [rbp-340h]
  unsigned __int64 v171; // [rsp+128h] [rbp-338h]
  __int64 v172[2]; // [rsp+130h] [rbp-330h] BYREF
  __int64 *v173; // [rsp+140h] [rbp-320h]
  _QWORD *v174; // [rsp+150h] [rbp-310h] BYREF
  size_t v175; // [rsp+158h] [rbp-308h]
  _QWORD v176[2]; // [rsp+160h] [rbp-300h] BYREF
  const char *v177; // [rsp+170h] [rbp-2F0h] BYREF
  __int64 v178; // [rsp+178h] [rbp-2E8h]
  const char *v179; // [rsp+180h] [rbp-2E0h]
  __int64 v180; // [rsp+188h] [rbp-2D8h]
  _QWORD v181[6]; // [rsp+190h] [rbp-2D0h] BYREF
  __int64 v182; // [rsp+1C0h] [rbp-2A0h] BYREF
  int v183; // [rsp+1C8h] [rbp-298h]
  __int64 v184; // [rsp+1D4h] [rbp-28Ch]
  __int64 v185; // [rsp+200h] [rbp-260h] BYREF
  int v186; // [rsp+208h] [rbp-258h]
  __int64 v187; // [rsp+214h] [rbp-24Ch]
  _DWORD v188[5]; // [rsp+240h] [rbp-220h] BYREF
  unsigned int v189; // [rsp+254h] [rbp-20Ch]
  unsigned int v190; // [rsp+268h] [rbp-1F8h]
  char v191; // [rsp+26Dh] [rbp-1F3h]
  unsigned __int8 v192; // [rsp+26Eh] [rbp-1F2h]
  char v193; // [rsp+270h] [rbp-1F0h]
  unsigned __int8 v194; // [rsp+272h] [rbp-1EEh]
  char v195; // [rsp+273h] [rbp-1EDh]
  unsigned int v196; // [rsp+274h] [rbp-1ECh]
  __int64 v197; // [rsp+290h] [rbp-1D0h] BYREF
  __int64 v198; // [rsp+298h] [rbp-1C8h]
  __int64 *v199; // [rsp+2A0h] [rbp-1C0h] BYREF
  unsigned int v200; // [rsp+2A8h] [rbp-1B8h]
  _BYTE *v201; // [rsp+2E0h] [rbp-180h] BYREF
  __int64 v202; // [rsp+2E8h] [rbp-178h]
  _BYTE v203[32]; // [rsp+2F0h] [rbp-170h] BYREF
  size_t v204; // [rsp+310h] [rbp-150h] BYREF
  void **v205; // [rsp+318h] [rbp-148h]
  __int64 v206; // [rsp+320h] [rbp-140h]
  int v207; // [rsp+328h] [rbp-138h]
  char v208; // [rsp+32Ch] [rbp-134h]
  _BYTE v209[16]; // [rsp+330h] [rbp-130h] BYREF
  __int64 v210; // [rsp+340h] [rbp-120h] BYREF
  void **v211; // [rsp+348h] [rbp-118h]
  unsigned int v212; // [rsp+354h] [rbp-10Ch]
  int v213; // [rsp+358h] [rbp-108h]
  char v214; // [rsp+35Ch] [rbp-104h]
  char v215[256]; // [rsp+360h] [rbp-100h] BYREF

  v11 = *a5;
  v12 = **(_QWORD **)(**(_QWORD **)(a3 + 8) + 32LL);
  v181[1] = a5[4];
  v181[0] = v11;
  v13 = *(_QWORD *)(v12 + 72);
  v181[2] = a5[3];
  v181[3] = v13;
  sub_1049690(v172, v13);
  v158 = *a2;
  v17 = a5[1];
  v165[1] = *(unsigned int *)(a3 + 16);
  v141 = v17;
  v161 = a5[6];
  v162 = (__int64 *)a5[4];
  v143 = a5[3];
  v142 = a5[2];
  v165[0] = *(_QWORD *)(a3 + 8);
  v18 = *(_QWORD *)v165[0];
  v197 = 0;
  v146 = v18;
  v19 = (unsigned __int64 *)&v199;
  v198 = 1;
  do
  {
    *v19 = -4096;
    v19 += 2;
  }
  while ( v19 != (unsigned __int64 *)&v201 );
  v20 = (unsigned __int64)&v197;
  v201 = v203;
  v202 = 0x400000000LL;
  sub_F76FB0(v165, (__int64)&v197, (__int64)&v201, v14, v15, v16);
  v25 = v202;
  v148 = 0;
  if ( (_DWORD)v202 )
  {
    v26 = v6;
    v27 = v149;
    do
    {
      v28 = (unsigned __int64)v201;
      v29 = *(_QWORD *)&v201[8 * v25 - 8];
      if ( (v198 & 1) != 0 )
      {
        v30 = (__int64 *)&v199;
        v31 = 3;
      }
      else
      {
        v30 = v199;
        if ( !v200 )
          goto LABEL_9;
        v31 = v200 - 1;
      }
      v32 = v31 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v33 = &v30[2 * v32];
      v34 = *v33;
      if ( v29 == *v33 )
      {
LABEL_8:
        *v33 = -8192;
        ++HIDWORD(v198);
        v28 = (unsigned __int64)v201;
        LODWORD(v198) = (2 * ((unsigned int)v198 >> 1) - 2) | v198 & 1;
        v25 = v202;
      }
      else
      {
        v84 = 1;
        while ( v34 != -4096 )
        {
          v85 = v84 + 1;
          v32 = v31 & (v84 + v32);
          v33 = &v30[2 * v32];
          v34 = *v33;
          if ( v29 == *v33 )
            goto LABEL_8;
          v84 = v85;
        }
      }
LABEL_9:
      v35 = (unsigned int)(v25 - 1);
      v36 = (__int64 *)(v28 + 8 * v35 - 8);
      do
      {
        LODWORD(v202) = v35;
        if ( !(_DWORD)v35 )
          break;
        v37 = *v36;
        LODWORD(v35) = v35 - 1;
        --v36;
      }
      while ( !v37 );
      v38 = **(_QWORD **)(v29 + 32);
      if ( v38 && (*(_BYTE *)(v38 + 7) & 0x10) != 0 )
      {
        v81 = sub_BD5D20(v38);
        v41 = v81;
        v40 = v82;
        v174 = v176;
        if ( &v81[v82] && !v81 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v204 = v82;
        if ( v82 <= 0xF )
        {
          if ( v82 == 1 )
          {
            LOBYTE(v176[0]) = *v81;
            goto LABEL_16;
          }
          if ( !v82 )
            goto LABEL_16;
          v39 = v176;
        }
        else
        {
          v154 = v81;
          v83 = sub_22409D0((__int64)&v174, &v204, 0);
          v41 = v154;
          v174 = (_QWORD *)v83;
          v39 = (_QWORD *)v83;
          v176[0] = v204;
        }
      }
      else
      {
        v39 = v176;
        v204 = 14;
        v40 = 14;
        v41 = "<unnamed loop>";
        v174 = v176;
      }
      memcpy(v39, v41, v40);
      v40 = v204;
LABEL_16:
      BYTE1(v27) = 0;
      v175 = v40;
      v42 = v159;
      *((_BYTE *)v174 + v40) = 0;
      v43 = v160;
      BYTE4(v204) = 0;
      BYTE1(v42) = 0;
      BYTE1(v43) = 0;
      v160 = (unsigned __int8)v160;
      BYTE4(v185) = 0;
      BYTE4(v182) = 0;
      v159 = (unsigned __int8)v159;
      sub_2880090(v188, v29, v162, v161, 0, 0, v172, v158, v182, v185, v27, v42, v43, v204);
      v44 = v26;
      BYTE1(v44) = 0;
      v20 = (unsigned __int64)v162;
      v26 = v44;
      v45 = v7;
      BYTE1(v45) = 0;
      v7 = v45;
      v164 = sub_2A04E50(v29, v162, v161, v45, v26, 0);
      v50 = sub_F6E690(v29, (__int64)v162, v46, v47, v48, v49);
      if ( (v50 & 2) != 0 )
        goto LABEL_63;
      if ( (v50 & 5) != 0 )
        v195 = 1;
      v150 = sub_C52410();
      v51 = sub_C959E0();
      v52 = (char *)v150[2];
      v53 = (char *)(v150 + 1);
      if ( v52 )
      {
        v54 = (char *)(v150 + 1);
        do
        {
          while ( 1 )
          {
            v55 = *((_QWORD *)v52 + 2);
            v56 = *((_QWORD *)v52 + 3);
            if ( v51 <= *((_QWORD *)v52 + 4) )
              break;
            v52 = (char *)*((_QWORD *)v52 + 3);
            if ( !v56 )
              goto LABEL_24;
          }
          v54 = v52;
          v52 = (char *)*((_QWORD *)v52 + 2);
        }
        while ( v55 );
LABEL_24:
        if ( v54 != v53 && v51 >= *((_QWORD *)v54 + 4) )
          v53 = v54;
      }
      v151 = v53;
      if ( v53 != (char *)sub_C52410() + 8 )
      {
        v57 = v151[7];
        if ( v57 )
        {
          v58 = v151 + 6;
          do
          {
            while ( 1 )
            {
              v59 = *(_QWORD *)(v57 + 16);
              v60 = *(_QWORD *)(v57 + 24);
              if ( *(_DWORD *)(v57 + 32) >= dword_5001BE8 )
                break;
              v57 = *(_QWORD *)(v57 + 24);
              if ( !v60 )
                goto LABEL_33;
            }
            v58 = (_DWORD *)v57;
            v57 = *(_QWORD *)(v57 + 16);
          }
          while ( v59 );
LABEL_33:
          if ( v58 != (_DWORD *)(v151 + 6) && dword_5001BE8 >= v58[8] && (int)v58[9] > 0 )
            v195 = qword_5001C68;
        }
      }
      v152 = sub_C52410();
      v20 = sub_C959E0();
      v61 = (char *)v152[2];
      v62 = (char *)(v152 + 1);
      if ( v61 )
      {
        v63 = (char *)(v152 + 1);
        do
        {
          while ( 1 )
          {
            v64 = *((_QWORD *)v61 + 2);
            v65 = *((_QWORD *)v61 + 3);
            if ( v20 <= *((_QWORD *)v61 + 4) )
              break;
            v61 = (char *)*((_QWORD *)v61 + 3);
            if ( !v65 )
              goto LABEL_42;
          }
          v63 = v61;
          v61 = (char *)*((_QWORD *)v61 + 2);
        }
        while ( v64 );
LABEL_42:
        if ( v63 != v62 && v20 >= *((_QWORD *)v63 + 4) )
          v62 = v63;
      }
      v153 = v62;
      v66 = (char *)sub_C52410();
      v23 = (__int64)v153;
      if ( v153 != v66 + 8 )
      {
        v67 = *((_QWORD *)v153 + 7);
        v24 = (__int64)(v153 + 48);
        if ( v67 )
        {
          v20 = (unsigned int)dword_5001A28;
          v68 = v153 + 48;
          do
          {
            while ( 1 )
            {
              v22 = *(_QWORD *)(v67 + 16);
              v21 = *(_QWORD *)(v67 + 24);
              if ( *(_DWORD *)(v67 + 32) >= dword_5001A28 )
                break;
              v67 = *(_QWORD *)(v67 + 24);
              if ( !v21 )
                goto LABEL_51;
            }
            v68 = (_QWORD *)v67;
            v67 = *(_QWORD *)(v67 + 16);
          }
          while ( v22 );
LABEL_51:
          if ( v68 != (_QWORD *)v24 && dword_5001A28 >= *((_DWORD *)v68 + 8) )
          {
            v21 = *((unsigned int *)v68 + 9);
            if ( (int)v21 > 0 )
              v196 = qword_5001AA8;
          }
        }
      }
      if ( !v195 )
        goto LABEL_63;
      v23 = v196;
      if ( !v196 )
        goto LABEL_63;
      if ( (unsigned __int8)sub_287D4E0(v29, "llvm.loop.unroll.", 0x11u, v22, v196, v24) )
      {
        v20 = (unsigned __int64)"llvm.loop.unroll_and_jam.";
        if ( !(unsigned __int8)sub_287D4E0(v29, "llvm.loop.unroll_and_jam.", 0x19u, v69, v70, v71) )
          goto LABEL_63;
      }
      v20 = (unsigned __int64)v162;
      if ( !(unsigned __int8)sub_2A1AA10(v29, v162, v142, v181, v143) )
        goto LABEL_63;
      v204 = 0;
      v206 = 32;
      v205 = (void **)v209;
      v207 = 0;
      v208 = 1;
      sub_30AB790(v29, v141, &v204);
      v139 = **(_QWORD **)(v29 + 8);
      sub_2880BC0(&v182, v139, v161, &v204, v190);
      v20 = v29;
      sub_2880BC0(&v185, v29, v161, &v204, v190);
      if ( !(unsigned __int8)sub_2880E50(&v182) )
        goto LABEL_61;
      v140 = sub_2880E50(&v185);
      if ( !v140 )
        goto LABEL_61;
      if ( !v183 )
        v131 = v182;
      v21 = HIDWORD(v187) | HIDWORD(v184) | (unsigned int)v187 | (unsigned int)v184;
      if ( v187 | v184 )
        goto LABEL_61;
      v137 = sub_D49300(v29, v29, v21, v22, v23, v24);
      v92 = sub_D49300(v139, v29, v88, v89, v90, v91);
      v177 = "llvm.loop.unroll_and_jam.followup_all";
      v179 = "llvm.loop.unroll_and_jam.followup_remainder_inner";
      v130 = v92;
      v178 = 37;
      v180 = 49;
      v93 = sub_F6E0D0(v137, (__int64)&v177, 2, byte_3F871B3, 0);
      v167 = v94;
      v166 = v93;
      if ( (_BYTE)v94 )
        sub_D49440(v139, v166, v94, v95, v96, v97);
      v135 = sub_D47930(v29);
      v132 = sub_D47930(v139);
      v133 = v135;
      v136 = sub_DBA790((__int64)v162, v29, v135);
      v134 = sub_DE5E70(v162, v29, v133);
      v147 = sub_DBA790((__int64)v162, v139, v132);
      if ( !v186 )
        v129 = v185;
      v20 = v161;
      LOBYTE(v177) = 0;
      if ( (unsigned __int8)sub_28873F0(
                              v29,
                              v161,
                              v142,
                              v143,
                              v141,
                              (_DWORD)v162,
                              (__int64)&v204,
                              (__int64)v172,
                              v136,
                              0,
                              0,
                              v134,
                              (__int64)&v185,
                              (__int64)v188,
                              (__int64)&v164,
                              (__int64)&v177,
                              0)
        || (_BYTE)v177 )
      {
LABEL_138:
        v189 = 0;
LABEL_61:
        if ( !v208 )
          _libc_free((unsigned __int64)v205);
        goto LABEL_63;
      }
      v155 = sub_C52410();
      v98 = sub_C959E0();
      v99 = (_QWORD *)v155[2];
      v100 = v155 + 1;
      if ( v99 )
      {
        v101 = v155 + 1;
        do
        {
          v20 = v99[3];
          if ( v98 > v99[4] )
          {
            v99 = (_QWORD *)v99[3];
          }
          else
          {
            v101 = v99;
            v99 = (_QWORD *)v99[2];
          }
        }
        while ( v99 );
        if ( v101 != v100 && v98 >= v101[4] )
          v100 = v101;
      }
      if ( v100 == (_QWORD *)((char *)sub_C52410() + 8) )
        goto LABEL_149;
      v105 = v100[7];
      v20 = (unsigned __int64)(v100 + 6);
      if ( !v105 )
        goto LABEL_149;
      v22 = (unsigned int)dword_5001B08;
      v21 = (unsigned __int64)(v100 + 6);
      do
      {
        v23 = *(_QWORD *)(v105 + 16);
        if ( *(_DWORD *)(v105 + 32) < dword_5001B08 )
        {
          v105 = *(_QWORD *)(v105 + 24);
        }
        else
        {
          v21 = v105;
          v105 = *(_QWORD *)(v105 + 16);
        }
      }
      while ( v105 );
      if ( v20 == v21 || dword_5001B08 < *(_DWORD *)(v21 + 32) || (v22 = *(unsigned int *)(v21 + 36), (int)v22 <= 0) )
      {
LABEL_149:
        v156 = 0;
      }
      else
      {
        v20 = (unsigned int)qword_5001B88;
        v193 = 1;
        v189 = qword_5001B88;
        v156 = v192;
        if ( v192 )
        {
          v22 = v190;
          v23 = v188[0];
          if ( v190 + (unsigned int)qword_5001B88 * (unsigned __int64)(v129 - v190) < v188[0] )
          {
            v21 = v196;
            if ( v190 + (unsigned int)qword_5001B88 * (unsigned __int64)(v131 - v190) < v196 )
              goto LABEL_183;
          }
        }
        else
        {
          v156 = v140;
        }
      }
      v106 = sub_D49300(v29, v20, v21, v22, v23, v24);
      if ( v106 )
      {
        v107 = sub_2A11940(v106, "llvm.loop.unroll_and_jam.count", 30);
        if ( v107 )
        {
          v21 = *(unsigned __int8 *)(v107 - 16);
          if ( (v21 & 2) != 0 )
          {
            v108 = *(_QWORD *)(v107 - 32);
          }
          else
          {
            v21 = 8LL * (((unsigned __int8)v21 >> 2) & 0xF);
            v108 = v107 - v21 - 16;
          }
          v109 = *(_QWORD *)(*(_QWORD *)(v108 + 8) + 136LL);
          v22 = *(_QWORD *)(v109 + 24);
          if ( *(_DWORD *)(v109 + 32) > 0x40u )
            v22 = *(_QWORD *)v22;
          v20 = (unsigned int)v22;
          if ( (_DWORD)v22 )
          {
            v189 = v22;
            v191 = 1;
            v193 = 1;
            if ( v192 || (v21 = v134 % (unsigned int)v22) == 0 )
            {
              v22 = (unsigned int)v22;
              v23 = v188[0];
              if ( v190 + (unsigned int)v22 * (unsigned __int64)(v129 - v190) < v188[0] )
              {
                v21 = v196;
                v22 = (v131 - v190) * (unsigned __int64)(unsigned int)v22;
                if ( v22 + (unsigned __int64)v190 < v196 )
                  goto LABEL_220;
              }
            }
            v110 = sub_D49300(v29, (__int64)"llvm.loop.unroll_and_jam.enable", v21, v22, v23, v24);
            v20 = (unsigned __int64)"llvm.loop.unroll_and_jam.enable";
            v156 = v140;
            if ( !v110 )
            {
              v104 = (unsigned int)qword_50019C8;
              v20 = v189;
              v196 = qword_50019C8;
              if ( !v192 )
              {
LABEL_137:
                v22 = v190;
                v20 = v131 - v190;
                v21 = v190 + v20 * v189;
                if ( v104 <= v21 )
                  goto LABEL_138;
LABEL_219:
                v20 = v189;
              }
LABEL_220:
              v156 = v140;
              goto LABEL_183;
            }
LABEL_162:
            v110 = sub_2A11940(v110, "llvm.loop.unroll_and_jam.enable", 31);
            goto LABEL_163;
          }
        }
      }
      v110 = sub_D49300(v29, (__int64)"llvm.loop.unroll_and_jam.enable", v21, v22, v23, v24);
      v20 = (unsigned __int64)"llvm.loop.unroll_and_jam.enable";
      if ( v110 )
        goto LABEL_162;
LABEL_163:
      LOBYTE(v21) = v156 | (v110 != 0);
      if ( (_BYTE)v21 )
      {
        v104 = (unsigned int)qword_50019C8;
        v21 = v192;
        v196 = qword_50019C8;
        if ( !v192 )
          goto LABEL_137;
        if ( v156 )
          goto LABEL_219;
        v20 = v189;
        v22 = v189;
        if ( !v189 )
          goto LABEL_220;
LABEL_209:
        v127 = (unsigned int)v20;
        v20 = 0;
        v24 = v196;
        v23 = v131 - v190;
        v128 = v190 + v23 * v127;
        while ( v196 <= v128 )
        {
          v128 -= v23;
          v22 = (unsigned int)(v22 - 1);
          if ( !(_DWORD)v22 )
          {
            v189 = 0;
            goto LABEL_226;
          }
          v20 = v140;
        }
        if ( (_BYTE)v20 )
          v189 = v22;
LABEL_226:
        if ( (_BYTE)v21 )
          goto LABEL_219;
        goto LABEL_166;
      }
      if ( v192 )
      {
        v20 = v189;
        v22 = v189;
        if ( v189 )
          goto LABEL_209;
      }
      else
      {
        v22 = v189;
        v21 = v196;
        if ( v190 + v189 * (unsigned __int64)(v131 - v190) >= v196 )
          goto LABEL_138;
      }
LABEL_166:
      if ( v147 && (unsigned int)(v147 * v131) < v188[0] )
        goto LABEL_138;
      v21 = *(_QWORD *)(v139 + 32);
      if ( *(_QWORD *)(v139 + 40) - v21 != 8 )
        goto LABEL_138;
      v111 = *(_QWORD *)(*(_QWORD *)v21 + 56LL);
      v22 = *(_QWORD *)v21 + 48LL;
      if ( v111 == v22 )
        goto LABEL_138;
      v157 = v26;
      v138 = v7;
      v112 = 0;
      v113 = *(_QWORD *)v21 + 48LL;
      do
      {
        if ( !v111 )
          BUG();
        if ( *(_BYTE *)(v111 - 24) == 61 )
        {
          v20 = (unsigned __int64)sub_DDFBA0((__int64)v162, *(_QWORD *)(v111 - 56), (char *)v29);
          v112 -= !sub_DADE90((__int64)v162, v20, v29) - 1;
        }
        v111 = *(_QWORD *)(v111 + 8);
      }
      while ( v111 != v113 );
      v23 = v112;
      v7 = v138;
      v26 = v157;
      if ( !(_DWORD)v23 )
        goto LABEL_138;
      v156 = 0;
      v20 = v189;
LABEL_183:
      if ( (unsigned int)v20 <= 1 )
        goto LABEL_61;
      if ( v136 && v136 < (unsigned int)v20 )
      {
        v189 = v136;
        LODWORD(v20) = v136;
      }
      v163 = 0;
      v114 = sub_2A1CF00(
               v29,
               v20,
               v136,
               v134,
               v194,
               v143,
               (__int64)v162,
               v142,
               v141,
               v161,
               (__int64)v172,
               (__int64)&v163);
      if ( v163 )
      {
        v177 = "llvm.loop.unroll_and_jam.followup_all";
        v178 = 37;
        v179 = "llvm.loop.unroll_and_jam.followup_remainder_outer";
        v180 = 49;
        v115 = sub_F6E0D0(v137, (__int64)&v177, 2, byte_3F871B3, 0);
        v171 = v116;
        v170 = v115;
        if ( (_BYTE)v116 )
          sub_D49440(v163, v170, v116, v117, v118, v119);
      }
      v177 = "llvm.loop.unroll_and_jam.followup_all";
      v178 = 37;
      v179 = "llvm.loop.unroll_and_jam.followup_inner";
      v180 = 39;
      v120 = sub_F6E0D0(v137, (__int64)&v177, 2, byte_3F871B3, 0);
      v169 = v121;
      v168 = v120;
      if ( (_BYTE)v121 )
      {
        v20 = v168;
        sub_D49440(v139, v168, v121, v122, v123, v124);
      }
      else
      {
        v20 = v130;
        sub_D49440(v139, v130, v121, v122, v123, v124);
      }
      if ( v114 != 1 )
        goto LABEL_194;
      v20 = (unsigned __int64)&v177;
      v177 = "llvm.loop.unroll_and_jam.followup_all";
      v178 = 37;
      v179 = "llvm.loop.unroll_and_jam.followup_outer";
      v180 = 39;
      v125 = sub_F6E0D0(v137, (__int64)&v177, 2, byte_3F871B3, 0);
      v171 = v21;
      v170 = v125;
      if ( (_BYTE)v21 )
      {
        v20 = v170;
        sub_D49440(v29, v170, v21, v22, v126, v24);
        if ( !v208 )
          _libc_free((unsigned __int64)v205);
      }
      else
      {
LABEL_194:
        if ( v114 != 2 && v156 )
          sub_D4A9E0(v29);
        if ( !v208 )
          _libc_free((unsigned __int64)v205);
        v23 = (__int64)v174;
        if ( !v114 )
          goto LABEL_63;
        LOBYTE(v21) = v114 == 2 && v146 == v29;
        v148 = v21;
        if ( (_BYTE)v21 )
        {
          v20 = v29;
          sub_22D0060(*(_QWORD *)(a6 + 8), v29, (__int64)v174, v175);
          if ( v29 == *(_QWORD *)(a6 + 16) )
            *(_BYTE *)(a6 + 24) = 1;
          goto LABEL_63;
        }
      }
      v148 = v140;
LABEL_63:
      if ( v174 != v176 )
      {
        v20 = v176[0] + 1LL;
        j_j___libc_free_0((unsigned __int64)v174);
      }
      v25 = v202;
    }
    while ( (_DWORD)v202 );
  }
  if ( v201 != v203 )
    _libc_free((unsigned __int64)v201);
  if ( (v198 & 1) == 0 )
  {
    v20 = 16LL * v200;
    sub_C7D6A0((__int64)v199, v20, 8);
  }
  v72 = (void *)(a1 + 32);
  v73 = (void *)(a1 + 80);
  if ( !v148 )
  {
    *(_QWORD *)(a1 + 8) = v72;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v73;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    goto LABEL_102;
  }
  sub_22D0390((__int64)&v204, v20, v21, v22, v23, v24);
  if ( !v214 )
  {
    v103 = sub_C8CA60((__int64)&v210, (__int64)&unk_4F876D0);
    if ( v103 )
    {
      *v103 = -2;
      ++v210;
      v77 = v212;
      v79 = ++v213;
    }
    else
    {
      v77 = v212;
      v79 = v213;
    }
    goto LABEL_77;
  }
  v76 = (__int64 **)&v211[v212];
  v77 = v212;
  if ( v211 != (void **)v76 )
  {
    v78 = v211;
    while ( *v78 != &unk_4F876D0 )
    {
      if ( v76 == (__int64 **)++v78 )
        goto LABEL_121;
    }
    v76 = (__int64 **)v211[--v212];
    *v78 = v76;
    v77 = v212;
    ++v210;
    v79 = v213;
LABEL_77:
    if ( (_DWORD)v77 != v79 )
      goto LABEL_78;
    goto LABEL_122;
  }
LABEL_121:
  if ( v212 != v213 )
    goto LABEL_78;
LABEL_122:
  if ( v208 )
  {
    v80 = v205;
    v102 = (__int64 **)&v205[HIDWORD(v206)];
    v77 = HIDWORD(v206);
    v76 = (__int64 **)v205;
    if ( v205 != (void **)v102 )
    {
      while ( *v76 != &qword_4F82400 )
      {
        if ( v102 == ++v76 )
        {
LABEL_82:
          while ( *v80 != &unk_4F876D0 )
          {
            if ( ++v80 == (void **)v76 )
              goto LABEL_130;
          }
          goto LABEL_83;
        }
      }
      goto LABEL_83;
    }
    goto LABEL_130;
  }
  if ( sub_C8CA60((__int64)&v204, (__int64)&qword_4F82400) )
    goto LABEL_83;
LABEL_78:
  if ( !v208 )
  {
LABEL_132:
    sub_C8CC70((__int64)&v204, (__int64)&unk_4F876D0, (__int64)v76, v77, v74, v75);
    goto LABEL_83;
  }
  v80 = v205;
  v77 = HIDWORD(v206);
  v76 = (__int64 **)&v205[HIDWORD(v206)];
  if ( v76 != (__int64 **)v205 )
    goto LABEL_82;
LABEL_130:
  if ( (unsigned int)v77 >= (unsigned int)v206 )
    goto LABEL_132;
  HIDWORD(v206) = v77 + 1;
  *v76 = (__int64 *)&unk_4F876D0;
  ++v204;
LABEL_83:
  sub_C8CF70(a1, v72, 2, (__int64)v209, (__int64)&v204);
  sub_C8CF70(a1 + 48, v73, 2, (__int64)v215, (__int64)&v210);
  if ( !v214 )
    _libc_free((unsigned __int64)v211);
  if ( !v208 )
    _libc_free((unsigned __int64)v205);
LABEL_102:
  v86 = v173;
  if ( v173 )
  {
    sub_FDC110(v173);
    j_j___libc_free_0((unsigned __int64)v86);
  }
  return a1;
}
