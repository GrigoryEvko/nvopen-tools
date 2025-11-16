// Function: sub_3830830
// Address: 0x3830830
//
unsigned __int8 *__fastcall sub_3830830(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 *v3; // r15
  __int64 v4; // rsi
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned int v15; // ebx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rsi
  unsigned int *v21; // rbx
  unsigned int *v22; // r14
  unsigned int *v23; // r15
  __int64 v24; // rdx
  unsigned __int64 v25; // r12
  __int64 v26; // rdx
  unsigned __int16 v27; // r14
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int16 *v31; // rdx
  int v32; // eax
  __int64 v33; // rdx
  unsigned __int16 v34; // r12
  __int64 v35; // rdx
  unsigned __int16 *v36; // rdx
  int v37; // eax
  __int64 v38; // rdx
  unsigned __int16 v39; // ax
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rax
  unsigned __int16 v43; // r12
  __int64 v44; // rax
  _QWORD *v45; // rax
  _QWORD *v46; // rcx
  _QWORD *i; // rdx
  unsigned int v48; // edi
  unsigned __int64 *v49; // rax
  __int64 v50; // rsi
  unsigned __int64 v51; // r12
  unsigned __int64 v52; // r13
  __int64 v53; // rax
  __int64 v54; // rbx
  __int64 v55; // rax
  __int64 v56; // rbx
  __int64 v57; // rsi
  __int64 v58; // rax
  __int64 v59; // rcx
  __int64 v60; // r8
  unsigned __int16 *v61; // rbx
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rbx
  __int64 v65; // rbx
  __int128 v66; // rax
  __int64 v67; // r9
  unsigned __int8 *v68; // rax
  unsigned int v69; // edx
  __int64 v70; // rax
  unsigned __int8 **v71; // rax
  int v72; // edx
  unsigned __int8 *v73; // rax
  unsigned __int8 *v74; // r12
  unsigned __int16 *v76; // rdx
  int v77; // eax
  __int64 v78; // rdx
  __int16 v79; // di
  __int64 v80; // rdx
  unsigned int v81; // r14d
  unsigned __int64 v82; // r13
  __int64 v83; // rdx
  __int64 v84; // rax
  unsigned __int64 v85; // rax
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // rdx
  __int64 v89; // rax
  unsigned __int64 v90; // rbx
  unsigned __int64 v91; // rdx
  unsigned __int64 *v92; // rax
  __int64 v93; // rax
  __int64 v94; // rsi
  unsigned __int64 *v95; // rax
  unsigned __int64 v96; // r12
  __int64 v97; // rbx
  __int64 v98; // rax
  unsigned __int16 v99; // dx
  __int64 v100; // r8
  __int64 v101; // rax
  __int64 v102; // r9
  __int64 v103; // rdx
  __int64 v104; // rcx
  __int64 v105; // r13
  unsigned __int16 v106; // ax
  __int64 v107; // rdx
  __int64 v108; // rcx
  __int64 v109; // r8
  __int64 v110; // rax
  __int64 v111; // rdi
  bool v112; // al
  __int64 v113; // rcx
  __int16 v114; // ax
  __int64 v115; // rdx
  __int64 v116; // rdx
  __int64 v117; // r13
  int v118; // esi
  __int64 v119; // rax
  __int64 v120; // r9
  __int64 v121; // rsi
  unsigned int v122; // edx
  __int64 v123; // rdx
  __int64 v124; // rdx
  __int64 v125; // rdx
  unsigned int v126; // edx
  __int64 v127; // rdx
  _QWORD *v128; // rdx
  __int64 v129; // r13
  _QWORD *v130; // r12
  _QWORD *v131; // r15
  int v132; // esi
  int v133; // eax
  __int64 v134; // r9
  __int64 v135; // r8
  __int64 v136; // rdx
  unsigned int v137; // ebx
  unsigned __int8 *v138; // rax
  unsigned int v139; // edx
  __int64 v140; // rdx
  __int128 v141; // [rsp-20h] [rbp-290h]
  __int128 v142; // [rsp-10h] [rbp-280h]
  __int128 v143; // [rsp-10h] [rbp-280h]
  __int64 v144; // [rsp+8h] [rbp-268h]
  __int16 v145; // [rsp+12h] [rbp-25Eh]
  int v146; // [rsp+1Ch] [rbp-254h]
  __int64 v148; // [rsp+30h] [rbp-240h]
  __int64 v149; // [rsp+38h] [rbp-238h]
  __int64 v150; // [rsp+40h] [rbp-230h]
  unsigned int v151; // [rsp+40h] [rbp-230h]
  __int64 v152; // [rsp+48h] [rbp-228h]
  __int16 v153; // [rsp+4Ah] [rbp-226h]
  __int64 v154; // [rsp+50h] [rbp-220h]
  __int64 v155; // [rsp+50h] [rbp-220h]
  unsigned int v156; // [rsp+58h] [rbp-218h]
  __int16 v157; // [rsp+5Ah] [rbp-216h]
  int v158; // [rsp+60h] [rbp-210h]
  unsigned int v159; // [rsp+68h] [rbp-208h]
  __int64 v160; // [rsp+68h] [rbp-208h]
  __int64 v161; // [rsp+68h] [rbp-208h]
  __int64 v162; // [rsp+70h] [rbp-200h]
  unsigned __int64 v163; // [rsp+70h] [rbp-200h]
  unsigned int *v164; // [rsp+78h] [rbp-1F8h]
  unsigned int v165; // [rsp+78h] [rbp-1F8h]
  _QWORD *v166; // [rsp+78h] [rbp-1F8h]
  __int64 v167; // [rsp+78h] [rbp-1F8h]
  unsigned int v168; // [rsp+78h] [rbp-1F8h]
  unsigned __int8 *v169; // [rsp+80h] [rbp-1F0h]
  __int64 v170; // [rsp+A0h] [rbp-1D0h] BYREF
  int v171; // [rsp+A8h] [rbp-1C8h]
  __int64 v172; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v173; // [rsp+B8h] [rbp-1B8h]
  __int64 v174; // [rsp+C0h] [rbp-1B0h] BYREF
  __int64 v175; // [rsp+C8h] [rbp-1A8h]
  __int64 v176; // [rsp+D0h] [rbp-1A0h] BYREF
  __int64 v177; // [rsp+D8h] [rbp-198h]
  unsigned __int16 v178; // [rsp+E0h] [rbp-190h] BYREF
  __int64 v179; // [rsp+E8h] [rbp-188h]
  unsigned __int16 v180; // [rsp+F0h] [rbp-180h] BYREF
  __int64 v181; // [rsp+F8h] [rbp-178h]
  unsigned __int16 v182; // [rsp+100h] [rbp-170h] BYREF
  __int64 v183; // [rsp+108h] [rbp-168h]
  unsigned __int16 v184; // [rsp+110h] [rbp-160h] BYREF
  __int64 v185; // [rsp+118h] [rbp-158h]
  unsigned __int16 v186; // [rsp+120h] [rbp-150h] BYREF
  __int64 v187; // [rsp+128h] [rbp-148h]
  unsigned __int16 v188; // [rsp+130h] [rbp-140h] BYREF
  __int64 v189; // [rsp+138h] [rbp-138h]
  __int64 v190; // [rsp+140h] [rbp-130h]
  __int64 v191; // [rsp+148h] [rbp-128h]
  unsigned __int64 v192; // [rsp+150h] [rbp-120h]
  __int64 v193; // [rsp+158h] [rbp-118h]
  __int16 v194; // [rsp+160h] [rbp-110h] BYREF
  __int64 v195; // [rsp+168h] [rbp-108h]
  __int64 v196; // [rsp+170h] [rbp-100h]
  __int64 v197; // [rsp+178h] [rbp-F8h]
  unsigned __int64 v198; // [rsp+180h] [rbp-F0h]
  __int64 v199; // [rsp+188h] [rbp-E8h]
  __int16 v200; // [rsp+190h] [rbp-E0h] BYREF
  __int64 v201; // [rsp+198h] [rbp-D8h]
  _QWORD *v202; // [rsp+1B0h] [rbp-C0h] BYREF
  __int64 v203; // [rsp+1B8h] [rbp-B8h]
  _QWORD v204[22]; // [rsp+1C0h] [rbp-B0h] BYREF

  v3 = a1;
  v4 = *(_QWORD *)(a2 + 80);
  v170 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v170, v4, 1);
  v5 = *a1;
  v171 = *(_DWORD *)(a2 + 72);
  v6 = *(_QWORD *)(a2 + 48);
  v7 = *(_QWORD *)(v6 + 8);
  LOWORD(v172) = *(_WORD *)v6;
  v8 = a1[1];
  v173 = v7;
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v5 + 592LL);
  if ( v9 == sub_2D56A50 )
  {
    v10 = v5;
    sub_2FE6CC0((__int64)&v202, v5, *(_QWORD *)(v8 + 64), v172, v173);
    LOWORD(v14) = v203;
    LOWORD(v174) = v203;
    v175 = v204[0];
  }
  else
  {
    v10 = *(_QWORD *)(v8 + 64);
    LODWORD(v14) = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v9)(v5, v10, (unsigned int)v172);
    LODWORD(v174) = v14;
    v175 = v140;
  }
  v159 = *(_DWORD *)(a2 + 64);
  if ( (_WORD)v14 )
  {
    v149 = 0;
    LODWORD(v14) = (unsigned __int16)v14 - 1;
    v15 = word_4456340[(int)v14];
    LOWORD(v14) = word_4456580[(int)v14];
  }
  else
  {
    v15 = sub_3007240((__int64)&v174);
    v14 = sub_3009970((__int64)&v174, v10, v16, v17, v18);
    v150 = v14;
    v149 = v19;
  }
  v20 = v150;
  LOWORD(v20) = v14;
  v151 = v20;
  if ( (_WORD)v172 )
  {
    if ( (unsigned __int16)(v172 - 176) <= 0x34u )
      goto LABEL_9;
LABEL_32:
    v42 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
    v43 = *(_WORD *)v42;
    v44 = *(_QWORD *)(v42 + 8);
    LOWORD(v202) = v43;
    v203 = v44;
    if ( v43 )
    {
      if ( (unsigned __int16)(v43 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      }
      v146 = word_4456340[v43 - 1];
    }
    else
    {
      if ( sub_3007100((__int64)&v202) )
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
      v146 = sub_3007130((__int64)&v202, v20);
    }
    v45 = v204;
    v46 = v204;
    v202 = v204;
    v203 = 0x800000000LL;
    if ( v15 )
    {
      if ( v15 > 8uLL )
      {
        sub_C8D5F0((__int64)&v202, v204, v15, 0x10u, v12, v13);
        v46 = v202;
        v45 = &v202[2 * (unsigned int)v203];
      }
      for ( i = &v46[2 * v15]; i != v45; v45 += 2 )
      {
        if ( v45 )
        {
          *v45 = 0;
          *((_DWORD *)v45 + 2) = 0;
        }
      }
      LODWORD(v203) = v15;
    }
    if ( v159 )
    {
      v158 = 0;
      v148 = 0;
      v144 = 40LL * v159;
      do
      {
        HIWORD(v48) = v145;
        v49 = (unsigned __int64 *)(*(_QWORD *)(a2 + 40) + v148);
        v50 = *v49;
        v51 = *v49;
        v52 = v49[1];
        v53 = *((unsigned int *)v49 + 2);
        v160 = v50;
        v165 = v53;
        v54 = v53;
        v55 = *(_QWORD *)(v50 + 48);
        v56 = 16 * v54;
        v57 = *v3;
        v58 = v56 + v55;
        LOWORD(v48) = *(_WORD *)v58;
        sub_2FE6CC0((__int64)&v200, *v3, *(_QWORD *)(v3[1] + 64), v48, *(_QWORD *)(v58 + 8));
        v13 = v165;
        if ( (_BYTE)v200 == 1 )
        {
          v57 = v51;
          v160 = sub_37AE0F0((__int64)v3, v51, v52);
          v13 = v126;
          v56 = 16LL * v126;
        }
        v61 = (unsigned __int16 *)(*(_QWORD *)(v160 + 48) + v56);
        LODWORD(v62) = *v61;
        v63 = *((_QWORD *)v61 + 1);
        v200 = v62;
        v201 = v63;
        if ( (_WORD)v62 )
        {
          v152 = 0;
          LOWORD(v62) = word_4456580[(int)v62 - 1];
        }
        else
        {
          v168 = v13;
          v62 = sub_3009970((__int64)&v200, v57, v63, v59, v60);
          v13 = v168;
          v162 = v62;
          v152 = v123;
        }
        v64 = v162;
        v155 = (unsigned int)v13;
        LOWORD(v64) = v62;
        v162 = v64;
        v65 = 0;
        if ( v146 )
        {
          do
          {
            v166 = (_QWORD *)v3[1];
            *(_QWORD *)&v66 = sub_3400EE0((__int64)v166, v65, (__int64)&v170, 0, a3);
            v52 = v155 | v52 & 0xFFFFFFFF00000000LL;
            *((_QWORD *)&v141 + 1) = v52;
            *(_QWORD *)&v141 = v160;
            v68 = sub_3406EB0(v166, 0x9Eu, (__int64)&v170, (unsigned int)v162, v152, v67, v141, v66);
            v169 = sub_33FAFB0(v3[1], (__int64)v68, v69, (__int64)&v170, v151, v149, a3);
            v70 = (unsigned int)(v65++ + v158);
            v71 = (unsigned __int8 **)&v202[2 * v70];
            *v71 = v169;
            *((_DWORD *)v71 + 2) = v72;
          }
          while ( v146 != v65 );
        }
        v148 += 40;
        v158 += v146;
      }
      while ( v144 != v148 );
    }
    *((_QWORD *)&v142 + 1) = (unsigned int)v203;
    *(_QWORD *)&v142 = v202;
    v73 = sub_33FC220((_QWORD *)v3[1], 156, (__int64)&v170, v174, v175, v13, v142);
    goto LABEL_54;
  }
  if ( !sub_3007100((__int64)&v172) )
    goto LABEL_32;
LABEL_9:
  v21 = *(unsigned int **)(a2 + 40);
  v22 = &v21[10 * *(unsigned int *)(a2 + 64)];
  if ( v21 == v22 || v22 == v21 + 10 )
    goto LABEL_60;
  v23 = v21 + 10;
  v164 = &v21[10 * *(unsigned int *)(a2 + 64)];
  do
  {
    v31 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v21 + 48LL) + 16LL * v21[2]);
    v32 = *v31;
    v33 = *((_QWORD *)v31 + 1);
    LOWORD(v202) = v32;
    v203 = v33;
    if ( (_WORD)v32 )
    {
      v34 = word_4456580[v32 - 1];
      v35 = 0;
    }
    else
    {
      v34 = sub_3009970((__int64)&v202, v20, v33, v11, v12);
    }
    v183 = v35;
    v182 = v34;
    v36 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v23 + 48LL) + 16LL * v23[2]);
    v37 = *v36;
    v38 = *((_QWORD *)v36 + 1);
    LOWORD(v202) = v37;
    v203 = v38;
    if ( (_WORD)v37 )
    {
      v39 = word_4456580[v37 - 1];
      v40 = 0;
    }
    else
    {
      v39 = sub_3009970((__int64)&v202, v20, v38, v11, v12);
      v34 = v182;
    }
    v184 = v39;
    v185 = v40;
    if ( v34 )
    {
      if ( (unsigned __int16)(v34 - 17) > 0xD3u )
        goto LABEL_13;
      v34 = word_4456580[v34 - 1];
      v24 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v182) )
      {
LABEL_13:
        v24 = v183;
        goto LABEL_14;
      }
      v34 = sub_3009970((__int64)&v182, v20, v41, v11, v12);
    }
LABEL_14:
    v188 = v34;
    v189 = v24;
    if ( v34 )
    {
      if ( v34 == 1 || (unsigned __int16)(v34 - 504) <= 7u )
LABEL_132:
        BUG();
      v25 = *(_QWORD *)&byte_444C4A0[16 * v34 - 16];
    }
    else
    {
      v190 = sub_3007260((__int64)&v188);
      v25 = v190;
      v191 = v26;
    }
    v27 = v184;
    if ( v184 )
    {
      if ( (unsigned __int16)(v184 - 17) > 0xD3u )
        goto LABEL_18;
      v27 = word_4456580[v184 - 1];
      v28 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v184) )
      {
LABEL_18:
        v28 = v185;
        goto LABEL_19;
      }
      v27 = sub_3009970((__int64)&v184, v20, v115, v11, v12);
      v28 = v116;
    }
LABEL_19:
    v186 = v27;
    v187 = v28;
    if ( v27 )
    {
      if ( v27 == 1 || (unsigned __int16)(v27 - 504) <= 7u )
        goto LABEL_132;
      v29 = *(_QWORD *)&byte_444C4A0[16 * v27 - 16];
    }
    else
    {
      v29 = sub_3007260((__int64)&v186);
      v192 = v29;
      v193 = v30;
    }
    if ( v25 < v29 )
      v21 = v23;
    v23 += 10;
  }
  while ( v164 != v23 );
  v3 = a1;
LABEL_60:
  v76 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v21 + 48LL) + 16LL * v21[2]);
  v77 = *v76;
  v78 = *((_QWORD *)v76 + 1);
  LOWORD(v202) = v77;
  v203 = v78;
  if ( (_WORD)v77 )
  {
    v79 = word_4456580[v77 - 1];
    v80 = 0;
  }
  else
  {
    v79 = sub_3009970((__int64)&v202, v20, v78, v11, v12);
  }
  LOWORD(v176) = v79;
  v202 = v204;
  v203 = 0x800000000LL;
  v177 = v80;
  if ( v159 )
  {
    HIWORD(v81) = v157;
    v167 = 0;
    v161 = 40LL * v159;
    while ( 1 )
    {
      v94 = *v3;
      v95 = (unsigned __int64 *)(*(_QWORD *)(a2 + 40) + v167);
      v96 = *v95;
      v163 = v95[1];
      v97 = *((unsigned int *)v95 + 2);
      v98 = *(_QWORD *)(*v95 + 48) + 16 * v97;
      v99 = *(_WORD *)v98;
      v100 = *(_QWORD *)(v98 + 8);
      v101 = v3[1];
      LOWORD(v81) = v99;
      v178 = v99;
      v102 = *(_QWORD *)(v101 + 64);
      v179 = v100;
      sub_2FE6CC0((__int64)&v200, v94, v102, v81, v100);
      if ( (_BYTE)v200 == 1 )
      {
        v94 = v96;
        v96 = sub_37AE0F0((__int64)v3, v96, v163);
        v97 = (unsigned int)v103;
      }
      if ( v178 )
      {
        v105 = 0;
        v106 = word_4456580[v178 - 1];
      }
      else
      {
        v106 = sub_3009970((__int64)&v178, v94, v103, v104, v86);
        v105 = v124;
      }
      v180 = v106;
      v181 = v105;
      if ( v106 )
      {
        if ( (unsigned __int16)(v106 - 17) > 0xD3u )
        {
          v200 = v106;
          v201 = v105;
          goto LABEL_66;
        }
        v106 = word_4456580[v106 - 1];
        v125 = 0;
      }
      else
      {
        if ( !sub_30070B0((__int64)&v180) )
        {
          v201 = v105;
          v200 = 0;
LABEL_84:
          v110 = sub_3007260((__int64)&v200);
          v111 = v83;
          LODWORD(v83) = (unsigned __int16)v176;
          v196 = v110;
          v82 = v110;
          v197 = v111;
          if ( (_WORD)v176 )
            goto LABEL_69;
          goto LABEL_85;
        }
        v106 = sub_3009970((__int64)&v180, v94, v107, v108, v109);
      }
      v200 = v106;
      v201 = v125;
      if ( !v106 )
        goto LABEL_84;
LABEL_66:
      if ( v106 == 1 || (unsigned __int16)(v106 - 504) <= 7u )
        goto LABEL_132;
      v82 = *(_QWORD *)&byte_444C4A0[16 * v106 - 16];
      LODWORD(v83) = (unsigned __int16)v176;
      if ( (_WORD)v176 )
      {
LABEL_69:
        if ( (unsigned __int16)(v83 - 17) <= 0xD3u )
        {
          LOWORD(v83) = word_4456580[(unsigned __int16)v83 - 1];
          v84 = 0;
          goto LABEL_71;
        }
        goto LABEL_70;
      }
LABEL_85:
      v156 = v83;
      v112 = sub_30070B0((__int64)&v176);
      LOWORD(v83) = v156;
      if ( v112 )
      {
        v114 = sub_3009970((__int64)&v176, v94, v156, v113, v86);
        v86 = v83;
        LOWORD(v83) = v114;
        v84 = v86;
        goto LABEL_71;
      }
LABEL_70:
      v84 = v177;
LABEL_71:
      v194 = v83;
      v195 = v84;
      if ( (_WORD)v83 )
      {
        if ( (_WORD)v83 == 1 || (unsigned __int16)(v83 - 504) <= 7u )
          goto LABEL_132;
        v85 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v83 - 16];
      }
      else
      {
        v85 = sub_3007260((__int64)&v194);
        v198 = v85;
        v199 = v88;
      }
      if ( v85 > v82 )
      {
        v117 = v3[1];
        if ( v178 )
        {
          v118 = word_4456340[v178 - 1];
          if ( (unsigned __int16)(v178 - 176) > 0x34u )
            LOWORD(v119) = sub_2D43050(v176, v118);
          else
            LOWORD(v119) = sub_2D43AD0(v176, v118);
          v120 = 0;
        }
        else
        {
          v119 = sub_3009490(&v178, v176, v177);
          v154 = v119;
          v120 = v127;
        }
        v121 = v154;
        LOWORD(v121) = v119;
        v154 = v121;
        v163 = v97 | v163 & 0xFFFFFFFF00000000LL;
        v96 = (unsigned __int64)sub_33FAFB0(v117, v96, v97, (__int64)&v170, (unsigned int)v121, v120, a3);
        v97 = v122;
      }
      v89 = (unsigned int)v203;
      v90 = v163 & 0xFFFFFFFF00000000LL | v97;
      v91 = (unsigned int)v203 + 1LL;
      if ( v91 > HIDWORD(v203) )
      {
        sub_C8D5F0((__int64)&v202, v204, v91, 0x10u, v86, v87);
        v89 = (unsigned int)v203;
      }
      v92 = &v202[2 * v89];
      v167 += 40;
      v92[1] = v90;
      *v92 = v96;
      v93 = (unsigned int)(v203 + 1);
      LODWORD(v203) = v203 + 1;
      if ( v161 == v167 )
      {
        v128 = v202;
        v79 = v176;
        goto LABEL_119;
      }
    }
  }
  v128 = v204;
  v93 = 0;
LABEL_119:
  v129 = v93;
  v130 = v128;
  v131 = (_QWORD *)v3[1];
  if ( (_WORD)v172 )
  {
    v132 = word_4456340[(unsigned __int16)v172 - 1];
    if ( (unsigned __int16)(v172 - 176) <= 0x34u )
      LOWORD(v133) = sub_2D43AD0(v79, v132);
    else
      LOWORD(v133) = sub_2D43050(v79, v132);
    v135 = 0;
  }
  else
  {
    v133 = sub_3009490((unsigned __int16 *)&v172, v176, v177);
    v153 = HIWORD(v133);
    v135 = v136;
  }
  HIWORD(v137) = v153;
  *((_QWORD *)&v143 + 1) = v129;
  *(_QWORD *)&v143 = v130;
  LOWORD(v137) = v133;
  v138 = sub_33FC220(v131, 159, (__int64)&v170, v137, v135, v134, v143);
  v73 = sub_33FAFB0((__int64)v131, (__int64)v138, v139, (__int64)&v170, (unsigned int)v174, v175, a3);
LABEL_54:
  v74 = v73;
  if ( v202 != v204 )
    _libc_free((unsigned __int64)v202);
  if ( v170 )
    sub_B91220((__int64)&v170, v170);
  return v74;
}
