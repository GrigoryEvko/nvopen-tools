// Function: sub_1C81350
// Address: 0x1c81350
//
void __fastcall sub_1C81350(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 **a4,
        __int64 a5,
        __int64 **a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        unsigned int a16,
        unsigned __int8 a17,
        unsigned __int8 a18,
        __int64 a19,
        __int64 a20)
{
  unsigned __int64 v23; // r14
  _QWORD *v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // r12
  unsigned __int64 v27; // r14
  unsigned __int8 *v28; // rsi
  __int64 v29; // rax
  _QWORD *v30; // rax
  double v31; // xmm4_8
  double v32; // xmm5_8
  _QWORD *v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 *v38; // r14
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // rsi
  unsigned __int8 *v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // r14
  int v48; // eax
  __int64 v49; // rax
  int v50; // edx
  __int64 v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rcx
  unsigned __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rcx
  _QWORD *v58; // rax
  __int64 v59; // r10
  __int64 v60; // rdx
  __int64 *v61; // r14
  __int64 v62; // rcx
  __int64 v63; // rax
  __int64 v64; // r10
  __int64 v65; // rsi
  __int64 v66; // r14
  unsigned __int8 *v67; // rsi
  _QWORD *v68; // rax
  _QWORD *v69; // r14
  unsigned __int64 v70; // rsi
  __int64 v71; // rax
  __int64 v72; // rsi
  __int64 v73; // rdx
  unsigned __int8 *v74; // rsi
  __int64 v75; // rax
  unsigned __int8 *v76; // rsi
  __int64 v77; // rdx
  __int64 v78; // rcx
  unsigned __int8 **v79; // r8
  __int64 v80; // r9
  __int64 v81; // r14
  int v82; // eax
  __int64 v83; // rax
  int v84; // edx
  __int64 v85; // rdx
  __int64 *v86; // rax
  __int64 v87; // rcx
  unsigned __int64 v88; // rdx
  __int64 v89; // rdx
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // rax
  __int64 v93; // r14
  _QWORD *v94; // rax
  _QWORD *v95; // rbx
  unsigned __int64 *v96; // r12
  __int64 v97; // rax
  unsigned __int64 v98; // rcx
  __int64 v99; // rsi
  unsigned __int8 *v100; // rsi
  __int64 v101; // rax
  __int64 v102; // rsi
  __int64 v103; // rax
  __int64 v104; // rsi
  __int64 v105; // rax
  __int64 *v106; // r15
  __int64 v107; // rcx
  __int64 v108; // rax
  __int64 v109; // rsi
  __int64 v110; // rdx
  unsigned __int8 *v111; // rsi
  __int64 v112; // rax
  unsigned __int8 *v113; // rsi
  int v114; // ebx
  __int64 v115; // rsi
  _BYTE *v116; // r14
  __int64 v117; // rax
  __int64 v118; // rax
  _QWORD *v119; // rax
  _QWORD *v120; // r13
  unsigned __int64 v121; // rsi
  __int64 v122; // rax
  __int64 v123; // rsi
  __int64 v124; // rdx
  unsigned __int8 *v125; // rsi
  int v126; // r8d
  int v127; // r9d
  __int64 v128; // rax
  int v129; // ebx
  __int64 v130; // r13
  __int64 v131; // rax
  __int64 v132; // r14
  _QWORD *v133; // rax
  _QWORD *v134; // r13
  unsigned __int64 v135; // rsi
  __int64 v136; // rax
  __int64 v137; // rsi
  __int64 v138; // rdx
  unsigned __int8 *v139; // rsi
  __int64 v140; // rdi
  __int64 v141; // rax
  __int64 *v142; // rbx
  __int64 v143; // rcx
  __int64 v144; // rax
  __int64 v145; // rbx
  __int64 v146; // rsi
  unsigned __int8 *v147; // rsi
  __int64 *v148; // r13
  __int64 v149; // rax
  __int64 v150; // rcx
  __int64 v151; // r13
  __int64 v152; // rsi
  unsigned __int8 *v153; // rsi
  __int64 *v154; // rbx
  __int64 v155; // rax
  __int64 v156; // rcx
  __int64 v157; // rsi
  unsigned __int8 *v158; // rsi
  __int64 v159; // [rsp+18h] [rbp-1A8h]
  __int64 v160; // [rsp+20h] [rbp-1A0h]
  __int64 v161; // [rsp+28h] [rbp-198h]
  __int64 v162; // [rsp+30h] [rbp-190h]
  __int64 v163; // [rsp+38h] [rbp-188h]
  __int64 v164; // [rsp+38h] [rbp-188h]
  unsigned __int64 v165; // [rsp+38h] [rbp-188h]
  __int64 v167; // [rsp+50h] [rbp-170h]
  __int64 v169; // [rsp+58h] [rbp-168h]
  _QWORD *v170; // [rsp+58h] [rbp-168h]
  __int64 v171; // [rsp+58h] [rbp-168h]
  __int64 v172; // [rsp+58h] [rbp-168h]
  __int64 v173; // [rsp+58h] [rbp-168h]
  __int64 v174; // [rsp+58h] [rbp-168h]
  unsigned __int64 *v175; // [rsp+58h] [rbp-168h]
  __int64 *v176; // [rsp+58h] [rbp-168h]
  __int64 v177; // [rsp+58h] [rbp-168h]
  unsigned __int64 *v178; // [rsp+58h] [rbp-168h]
  __int64 v179; // [rsp+58h] [rbp-168h]
  unsigned __int64 *v180; // [rsp+58h] [rbp-168h]
  unsigned __int8 *v181; // [rsp+68h] [rbp-158h] BYREF
  __int64 v182[2]; // [rsp+70h] [rbp-150h] BYREF
  __int16 v183; // [rsp+80h] [rbp-140h]
  __int64 v184[2]; // [rsp+90h] [rbp-130h] BYREF
  __int16 v185; // [rsp+A0h] [rbp-120h]
  unsigned __int8 *v186; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v187; // [rsp+B8h] [rbp-108h]
  __int64 *v188; // [rsp+C0h] [rbp-100h]
  __int64 v189; // [rsp+C8h] [rbp-F8h]
  __int64 v190; // [rsp+D0h] [rbp-F0h]
  int v191; // [rsp+D8h] [rbp-E8h]
  __int64 v192; // [rsp+E0h] [rbp-E0h]
  __int64 v193; // [rsp+E8h] [rbp-D8h]
  char *v194; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v195; // [rsp+108h] [rbp-B8h]
  _QWORD v196[3]; // [rsp+110h] [rbp-B0h] BYREF
  int v197; // [rsp+128h] [rbp-98h]
  __int64 v198; // [rsp+130h] [rbp-90h]
  __int64 v199; // [rsp+138h] [rbp-88h]

  if ( *(_BYTE *)(a15 + 16) != 13 )
    goto LABEL_5;
  v23 = *(_QWORD *)(a15 + 24);
  if ( *(_DWORD *)(a15 + 32) > 0x40u )
    v23 = **(_QWORD **)(a15 + 24);
  if ( (unsigned int)dword_4FBD560 < v23 )
  {
LABEL_5:
    v24 = (_QWORD *)a1[5];
    LOWORD(v196[0]) = 259;
    v194 = "split";
    v160 = (__int64)v24;
    v161 = sub_157FBF0(v24, a1 + 3, (__int64)&v194);
    v194 = "loadstoreloop";
    LOWORD(v196[0]) = 259;
    v25 = (_QWORD *)sub_22077B0(64);
    v26 = (__int64)v25;
    if ( v25 )
      sub_157FB60(v25, a19, (__int64)&v194, a20, v161);
    v27 = sub_157EBA0(v160);
    v186 = 0;
    v189 = sub_16498A0(v27);
    v190 = 0;
    v191 = 0;
    v192 = 0;
    v193 = 0;
    v187 = *(_QWORD *)(v27 + 40);
    v188 = (__int64 *)(v27 + 24);
    v28 = *(unsigned __int8 **)(v27 + 48);
    v194 = (char *)v28;
    if ( v28 )
    {
      sub_1623A60((__int64)&v194, (__int64)v28, 2);
      v186 = (unsigned __int8 *)v194;
      if ( v194 )
        sub_1623210((__int64)&v194, (unsigned __int8 *)v194, (__int64)&v186);
    }
    v185 = 257;
    if ( a4 != *(__int64 ***)a3 )
    {
      if ( *(_BYTE *)(a3 + 16) > 0x10u )
      {
        LOWORD(v196[0]) = 257;
        v105 = sub_15FDBD0(47, a3, (__int64)a4, (__int64)&v194, 0);
        a3 = v105;
        if ( v187 )
        {
          v106 = v188;
          sub_157E9D0(v187 + 40, v105);
          v107 = *v106;
          v108 = *(_QWORD *)(a3 + 24);
          *(_QWORD *)(a3 + 32) = v106;
          v107 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(a3 + 24) = v107 | v108 & 7;
          *(_QWORD *)(v107 + 8) = a3 + 24;
          *v106 = *v106 & 7 | (a3 + 24);
        }
        sub_164B780(a3, v184);
        if ( v186 )
        {
          v182[0] = (__int64)v186;
          sub_1623A60((__int64)v182, (__int64)v186, 2);
          v109 = *(_QWORD *)(a3 + 48);
          v110 = a3 + 48;
          if ( v109 )
          {
            sub_161E7C0(a3 + 48, v109);
            v110 = a3 + 48;
          }
          v111 = (unsigned __int8 *)v182[0];
          *(_QWORD *)(a3 + 48) = v182[0];
          if ( v111 )
            sub_1623210((__int64)v182, v111, v110);
        }
      }
      else
      {
        a3 = sub_15A46C0(47, (__int64 ***)a3, a4, 0);
      }
    }
    v185 = 257;
    if ( a6 != *(__int64 ***)a5 )
    {
      if ( *(_BYTE *)(a5 + 16) > 0x10u )
      {
        LOWORD(v196[0]) = 257;
        v141 = sub_15FDBD0(47, a5, (__int64)a6, (__int64)&v194, 0);
        a5 = v141;
        if ( v187 )
        {
          v142 = v188;
          sub_157E9D0(v187 + 40, v141);
          v143 = *v142;
          v144 = *(_QWORD *)(a5 + 24);
          *(_QWORD *)(a5 + 32) = v142;
          v143 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(a5 + 24) = v143 | v144 & 7;
          *(_QWORD *)(v143 + 8) = a5 + 24;
          *v142 = *v142 & 7 | (a5 + 24);
        }
        sub_164B780(a5, v184);
        if ( v186 )
        {
          v182[0] = (__int64)v186;
          sub_1623A60((__int64)v182, (__int64)v186, 2);
          v145 = a5 + 48;
          v146 = *(_QWORD *)(a5 + 48);
          if ( v146 )
            sub_161E7C0(v145, v146);
          v147 = (unsigned __int8 *)v182[0];
          *(_QWORD *)(a5 + 48) = v182[0];
          if ( v147 )
            sub_1623210((__int64)v182, v147, v145);
        }
      }
      else
      {
        a5 = sub_15A46C0(47, (__int64 ***)a5, a6, 0);
      }
    }
    LOWORD(v196[0]) = 257;
    v29 = sub_15A0680(*(_QWORD *)a15, 0, 0);
    v163 = sub_12AA0C0((__int64 *)&v186, 0x22u, (_BYTE *)a15, v29, (__int64)&v194);
    v30 = sub_1648A60(56, 3u);
    v33 = v30;
    if ( v30 )
      sub_15F83E0((__int64)v30, v26, v161, v163, 0);
    sub_1AA6530(v27, v33, a7, a8, a9, a10, v31, v32, a13, a14);
    v34 = sub_157E9C0(v26);
    v194 = 0;
    v196[1] = v34;
    v196[0] = v26 + 40;
    v196[2] = 0;
    v35 = *(_QWORD *)a15;
    v197 = 0;
    v198 = 0;
    v162 = v35;
    v183 = 257;
    v199 = 0;
    v195 = v26;
    v185 = 257;
    v36 = sub_1648B60(64);
    v37 = v36;
    if ( v36 )
    {
      v164 = v36;
      sub_15F1EA0(v36, v162, 53, 0, 0, 0);
      *(_DWORD *)(v37 + 56) = 0;
      sub_164B780(v37, v184);
      sub_1648880(v37, *(_DWORD *)(v37 + 56), 1);
    }
    else
    {
      v164 = 0;
    }
    if ( v195 )
    {
      v38 = (__int64 *)v196[0];
      sub_157E9D0(v195 + 40, v37);
      v39 = *(_QWORD *)(v37 + 24);
      v40 = *v38;
      *(_QWORD *)(v37 + 32) = v38;
      v40 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v37 + 24) = v40 | v39 & 7;
      *(_QWORD *)(v40 + 8) = v37 + 24;
      *v38 = *v38 & 7 | (v37 + 24);
    }
    sub_164B780(v164, v182);
    if ( v194 )
    {
      v181 = (unsigned __int8 *)v194;
      sub_1623A60((__int64)&v181, (__int64)v194, 2);
      v41 = *(_QWORD *)(v37 + 48);
      if ( v41 )
        sub_161E7C0(v37 + 48, v41);
      v42 = v181;
      *(_QWORD *)(v37 + 48) = v181;
      if ( v42 )
        sub_1623210((__int64)&v181, v42, v37 + 48);
    }
    v47 = sub_15A0680(v162, 0, 0);
    v48 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
    if ( v48 == *(_DWORD *)(v37 + 56) )
    {
      sub_15F55D0(v37, 0, v43, v44, v45, v46);
      v48 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
    }
    v49 = (v48 + 1) & 0xFFFFFFF;
    v50 = v49 | *(_DWORD *)(v37 + 20) & 0xF0000000;
    *(_DWORD *)(v37 + 20) = v50;
    if ( (v50 & 0x40000000) != 0 )
      v51 = *(_QWORD *)(v37 - 8);
    else
      v51 = v164 - 24 * v49;
    v52 = (__int64 *)(v51 + 24LL * (unsigned int)(v49 - 1));
    if ( *v52 )
    {
      v53 = v52[1];
      v54 = v52[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v54 = v53;
      if ( v53 )
        *(_QWORD *)(v53 + 16) = *(_QWORD *)(v53 + 16) & 3LL | v54;
    }
    *v52 = v47;
    if ( v47 )
    {
      v55 = *(_QWORD *)(v47 + 8);
      v52[1] = v55;
      if ( v55 )
        *(_QWORD *)(v55 + 16) = (unsigned __int64)(v52 + 1) | *(_QWORD *)(v55 + 16) & 3LL;
      v52[2] = (v47 + 8) | v52[2] & 3;
      *(_QWORD *)(v47 + 8) = v52;
    }
    v56 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v37 + 23) & 0x40) != 0 )
      v57 = *(_QWORD *)(v37 - 8);
    else
      v57 = v164 - 24 * v56;
    *(_QWORD *)(v57 + 8LL * (unsigned int)(v56 - 1) + 24LL * *(unsigned int *)(v37 + 56) + 8) = v160;
    v185 = 257;
    v183 = 257;
    v169 = sub_12815B0((__int64 *)&v194, 0, (_BYTE *)a3, v37, (__int64)v182);
    v58 = sub_1648A60(64, 1u);
    v59 = (__int64)v58;
    if ( v58 )
    {
      v60 = v169;
      v170 = v58;
      sub_15F9210((__int64)v58, *(_QWORD *)(*(_QWORD *)v60 + 24LL), v60, 0, a17, 0);
      v59 = (__int64)v170;
    }
    if ( v195 )
    {
      v61 = (__int64 *)v196[0];
      v171 = v59;
      sub_157E9D0(v195 + 40, v59);
      v59 = v171;
      v62 = *v61;
      v63 = *(_QWORD *)(v171 + 24);
      *(_QWORD *)(v171 + 32) = v61;
      v62 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v171 + 24) = v62 | v63 & 7;
      *(_QWORD *)(v62 + 8) = v171 + 24;
      *v61 = *v61 & 7 | (v171 + 24);
    }
    v172 = v59;
    sub_164B780(v59, v184);
    v64 = v172;
    if ( v194 )
    {
      v181 = (unsigned __int8 *)v194;
      sub_1623A60((__int64)&v181, (__int64)v194, 2);
      v64 = v172;
      v65 = *(_QWORD *)(v172 + 48);
      v66 = v172 + 48;
      if ( v65 )
      {
        sub_161E7C0(v172 + 48, v65);
        v64 = v172;
      }
      v67 = v181;
      *(_QWORD *)(v64 + 48) = v181;
      if ( v67 )
      {
        v173 = v64;
        sub_1623210((__int64)&v181, v67, v66);
        v64 = v173;
      }
    }
    v159 = v64;
    sub_15F8F50(v64, a16);
    v183 = 257;
    v174 = sub_12815B0((__int64 *)&v194, 0, (_BYTE *)a5, v37, (__int64)v182);
    v185 = 257;
    v68 = sub_1648A60(64, 2u);
    v69 = v68;
    if ( v68 )
      sub_15F9650((__int64)v68, v159, v174, a18, 0);
    if ( v195 )
    {
      v175 = (unsigned __int64 *)v196[0];
      sub_157E9D0(v195 + 40, (__int64)v69);
      v70 = *v175;
      v71 = v69[3] & 7LL;
      v69[4] = v175;
      v70 &= 0xFFFFFFFFFFFFFFF8LL;
      v69[3] = v70 | v71;
      *(_QWORD *)(v70 + 8) = v69 + 3;
      *v175 = *v175 & 7 | (unsigned __int64)(v69 + 3);
    }
    sub_164B780((__int64)v69, v184);
    if ( v194 )
    {
      v181 = (unsigned __int8 *)v194;
      sub_1623A60((__int64)&v181, (__int64)v194, 2);
      v72 = v69[6];
      v73 = (__int64)(v69 + 6);
      if ( v72 )
      {
        sub_161E7C0((__int64)(v69 + 6), v72);
        v73 = (__int64)(v69 + 6);
      }
      v74 = v181;
      v69[6] = v181;
      if ( v74 )
        sub_1623210((__int64)&v181, v74, v73);
    }
    sub_15F9450((__int64)v69, a16);
    v183 = 257;
    v75 = sub_15A0680(v162, 1, 0);
    v76 = (unsigned __int8 *)v75;
    if ( *(_BYTE *)(v37 + 16) > 0x10u || *(_BYTE *)(v75 + 16) > 0x10u )
    {
      v185 = 257;
      v101 = sub_15FB440(11, (__int64 *)v37, v75, (__int64)v184, 0);
      v81 = v101;
      if ( v195 )
      {
        v176 = (__int64 *)v196[0];
        sub_157E9D0(v195 + 40, v101);
        v102 = *v176;
        v103 = *(_QWORD *)(v81 + 24) & 7LL;
        *(_QWORD *)(v81 + 32) = v176;
        v102 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v81 + 24) = v102 | v103;
        *(_QWORD *)(v102 + 8) = v81 + 24;
        *v176 = *v176 & 7 | (v81 + 24);
      }
      sub_164B780(v81, v182);
      v76 = (unsigned __int8 *)v194;
      if ( v194 )
      {
        v181 = (unsigned __int8 *)v194;
        sub_1623A60((__int64)&v181, (__int64)v194, 2);
        v104 = *(_QWORD *)(v81 + 48);
        v79 = &v181;
        v77 = v81 + 48;
        if ( v104 )
        {
          sub_161E7C0(v81 + 48, v104);
          v79 = &v181;
          v77 = v81 + 48;
        }
        v76 = v181;
        *(_QWORD *)(v81 + 48) = v181;
        if ( v76 )
        {
          sub_1623210((__int64)&v181, v76, v77);
          v82 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
          if ( v82 != *(_DWORD *)(v37 + 56) )
            goto LABEL_62;
          goto LABEL_98;
        }
      }
    }
    else
    {
      v81 = sub_15A2B30((__int64 *)v37, v75, 0, 0, *(double *)a7.m128_u64, a8, a9);
    }
    v82 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
    if ( v82 != *(_DWORD *)(v37 + 56) )
    {
LABEL_62:
      v83 = (v82 + 1) & 0xFFFFFFF;
      v84 = v83 | *(_DWORD *)(v37 + 20) & 0xF0000000;
      *(_DWORD *)(v37 + 20) = v84;
      if ( (v84 & 0x40000000) != 0 )
        v85 = *(_QWORD *)(v37 - 8);
      else
        v85 = v164 - 24 * v83;
      v86 = (__int64 *)(v85 + 24LL * (unsigned int)(v83 - 1));
      if ( *v86 )
      {
        v87 = v86[1];
        v88 = v86[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v88 = v87;
        if ( v87 )
          *(_QWORD *)(v87 + 16) = *(_QWORD *)(v87 + 16) & 3LL | v88;
      }
      *v86 = v81;
      if ( v81 )
      {
        v89 = *(_QWORD *)(v81 + 8);
        v86[1] = v89;
        if ( v89 )
          *(_QWORD *)(v89 + 16) = (unsigned __int64)(v86 + 1) | *(_QWORD *)(v89 + 16) & 3LL;
        v86[2] = (v81 + 8) | v86[2] & 3;
        *(_QWORD *)(v81 + 8) = v86;
      }
      v90 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v37 + 23) & 0x40) != 0 )
        v91 = *(_QWORD *)(v37 - 8);
      else
        v91 = v164 - 24 * v90;
      *(_QWORD *)(v91 + 8LL * (unsigned int)(v90 - 1) + 24LL * *(unsigned int *)(v37 + 56) + 8) = v26;
      v183 = 257;
      v92 = sub_12AA0C0((__int64 *)&v194, 0x24u, (_BYTE *)v81, a15, (__int64)v182);
      v185 = 257;
      v93 = v92;
      v94 = sub_1648A60(56, 3u);
      v95 = v94;
      if ( v94 )
        sub_15F83E0((__int64)v94, v26, v161, v93, 0);
      if ( v195 )
      {
        v96 = (unsigned __int64 *)v196[0];
        sub_157E9D0(v195 + 40, (__int64)v95);
        v97 = v95[3];
        v98 = *v96;
        v95[4] = v96;
        v98 &= 0xFFFFFFFFFFFFFFF8LL;
        v95[3] = v98 | v97 & 7;
        *(_QWORD *)(v98 + 8) = v95 + 3;
        *v96 = *v96 & 7 | (unsigned __int64)(v95 + 3);
      }
      sub_164B780((__int64)v95, v184);
      if ( v194 )
      {
        v181 = (unsigned __int8 *)v194;
        sub_1623A60((__int64)&v181, (__int64)v194, 2);
        v99 = v95[6];
        if ( v99 )
          sub_161E7C0((__int64)(v95 + 6), v99);
        v100 = v181;
        v95[6] = v181;
        if ( v100 )
          sub_1623210((__int64)&v181, v100, (__int64)(v95 + 6));
        if ( v194 )
          sub_161E7C0((__int64)&v194, (__int64)v194);
      }
      goto LABEL_84;
    }
LABEL_98:
    sub_15F55D0(v37, (__int64)v76, v77, v78, (__int64)v79, v80);
    v82 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
    goto LABEL_62;
  }
  if ( !v23 )
    return;
  v186 = 0;
  v189 = sub_16498A0((__int64)a1);
  v112 = a1[5];
  v113 = (unsigned __int8 *)a1[6];
  v190 = 0;
  v191 = 0;
  v187 = v112;
  v188 = a1 + 3;
  v192 = 0;
  v193 = 0;
  v194 = (char *)v113;
  if ( v113 )
  {
    sub_1623A60((__int64)&v194, (__int64)v113, 2);
    v186 = (unsigned __int8 *)v194;
    if ( v194 )
      sub_1623210((__int64)&v194, (unsigned __int8 *)v194, (__int64)&v186);
  }
  v185 = 257;
  if ( a4 != *(__int64 ***)a3 )
  {
    if ( *(_BYTE *)(a3 + 16) > 0x10u )
    {
      LOWORD(v196[0]) = 257;
      a3 = sub_15FDBD0(47, a3, (__int64)a4, (__int64)&v194, 0);
      if ( v187 )
      {
        v148 = v188;
        sub_157E9D0(v187 + 40, a3);
        v149 = *(_QWORD *)(a3 + 24);
        v150 = *v148;
        *(_QWORD *)(a3 + 32) = v148;
        v150 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a3 + 24) = v150 | v149 & 7;
        *(_QWORD *)(v150 + 8) = a3 + 24;
        *v148 = *v148 & 7 | (a3 + 24);
      }
      sub_164B780(a3, v184);
      if ( v186 )
      {
        v182[0] = (__int64)v186;
        sub_1623A60((__int64)v182, (__int64)v186, 2);
        v151 = a3 + 48;
        v152 = *(_QWORD *)(a3 + 48);
        if ( v152 )
          sub_161E7C0(v151, v152);
        v153 = (unsigned __int8 *)v182[0];
        *(_QWORD *)(a3 + 48) = v182[0];
        if ( v153 )
          sub_1623210((__int64)v182, v153, v151);
      }
    }
    else
    {
      a3 = sub_15A46C0(47, (__int64 ***)a3, a4, 0);
    }
  }
  v185 = 257;
  if ( a6 != *(__int64 ***)a5 )
  {
    if ( *(_BYTE *)(a5 + 16) > 0x10u )
    {
      LOWORD(v196[0]) = 257;
      a5 = sub_15FDBD0(47, a5, (__int64)a6, (__int64)&v194, 0);
      if ( v187 )
      {
        v154 = v188;
        sub_157E9D0(v187 + 40, a5);
        v155 = *(_QWORD *)(a5 + 24);
        v156 = *v154;
        *(_QWORD *)(a5 + 32) = v154;
        v156 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a5 + 24) = v156 | v155 & 7;
        *(_QWORD *)(v156 + 8) = a5 + 24;
        *v154 = *v154 & 7 | (a5 + 24);
      }
      sub_164B780(a5, v184);
      if ( v186 )
      {
        v182[0] = (__int64)v186;
        sub_1623A60((__int64)v182, (__int64)v186, 2);
        v157 = *(_QWORD *)(a5 + 48);
        if ( v157 )
          sub_161E7C0(a5 + 48, v157);
        v158 = (unsigned __int8 *)v182[0];
        *(_QWORD *)(a5 + 48) = v182[0];
        if ( v158 )
          sub_1623210((__int64)v182, v158, a5 + 48);
      }
    }
    else
    {
      a5 = sub_15A46C0(47, (__int64 ***)a5, a6, 0);
    }
  }
  v165 = v23;
  v114 = 0;
  v115 = 0;
  v116 = (_BYTE *)a3;
  v167 = *(_QWORD *)a15;
  v194 = (char *)v196;
  v195 = 0x1000000000LL;
  do
  {
    v184[0] = (__int64)"src.memcpy.gep.unroll";
    v185 = 259;
    v117 = sub_15A0680(v167, v115, 0);
    v118 = sub_12815B0((__int64 *)&v186, a2, v116, v117, (__int64)v184);
    v185 = 257;
    v177 = v118;
    v119 = sub_1648A60(64, 1u);
    v120 = v119;
    if ( v119 )
      sub_15F9210((__int64)v119, *(_QWORD *)(*(_QWORD *)v177 + 24LL), v177, 0, a17, 0);
    if ( v187 )
    {
      v178 = (unsigned __int64 *)v188;
      sub_157E9D0(v187 + 40, (__int64)v120);
      v121 = *v178;
      v122 = v120[3] & 7LL;
      v120[4] = v178;
      v121 &= 0xFFFFFFFFFFFFFFF8LL;
      v120[3] = v121 | v122;
      *(_QWORD *)(v121 + 8) = v120 + 3;
      *v178 = *v178 & 7 | (unsigned __int64)(v120 + 3);
    }
    sub_164B780((__int64)v120, v184);
    if ( v186 )
    {
      v182[0] = (__int64)v186;
      sub_1623A60((__int64)v182, (__int64)v186, 2);
      v123 = v120[6];
      v124 = (__int64)(v120 + 6);
      if ( v123 )
      {
        sub_161E7C0((__int64)(v120 + 6), v123);
        v124 = (__int64)(v120 + 6);
      }
      v125 = (unsigned __int8 *)v182[0];
      v120[6] = v182[0];
      if ( v125 )
        sub_1623210((__int64)v182, v125, v124);
    }
    sub_15F8F50((__int64)v120, a16);
    v128 = (unsigned int)v195;
    if ( (unsigned int)v195 >= HIDWORD(v195) )
    {
      sub_16CD150((__int64)&v194, v196, 0, 8, v126, v127);
      v128 = (unsigned int)v195;
    }
    v115 = (unsigned int)++v114;
    *(_QWORD *)&v194[8 * v128] = v120;
    LODWORD(v195) = v195 + 1;
  }
  while ( v114 != v165 );
  v129 = 0;
  v130 = 0;
  do
  {
    v184[0] = (__int64)"dst.memcpy.gep.unroll";
    v185 = 259;
    v131 = sub_15A0680(v167, v130, 0);
    v179 = sub_12815B0((__int64 *)&v186, a2, (_BYTE *)a5, v131, (__int64)v184);
    v132 = *(_QWORD *)&v194[8 * v130];
    v185 = 257;
    v133 = sub_1648A60(64, 2u);
    v134 = v133;
    if ( v133 )
      sub_15F9650((__int64)v133, v132, v179, a18, 0);
    if ( v187 )
    {
      v180 = (unsigned __int64 *)v188;
      sub_157E9D0(v187 + 40, (__int64)v134);
      v135 = *v180;
      v136 = v134[3] & 7LL;
      v134[4] = v180;
      v135 &= 0xFFFFFFFFFFFFFFF8LL;
      v134[3] = v135 | v136;
      *(_QWORD *)(v135 + 8) = v134 + 3;
      *v180 = *v180 & 7 | (unsigned __int64)(v134 + 3);
    }
    sub_164B780((__int64)v134, v184);
    if ( v186 )
    {
      v182[0] = (__int64)v186;
      sub_1623A60((__int64)v182, (__int64)v186, 2);
      v137 = v134[6];
      v138 = (__int64)(v134 + 6);
      if ( v137 )
      {
        sub_161E7C0((__int64)(v134 + 6), v137);
        v138 = (__int64)(v134 + 6);
      }
      v139 = (unsigned __int8 *)v182[0];
      v134[6] = v182[0];
      if ( v139 )
        sub_1623210((__int64)v182, v139, v138);
    }
    v140 = (__int64)v134;
    v130 = (unsigned int)++v129;
    sub_15F9450(v140, a16);
  }
  while ( v129 != v165 );
  if ( v194 != (char *)v196 )
    _libc_free((unsigned __int64)v194);
LABEL_84:
  if ( v186 )
    sub_161E7C0((__int64)&v186, (__int64)v186);
}
