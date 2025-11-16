// Function: sub_20D1500
// Address: 0x20d1500
//
_BOOL8 __fastcall sub_20D1500(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        __int64 a5,
        _QWORD *a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        _QWORD *a15,
        int a16,
        int a17,
        int *a18)
{
  __int64 v18; // rax
  __int64 v19; // r13
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  unsigned __int8 *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int8 *v29; // rsi
  unsigned int v30; // eax
  __int64 v31; // rax
  __int64 v32; // r15
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 **v36; // r15
  __int64 **v37; // rdx
  int v38; // r8d
  int v39; // r9d
  __int64 v40; // rax
  __int64 v41; // rax
  _QWORD *v42; // r12
  unsigned int v43; // r15d
  __int64 *v44; // r14
  unsigned __int64 *v45; // rbx
  __int64 v46; // rax
  unsigned __int64 v47; // rcx
  __int64 v48; // rsi
  unsigned __int8 *v49; // rsi
  __int64 **v50; // rdx
  _QWORD *v51; // rax
  _QWORD *v52; // r12
  unsigned __int64 *v53; // rbx
  __int64 v54; // rax
  unsigned __int64 v55; // rcx
  __int64 v56; // rsi
  unsigned __int8 *v57; // rsi
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rbx
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 **v63; // r8
  unsigned __int8 *v64; // rcx
  __int64 v65; // rdx
  __int64 **v66; // r12
  __int64 **v67; // r13
  __int64 v68; // rbx
  unsigned __int8 *v69; // rsi
  __int64 v70; // r12
  __int64 v71; // rax
  const char *v72; // rsi
  size_t v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rax
  _QWORD *v76; // r12
  unsigned __int64 *v77; // r14
  __int64 v78; // rax
  unsigned __int64 v79; // rcx
  __int64 v80; // rsi
  unsigned __int8 *v81; // rsi
  __int64 v82; // rax
  __int64 v83; // rax
  double v84; // xmm4_8
  double v85; // xmm5_8
  __int64 v87; // rdx
  _QWORD *v88; // rax
  _QWORD *v89; // r12
  unsigned __int64 *v90; // r14
  __int64 v91; // rax
  unsigned __int64 v92; // rcx
  __int64 v93; // rsi
  unsigned __int8 *v94; // rsi
  double v95; // xmm4_8
  double v96; // xmm5_8
  unsigned int v97; // ebx
  _QWORD *v98; // rax
  __int64 v99; // r12
  __int64 *v100; // rbx
  __int64 v101; // rax
  __int64 v102; // rcx
  __int64 v103; // rsi
  unsigned __int8 *v104; // rsi
  __int64 **v105; // rdx
  _QWORD *v106; // rax
  _QWORD *v107; // rbx
  unsigned __int64 *v108; // r12
  __int64 v109; // rax
  unsigned __int64 v110; // rcx
  __int64 v111; // rsi
  unsigned __int8 *v112; // rsi
  __int64 v113; // rax
  __int64 **v114; // r12
  unsigned int v115; // ebx
  _QWORD *v116; // rax
  unsigned __int64 *v117; // rbx
  unsigned __int64 v118; // rcx
  __int64 v119; // rax
  __int64 v120; // rsi
  unsigned __int8 *v121; // rsi
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 *v124; // rbx
  __int64 v125; // rax
  __int64 v126; // rcx
  __int64 v127; // rsi
  unsigned __int8 *v128; // rsi
  unsigned __int64 *v129; // rbx
  unsigned __int64 v130; // rcx
  __int64 v131; // rax
  __int64 v132; // rbx
  __int64 v133; // rsi
  unsigned __int8 *v134; // rsi
  unsigned __int64 *v135; // rbx
  __int64 v136; // rax
  unsigned __int64 v137; // rcx
  __int64 v138; // rsi
  unsigned __int8 *v139; // rsi
  __int64 *v140; // [rsp+8h] [rbp-218h]
  __int64 v141; // [rsp+10h] [rbp-210h]
  __int64 **v142; // [rsp+18h] [rbp-208h]
  bool v143; // [rsp+26h] [rbp-1FAh]
  bool v144; // [rsp+27h] [rbp-1F9h]
  __int64 v145; // [rsp+30h] [rbp-1F0h]
  __int64 v146; // [rsp+38h] [rbp-1E8h]
  __int64 *v147; // [rsp+40h] [rbp-1E0h]
  _QWORD *v148; // [rsp+40h] [rbp-1E0h]
  __int64 *v149; // [rsp+50h] [rbp-1D0h]
  __int64 *v150; // [rsp+58h] [rbp-1C8h]
  __int64 v151; // [rsp+60h] [rbp-1C0h]
  __int64 v152; // [rsp+68h] [rbp-1B8h]
  int v153; // [rsp+70h] [rbp-1B0h]
  unsigned int v155; // [rsp+74h] [rbp-1ACh]
  __int64 v157; // [rsp+88h] [rbp-198h]
  bool v158; // [rsp+90h] [rbp-190h]
  __int64 v159; // [rsp+90h] [rbp-190h]
  __int64 *v160; // [rsp+98h] [rbp-188h]
  __int64 v161; // [rsp+98h] [rbp-188h]
  __int64 v162; // [rsp+98h] [rbp-188h]
  __int64 v163; // [rsp+A0h] [rbp-180h] BYREF
  unsigned __int8 *v164; // [rsp+A8h] [rbp-178h] BYREF
  __int64 v165[2]; // [rsp+B0h] [rbp-170h] BYREF
  __int16 v166; // [rsp+C0h] [rbp-160h]
  __int64 **v167; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v168; // [rsp+D8h] [rbp-148h]
  _QWORD v169[6]; // [rsp+E0h] [rbp-140h] BYREF
  unsigned __int8 *v170; // [rsp+110h] [rbp-110h] BYREF
  __int64 v171; // [rsp+118h] [rbp-108h]
  _WORD v172[24]; // [rsp+120h] [rbp-100h] BYREF
  unsigned __int8 *v173; // [rsp+150h] [rbp-D0h] BYREF
  __int64 v174; // [rsp+158h] [rbp-C8h]
  __int64 *v175; // [rsp+160h] [rbp-C0h]
  __int64 v176; // [rsp+168h] [rbp-B8h]
  __int64 v177; // [rsp+170h] [rbp-B0h]
  int v178; // [rsp+178h] [rbp-A8h]
  __int64 v179; // [rsp+180h] [rbp-A0h]
  __int64 v180; // [rsp+188h] [rbp-98h]
  unsigned __int8 *v181; // [rsp+1A0h] [rbp-80h] BYREF
  __int64 v182; // [rsp+1A8h] [rbp-78h]
  __int64 *v183; // [rsp+1B0h] [rbp-70h]
  __int64 v184; // [rsp+1B8h] [rbp-68h]
  __int64 v185; // [rsp+1C0h] [rbp-60h]
  int v186; // [rsp+1C8h] [rbp-58h]
  __int64 v187; // [rsp+1D0h] [rbp-50h]
  __int64 v188; // [rsp+1D8h] [rbp-48h]

  v19 = a2;
  v21 = a3;
  v160 = (__int64 *)sub_16498A0(a2);
  v146 = sub_15F2050(a2);
  v157 = sub_1632FA0(v146);
  v22 = sub_16498A0(a2);
  v23 = *(unsigned __int8 **)(a2 + 48);
  v173 = 0;
  v176 = v22;
  v24 = *(_QWORD *)(v19 + 40);
  v177 = 0;
  v174 = v24;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v175 = (__int64 *)(v19 + 24);
  v181 = v23;
  if ( v23 )
  {
    sub_1623A60((__int64)&v181, (__int64)v23, 2);
    v173 = v181;
    if ( v181 )
      sub_1623210((__int64)&v181, v181, (__int64)&v173);
  }
  v25 = *(_QWORD *)(sub_15F2060(v19) + 80);
  if ( !v25 )
    BUG();
  v26 = *(_QWORD *)(v25 + 24);
  if ( !v26 )
  {
    v18 = sub_16498A0(0);
    v181 = 0;
    v183 = 0;
    v184 = v18;
    v185 = 0;
    v186 = 0;
    v187 = 0;
    v188 = 0;
    v182 = 0;
    BUG();
  }
  v27 = sub_16498A0(v26 - 24);
  v181 = 0;
  v184 = v27;
  v185 = 0;
  v186 = 0;
  v187 = 0;
  v188 = 0;
  v28 = *(_QWORD *)(v26 + 16);
  v183 = (__int64 *)v26;
  v182 = v28;
  v29 = *(unsigned __int8 **)(v26 + 24);
  v170 = v29;
  if ( v29 )
  {
    sub_1623A60((__int64)&v170, (__int64)v29, 2);
    if ( v181 )
      sub_161E7C0((__int64)&v181, (__int64)v181);
    v181 = v170;
    if ( v170 )
      sub_1623210((__int64)&v170, v170, (__int64)&v181);
  }
  v158 = 0;
  v30 = (unsigned int)sub_15A96E0(v157) < 0x40 ? 8 : 16;
  if ( (unsigned int)v21 <= a4 && (unsigned int)v21 <= 0x10 )
  {
    v87 = 65814;
    if ( _bittest64(&v87, v21) )
      v158 = (unsigned int)v21 <= v30;
  }
  v150 = (__int64 *)sub_1644C60(v160, 8 * (int)v21);
  v155 = sub_15AAE50(v157, (__int64)v150);
  v31 = sub_1643360(v160);
  v151 = sub_159C470(v31, (unsigned int)v21, 0);
  v32 = dword_430AC00[a16];
  v33 = sub_1643350(v160);
  v152 = 0;
  v145 = sub_159C470(v33, v32, 0);
  if ( a15 )
  {
    v34 = dword_430AC00[a17];
    v35 = sub_1643350(v160);
    v152 = sub_159C470(v35, v34, 0);
  }
  v36 = *(__int64 ***)v19;
  v142 = *(__int64 ***)v19;
  v141 = sub_1643270(v160);
  v143 = v36 != (__int64 **)v141;
  if ( v158 )
  {
    switch ( (int)v21 )
    {
      case 1:
        v153 = a18[1];
        break;
      case 2:
        v153 = a18[2];
        break;
      case 4:
        v153 = a18[3];
        break;
      case 8:
        v153 = a18[4];
        break;
      case 16:
        v153 = a18[5];
        break;
      default:
        break;
    }
    v144 = 0;
    v167 = (__int64 **)v169;
    v168 = 0x600000000LL;
    v163 = 0;
  }
  else
  {
    v153 = *a18;
    if ( *a18 == 462 )
      goto LABEL_93;
    v167 = (__int64 **)v169;
    v168 = 0x600000001LL;
    v163 = 0;
    v59 = sub_15A9620(v157, (__int64)v160, 0);
    v169[0] = sub_159C470(v59, (unsigned int)v21, 0);
    v144 = a6 != 0;
  }
  v166 = 257;
  v37 = (__int64 **)sub_16471D0(v160, 0);
  if ( v37 != *(__int64 ***)a5 )
  {
    if ( *(_BYTE *)(a5 + 16) > 0x10u )
    {
      v172[0] = 257;
      a5 = sub_15FDBD0(47, a5, (__int64)v37, (__int64)&v170, 0);
      if ( v174 )
      {
        v124 = v175;
        sub_157E9D0(v174 + 40, a5);
        v125 = *(_QWORD *)(a5 + 24);
        v126 = *v124;
        *(_QWORD *)(a5 + 32) = v124;
        v126 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a5 + 24) = v126 | v125 & 7;
        *(_QWORD *)(v126 + 8) = a5 + 24;
        *v124 = *v124 & 7 | (a5 + 24);
      }
      sub_164B780(a5, v165);
      if ( v173 )
      {
        v164 = v173;
        sub_1623A60((__int64)&v164, (__int64)v173, 2);
        v127 = *(_QWORD *)(a5 + 48);
        if ( v127 )
          sub_161E7C0(a5 + 48, v127);
        v128 = v164;
        *(_QWORD *)(a5 + 48) = v164;
        if ( v128 )
          sub_1623210((__int64)&v164, v128, a5 + 48);
      }
    }
    else
    {
      a5 = sub_15A46C0(47, (__int64 ***)a5, v37, 0);
    }
  }
  v40 = (unsigned int)v168;
  if ( (unsigned int)v168 >= HIDWORD(v168) )
  {
    sub_16CD150((__int64)&v167, v169, 0, 8, v38, v39);
    v40 = (unsigned int)v168;
  }
  v167[v40] = (__int64 *)a5;
  v41 = (unsigned int)(v168 + 1);
  LODWORD(v168) = v168 + 1;
  if ( !a15 )
  {
    if ( !a6 )
    {
      if ( v142 == (__int64 **)v141 )
      {
        v149 = 0;
        v44 = 0;
        v147 = 0;
        v140 = 0;
        goto LABEL_58;
      }
      v44 = 0;
      v149 = 0;
      v147 = 0;
      if ( v158 )
      {
        v140 = 0;
        goto LABEL_58;
      }
      goto LABEL_141;
    }
    v44 = 0;
    v149 = 0;
LABEL_54:
    if ( v158 )
    {
      v172[0] = 257;
      v60 = sub_17FE280((__int64 *)&v173, (__int64)a6, (__int64)v150, (__int64 *)&v170);
      v61 = (unsigned int)v168;
      if ( (unsigned int)v168 >= HIDWORD(v168) )
      {
        sub_16CD150((__int64)&v167, v169, 0, 8, v38, v39);
        v61 = (unsigned int)v168;
      }
      v147 = 0;
      v140 = 0;
      v167[v61] = (__int64 *)v60;
      a6 = 0;
      v41 = (unsigned int)(v168 + 1);
      LODWORD(v168) = v168 + 1;
      goto LABEL_58;
    }
    v166 = 257;
    v148 = (_QWORD *)*a6;
    v97 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v182 + 56) + 40LL)) + 4);
    v172[0] = 257;
    v98 = sub_1648A60(64, 1u);
    v99 = (__int64)v98;
    if ( v98 )
      sub_15F8BC0((__int64)v98, v148, v97, 0, (__int64)&v170, 0);
    if ( v182 )
    {
      v100 = v183;
      sub_157E9D0(v182 + 40, v99);
      v101 = *(_QWORD *)(v99 + 24);
      v102 = *v100;
      *(_QWORD *)(v99 + 32) = v100;
      v102 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v99 + 24) = v102 | v101 & 7;
      *(_QWORD *)(v102 + 8) = v99 + 24;
      *v100 = *v100 & 7 | (v99 + 24);
    }
    sub_164B780(v99, v165);
    if ( v181 )
    {
      v164 = v181;
      sub_1623A60((__int64)&v164, (__int64)v181, 2);
      v103 = *(_QWORD *)(v99 + 48);
      if ( v103 )
        sub_161E7C0(v99 + 48, v103);
      v104 = v164;
      *(_QWORD *)(v99 + 48) = v164;
      if ( v104 )
        sub_1623210((__int64)&v164, v104, v99 + 48);
    }
    sub_15F8A20(v99, v155);
    v166 = 257;
    v105 = (__int64 **)sub_16471D0(v160, 0);
    if ( v105 == *(__int64 ***)v99 )
    {
      v147 = (__int64 *)v99;
      sub_15E7DE0((__int64 *)&v173, (_QWORD *)v99, v151);
    }
    else
    {
      if ( *(_BYTE *)(v99 + 16) > 0x10u )
      {
        v172[0] = 257;
        v147 = (__int64 *)sub_15FDBD0(47, v99, (__int64)v105, (__int64)&v170, 0);
        if ( v174 )
        {
          v129 = (unsigned __int64 *)v175;
          sub_157E9D0(v174 + 40, (__int64)v147);
          v130 = *v129;
          v131 = v147[3];
          v147[4] = (__int64)v129;
          v130 &= 0xFFFFFFFFFFFFFFF8LL;
          v147[3] = v130 | v131 & 7;
          *(_QWORD *)(v130 + 8) = v147 + 3;
          *v129 = *v129 & 7 | (unsigned __int64)(v147 + 3);
        }
        sub_164B780((__int64)v147, v165);
        if ( v173 )
        {
          v164 = v173;
          sub_1623A60((__int64)&v164, (__int64)v173, 2);
          v132 = (__int64)(v147 + 6);
          v133 = v147[6];
          if ( v133 )
            sub_161E7C0(v132, v133);
          v134 = v164;
          v147[6] = (__int64)v164;
          if ( v134 )
            sub_1623210((__int64)&v164, v134, v132);
        }
      }
      else
      {
        v147 = (__int64 *)sub_15A46C0(47, (__int64 ***)v99, v105, 0);
      }
      sub_15E7DE0((__int64 *)&v173, v147, v151);
    }
    v172[0] = 257;
    v106 = sub_1648A60(64, 2u);
    v107 = v106;
    if ( v106 )
      sub_15F9650((__int64)v106, (__int64)a6, v99, 0, 0);
    if ( v174 )
    {
      v108 = (unsigned __int64 *)v175;
      sub_157E9D0(v174 + 40, (__int64)v107);
      v109 = v107[3];
      v110 = *v108;
      v107[4] = v108;
      v110 &= 0xFFFFFFFFFFFFFFF8LL;
      v107[3] = v110 | v109 & 7;
      *(_QWORD *)(v110 + 8) = v107 + 3;
      *v108 = *v108 & 7 | (unsigned __int64)(v107 + 3);
    }
    sub_164B780((__int64)v107, (__int64 *)&v170);
    if ( v173 )
    {
      v165[0] = (__int64)v173;
      sub_1623A60((__int64)v165, (__int64)v173, 2);
      v111 = v107[6];
      if ( v111 )
        sub_161E7C0((__int64)(v107 + 6), v111);
      v112 = (unsigned __int8 *)v165[0];
      v107[6] = v165[0];
      if ( v112 )
        sub_1623210((__int64)v165, v112, (__int64)(v107 + 6));
    }
    sub_15F9450((__int64)v107, v155);
    v113 = (unsigned int)v168;
    if ( (unsigned int)v168 >= HIDWORD(v168) )
    {
      sub_16CD150((__int64)&v167, v169, 0, 8, v38, v39);
      v113 = (unsigned int)v168;
    }
    v167[v113] = v147;
    v41 = (unsigned int)(v168 + 1);
    LODWORD(v168) = v168 + 1;
    if ( a15 || !v143 )
    {
      v140 = 0;
      a6 = 0;
      goto LABEL_58;
    }
LABEL_141:
    v114 = *(__int64 ***)v19;
    v166 = 257;
    v115 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v182 + 56) + 40LL)) + 4);
    v172[0] = 257;
    v116 = sub_1648A60(64, 1u);
    a6 = v116;
    if ( v116 )
      sub_15F8BC0((__int64)v116, v114, v115, 0, (__int64)&v170, 0);
    if ( v182 )
    {
      v117 = (unsigned __int64 *)v183;
      sub_157E9D0(v182 + 40, (__int64)a6);
      v118 = *v117;
      v119 = a6[3];
      a6[4] = v117;
      v118 &= 0xFFFFFFFFFFFFFFF8LL;
      a6[3] = v118 | v119 & 7;
      *(_QWORD *)(v118 + 8) = a6 + 3;
      *v117 = *v117 & 7 | (unsigned __int64)(a6 + 3);
    }
    sub_164B780((__int64)a6, v165);
    if ( v181 )
    {
      v164 = v181;
      sub_1623A60((__int64)&v164, (__int64)v181, 2);
      v120 = a6[6];
      if ( v120 )
        sub_161E7C0((__int64)(a6 + 6), v120);
      v121 = v164;
      a6[6] = v164;
      if ( v121 )
        sub_1623210((__int64)&v164, v121, (__int64)(a6 + 6));
    }
    sub_15F8A20((__int64)a6, v155);
    v172[0] = 257;
    v122 = sub_16471D0(v160, 0);
    v140 = (__int64 *)sub_12AA3B0((__int64 *)&v173, 0x2Fu, (__int64)a6, v122, (__int64)&v170);
    sub_15E7DE0((__int64 *)&v173, v140, v151);
    v123 = (unsigned int)v168;
    if ( (unsigned int)v168 >= HIDWORD(v168) )
    {
      sub_16CD150((__int64)&v167, v169, 0, 8, v38, v39);
      v123 = (unsigned int)v168;
    }
    v167[v123] = v140;
    v41 = (unsigned int)(v168 + 1);
    LODWORD(v168) = v168 + 1;
    goto LABEL_58;
  }
  v166 = 257;
  v42 = (_QWORD *)*a15;
  v43 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v182 + 56) + 40LL)) + 4);
  v172[0] = 257;
  v44 = sub_1648A60(64, 1u);
  if ( v44 )
    sub_15F8BC0((__int64)v44, v42, v43, 0, (__int64)&v170, 0);
  if ( v182 )
  {
    v45 = (unsigned __int64 *)v183;
    sub_157E9D0(v182 + 40, (__int64)v44);
    v46 = v44[3];
    v47 = *v45;
    v44[4] = (__int64)v45;
    v47 &= 0xFFFFFFFFFFFFFFF8LL;
    v44[3] = v47 | v46 & 7;
    *(_QWORD *)(v47 + 8) = v44 + 3;
    *v45 = *v45 & 7 | (unsigned __int64)(v44 + 3);
  }
  sub_164B780((__int64)v44, v165);
  if ( v181 )
  {
    v164 = v181;
    sub_1623A60((__int64)&v164, (__int64)v181, 2);
    v48 = v44[6];
    if ( v48 )
      sub_161E7C0((__int64)(v44 + 6), v48);
    v49 = v164;
    v44[6] = (__int64)v164;
    if ( v49 )
      sub_1623210((__int64)&v164, v49, (__int64)(v44 + 6));
  }
  sub_15F8A20((__int64)v44, v155);
  v166 = 257;
  v50 = (__int64 **)sub_16471D0(v160, 0);
  if ( v50 == (__int64 **)*v44 )
  {
    v149 = v44;
  }
  else if ( *((_BYTE *)v44 + 16) > 0x10u )
  {
    v172[0] = 257;
    v149 = (__int64 *)sub_15FDBD0(47, (__int64)v44, (__int64)v50, (__int64)&v170, 0);
    if ( v174 )
    {
      v135 = (unsigned __int64 *)v175;
      sub_157E9D0(v174 + 40, (__int64)v149);
      v136 = v149[3];
      v137 = *v135;
      v149[4] = (__int64)v135;
      v137 &= 0xFFFFFFFFFFFFFFF8LL;
      v149[3] = v137 | v136 & 7;
      *(_QWORD *)(v137 + 8) = v149 + 3;
      *v135 = *v135 & 7 | (unsigned __int64)(v149 + 3);
    }
    sub_164B780((__int64)v149, v165);
    if ( v173 )
    {
      v164 = v173;
      sub_1623A60((__int64)&v164, (__int64)v173, 2);
      v138 = v149[6];
      if ( v138 )
        sub_161E7C0((__int64)(v149 + 6), v138);
      v139 = v164;
      v149[6] = (__int64)v164;
      if ( v139 )
        sub_1623210((__int64)&v164, v139, (__int64)(v149 + 6));
    }
  }
  else
  {
    v149 = (__int64 *)sub_15A46C0(47, (__int64 ***)v44, v50, 0);
  }
  sub_15E7DE0((__int64 *)&v173, v149, v151);
  v172[0] = 257;
  v51 = sub_1648A60(64, 2u);
  v52 = v51;
  if ( v51 )
    sub_15F9650((__int64)v51, (__int64)a15, (__int64)v44, 0, 0);
  if ( v174 )
  {
    v53 = (unsigned __int64 *)v175;
    sub_157E9D0(v174 + 40, (__int64)v52);
    v54 = v52[3];
    v55 = *v53;
    v52[4] = v53;
    v55 &= 0xFFFFFFFFFFFFFFF8LL;
    v52[3] = v55 | v54 & 7;
    *(_QWORD *)(v55 + 8) = v52 + 3;
    *v53 = *v53 & 7 | (unsigned __int64)(v52 + 3);
  }
  sub_164B780((__int64)v52, (__int64 *)&v170);
  if ( v173 )
  {
    v165[0] = (__int64)v173;
    sub_1623A60((__int64)v165, (__int64)v173, 2);
    v56 = v52[6];
    if ( v56 )
      sub_161E7C0((__int64)(v52 + 6), v56);
    v57 = (unsigned __int8 *)v165[0];
    v52[6] = v165[0];
    if ( v57 )
      sub_1623210((__int64)v165, v57, (__int64)(v52 + 6));
  }
  sub_15F9450((__int64)v52, v155);
  v58 = (unsigned int)v168;
  if ( (unsigned int)v168 >= HIDWORD(v168) )
  {
    sub_16CD150((__int64)&v167, v169, 0, 8, v38, v39);
    v58 = (unsigned int)v168;
  }
  v167[v58] = v149;
  v41 = (unsigned int)(v168 + 1);
  LODWORD(v168) = v168 + 1;
  if ( a6 )
    goto LABEL_54;
  v147 = 0;
  v140 = 0;
LABEL_58:
  if ( HIDWORD(v168) <= (unsigned int)v41 )
  {
    sub_16CD150((__int64)&v167, v169, 0, 8, v38, v39);
    v41 = (unsigned int)v168;
  }
  v167[v41] = (__int64 *)v145;
  v62 = (unsigned int)(v168 + 1);
  LODWORD(v168) = v168 + 1;
  if ( v152 )
  {
    if ( (unsigned int)v62 >= HIDWORD(v168) )
    {
      sub_16CD150((__int64)&v167, v169, 0, 8, v38, v39);
      v62 = (unsigned int)v168;
    }
    v167[v62] = (__int64 *)v152;
    LODWORD(v168) = v168 + 1;
  }
  if ( a15 )
  {
    v150 = (__int64 *)sub_1643320(v160);
    v163 = sub_1563AB0(&v163, v160, 0, 58);
  }
  else if ( !v143 || !v158 )
  {
    v150 = (__int64 *)sub_1643270(v160);
  }
  v171 = 0x600000000LL;
  v170 = (unsigned __int8 *)v172;
  v63 = &v167[(unsigned int)v168];
  if ( v167 == v63 )
  {
    v65 = 0;
    v69 = (unsigned __int8 *)v172;
  }
  else
  {
    v161 = v19;
    v64 = (unsigned __int8 *)v172;
    v65 = 0;
    v66 = v167 + 1;
    v67 = &v167[(unsigned int)v168];
    v68 = **v167;
    while ( 1 )
    {
      *(_QWORD *)&v64[8 * v65] = v68;
      v65 = (unsigned int)(v171 + 1);
      LODWORD(v171) = v171 + 1;
      if ( v67 == v66 )
        break;
      v68 = **v66;
      if ( HIDWORD(v171) <= (unsigned int)v65 )
      {
        sub_16CD150((__int64)&v170, v172, 0, 8, (int)v63, v39);
        v65 = (unsigned int)v171;
      }
      v64 = v170;
      ++v66;
    }
    v19 = v161;
    v69 = v170;
  }
  v70 = sub_1644EA0(v150, v69, v65, 0);
  v71 = *(_QWORD *)(a1 + 160);
  v72 = *(const char **)(v71 + 8LL * v153 + 74096);
  v73 = 0;
  if ( v72 )
  {
    v72 = *(const char **)(v71 + 8LL * v153 + 74096);
    v73 = strlen(v72);
  }
  v74 = sub_1632080(v146, (__int64)v72, v73, v70, v163);
  v166 = 257;
  v162 = sub_1285290(
           (__int64 *)&v173,
           *(_QWORD *)(*(_QWORD *)v74 + 24LL),
           v74,
           (int)v167,
           (unsigned int)v168,
           (__int64)v165,
           0);
  *(_QWORD *)(v162 + 56) = v163;
  if ( v144 )
    sub_15E7E90((__int64 *)&v173, v147, v151);
  if ( a15 )
  {
    v75 = sub_1599EF0(*(__int64 ***)v19);
    v166 = 257;
    v159 = v75;
    v76 = sub_1648A60(64, 1u);
    if ( v76 )
      sub_15F9210((__int64)v76, *(_QWORD *)(*v44 + 24), (__int64)v44, 0, 0, 0);
    if ( v174 )
    {
      v77 = (unsigned __int64 *)v175;
      sub_157E9D0(v174 + 40, (__int64)v76);
      v78 = v76[3];
      v79 = *v77;
      v76[4] = v77;
      v79 &= 0xFFFFFFFFFFFFFFF8LL;
      v76[3] = v79 | v78 & 7;
      *(_QWORD *)(v79 + 8) = v76 + 3;
      *v77 = *v77 & 7 | (unsigned __int64)(v76 + 3);
    }
    sub_164B780((__int64)v76, v165);
    if ( v173 )
    {
      v164 = v173;
      sub_1623A60((__int64)&v164, (__int64)v173, 2);
      v80 = v76[6];
      if ( v80 )
        sub_161E7C0((__int64)(v76 + 6), v80);
      v81 = v164;
      v76[6] = v164;
      if ( v81 )
        sub_1623210((__int64)&v164, v81, (__int64)(v76 + 6));
    }
    sub_15F8F50((__int64)v76, v155);
    sub_15E7E90((__int64 *)&v173, v149, v151);
    v166 = 257;
    LODWORD(v164) = 0;
    v82 = sub_17FE490((__int64 *)&v173, v159, (__int64)v76, &v164, 1, v165);
    v166 = 257;
    LODWORD(v164) = 1;
    v83 = sub_17FE490((__int64 *)&v173, v82, v162, &v164, 1, v165);
    sub_164D160(v19, v83, a7, a8, a9, a10, v84, v85, a13, a14);
  }
  else if ( v142 != (__int64 **)v141 )
  {
    v166 = 257;
    if ( v158 )
    {
      v89 = (_QWORD *)sub_17FE280((__int64 *)&v173, v162, *(_QWORD *)v19, v165);
    }
    else
    {
      v88 = sub_1648A60(64, 1u);
      v89 = v88;
      if ( v88 )
        sub_15F9210((__int64)v88, *(_QWORD *)(*a6 + 24LL), (__int64)a6, 0, 0, 0);
      if ( v174 )
      {
        v90 = (unsigned __int64 *)v175;
        sub_157E9D0(v174 + 40, (__int64)v89);
        v91 = v89[3];
        v92 = *v90;
        v89[4] = v90;
        v92 &= 0xFFFFFFFFFFFFFFF8LL;
        v89[3] = v92 | v91 & 7;
        *(_QWORD *)(v92 + 8) = v89 + 3;
        *v90 = *v90 & 7 | (unsigned __int64)(v89 + 3);
      }
      sub_164B780((__int64)v89, v165);
      if ( v173 )
      {
        v164 = v173;
        sub_1623A60((__int64)&v164, (__int64)v173, 2);
        v93 = v89[6];
        if ( v93 )
          sub_161E7C0((__int64)(v89 + 6), v93);
        v94 = v164;
        v89[6] = v164;
        if ( v94 )
          sub_1623210((__int64)&v164, v94, (__int64)(v89 + 6));
      }
      sub_15F8F50((__int64)v89, v155);
      sub_15E7E90((__int64 *)&v173, v140, v151);
    }
    sub_164D160(v19, (__int64)v89, a7, a8, a9, a10, v95, v96, a13, a14);
  }
  sub_15F20C0((_QWORD *)v19);
  if ( v170 != (unsigned __int8 *)v172 )
    _libc_free((unsigned __int64)v170);
  if ( v167 != v169 )
    _libc_free((unsigned __int64)v167);
  v158 = 1;
LABEL_93:
  if ( v181 )
    sub_161E7C0((__int64)&v181, (__int64)v181);
  if ( v173 )
    sub_161E7C0((__int64)&v173, (__int64)v173);
  return v158;
}
