// Function: sub_1EFDAC0
// Address: 0x1efdac0
//
__int64 __fastcall sub_1EFDAC0(
        _QWORD *a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  _QWORD *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rcx
  _QWORD *v12; // r13
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned __int8 *v17; // rsi
  _QWORD *v18; // rax
  unsigned __int8 *v19; // rsi
  unsigned __int8 *v20; // rsi
  unsigned int v21; // eax
  __int64 **v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r13
  __int64 v25; // rcx
  __int64 v26; // rbx
  unsigned __int8 *v27; // rsi
  _QWORD *v28; // r15
  _QWORD *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // r13
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  unsigned __int8 **v37; // r8
  __int64 v38; // r9
  __int64 v39; // rsi
  __int64 v40; // rsi
  int v41; // eax
  __int64 v42; // rax
  int v43; // edx
  __int64 v44; // rdx
  __int64 *v45; // rax
  __int64 v46; // rcx
  unsigned __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rdx
  int v53; // eax
  __int64 v54; // rax
  int v55; // edx
  __int64 v56; // rdx
  __int64 *v57; // rax
  __int64 v58; // rcx
  unsigned __int64 v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // rdx
  __int64 v62; // r13
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // r13
  __int64 v66; // r15
  __int64 v67; // rax
  __int64 v68; // r14
  _QWORD *v69; // r13
  __int64 v70; // rax
  _QWORD *v71; // r14
  unsigned __int8 *v72; // rsi
  _QWORD *v73; // rax
  unsigned __int8 *v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  _QWORD *v78; // rax
  __int64 v79; // r15
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 v82; // rsi
  __int64 v83; // rdx
  unsigned __int8 *v84; // rsi
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  unsigned __int8 *v88; // rsi
  _QWORD *v89; // rax
  _QWORD *v90; // rax
  __int64 v91; // r10
  __int64 *v92; // rsi
  __int64 *v93; // r15
  __int64 v94; // rcx
  __int64 v95; // rax
  __int64 v96; // r10
  __int64 v97; // rsi
  __int64 v98; // r15
  unsigned __int8 *v99; // rsi
  _QWORD *v100; // rax
  _QWORD **v101; // rax
  __int64 *v102; // rax
  __int64 v103; // rsi
  unsigned __int64 *v104; // r15
  __int64 v105; // rax
  unsigned __int64 v106; // rcx
  __int64 v107; // rsi
  unsigned __int8 *v108; // rsi
  _QWORD *v109; // rax
  __int64 *v110; // r13
  __int64 v111; // rax
  __int64 v112; // rcx
  __int64 v113; // rsi
  unsigned __int8 *v114; // rsi
  __int64 v115; // rax
  __int64 v116; // r13
  __int64 v117; // rbx
  __int64 *v118; // r15
  __int64 v119; // rax
  __int64 v120; // rcx
  __int64 v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // r8
  __int64 v124; // r9
  __int64 v125; // rsi
  __int64 v126; // rsi
  int v127; // eax
  __int64 v128; // rax
  int v129; // edx
  __int64 v130; // rdx
  __int64 *v131; // rax
  __int64 v132; // rcx
  unsigned __int64 v133; // rdx
  __int64 v134; // rdx
  __int64 v135; // rdx
  __int64 v136; // rax
  __int64 v137; // rcx
  __int64 v138; // rdx
  int v139; // eax
  __int64 v140; // rax
  int v141; // edx
  __int64 v142; // rdx
  __int64 *v143; // rax
  __int64 v144; // rcx
  unsigned __int64 v145; // rdx
  __int64 v146; // rdx
  __int64 v147; // rdx
  __int64 v148; // rbx
  __int64 v149; // rax
  double v150; // xmm4_8
  double v151; // xmm5_8
  __int64 result; // rax
  _QWORD *v153; // rax
  _QWORD *v154; // r15
  unsigned __int64 *v155; // rbx
  __int64 v156; // rax
  unsigned __int64 v157; // rcx
  __int64 v158; // rsi
  unsigned __int8 *v159; // rsi
  double v160; // xmm4_8
  double v161; // xmm5_8
  __int64 v162; // rax
  __int64 v163; // r15
  __int64 *v164; // rbx
  __int64 v165; // rax
  __int64 v166; // rcx
  __int64 v167; // rsi
  __int64 v168; // rbx
  __int64 v169; // rdx
  __int64 v170; // rax
  __int64 v171; // rax
  _QWORD *v172; // rax
  _QWORD *v173; // r13
  unsigned __int64 v174; // rsi
  __int64 v175; // rax
  __int64 v176; // rsi
  __int64 v177; // rdx
  unsigned __int8 *v178; // rsi
  __int64 v179; // rax
  __int64 v180; // rax
  __int64 v181; // rax
  double v182; // xmm4_8
  double v183; // xmm5_8
  __int64 v184; // [rsp+10h] [rbp-150h]
  __int64 *v185; // [rsp+18h] [rbp-148h]
  _QWORD *v186; // [rsp+18h] [rbp-148h]
  __int64 v187; // [rsp+20h] [rbp-140h]
  __int64 v188; // [rsp+28h] [rbp-138h]
  unsigned int v189; // [rsp+34h] [rbp-12Ch]
  __int64 *v190; // [rsp+38h] [rbp-128h]
  __int64 v191; // [rsp+40h] [rbp-120h]
  __int64 ***v192; // [rsp+48h] [rbp-118h]
  __int64 *v193; // [rsp+50h] [rbp-110h]
  __int64 v195; // [rsp+60h] [rbp-100h]
  __int64 v196; // [rsp+68h] [rbp-F8h]
  __int64 *v197; // [rsp+68h] [rbp-F8h]
  __int64 v198; // [rsp+68h] [rbp-F8h]
  __int64 v199; // [rsp+68h] [rbp-F8h]
  __int64 v200; // [rsp+68h] [rbp-F8h]
  _QWORD *v201; // [rsp+68h] [rbp-F8h]
  __int64 v202; // [rsp+70h] [rbp-F0h]
  unsigned __int64 *v203; // [rsp+70h] [rbp-F0h]
  _QWORD *v204; // [rsp+80h] [rbp-E0h]
  __int64 v205; // [rsp+80h] [rbp-E0h]
  __int64 v206; // [rsp+88h] [rbp-D8h]
  _QWORD *v207; // [rsp+88h] [rbp-D8h]
  __int64 v208; // [rsp+88h] [rbp-D8h]
  __int64 v209; // [rsp+88h] [rbp-D8h]
  __int64 v210; // [rsp+88h] [rbp-D8h]
  unsigned __int8 *v211; // [rsp+98h] [rbp-C8h] BYREF
  __int64 v212[2]; // [rsp+A0h] [rbp-C0h] BYREF
  __int16 v213; // [rsp+B0h] [rbp-B0h]
  unsigned __int8 *v214[2]; // [rsp+C0h] [rbp-A0h] BYREF
  __int16 v215; // [rsp+D0h] [rbp-90h]
  unsigned __int8 *v216; // [rsp+E0h] [rbp-80h] BYREF
  _QWORD *v217; // [rsp+E8h] [rbp-78h]
  __int64 *v218; // [rsp+F0h] [rbp-70h]
  _QWORD *v219; // [rsp+F8h] [rbp-68h]
  __int64 v220; // [rsp+100h] [rbp-60h]
  int v221; // [rsp+108h] [rbp-58h]
  __int64 v222; // [rsp+110h] [rbp-50h]
  __int64 v223; // [rsp+118h] [rbp-48h]

  v9 = a1;
  v10 = *((_DWORD *)a1 + 5) & 0xFFFFFFF;
  v192 = (__int64 ***)a1[-3 * v10];
  v11 = a1[3 * (1 - v10)];
  v12 = *(_QWORD **)(v11 + 24);
  v195 = a1[3 * (2 - v10)];
  v184 = a1[3 * (3 - v10)];
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = (_QWORD *)*v12;
  v13 = 0;
  if ( *(_BYTE *)(*a1 + 8LL) == 16 )
    v13 = *a1;
  v187 = v13;
  v190 = **(__int64 ***)(*a1 + 16LL);
  v14 = (_QWORD *)sub_16498A0((__int64)a1);
  v17 = (unsigned __int8 *)a1[6];
  v216 = 0;
  v219 = v14;
  v18 = (_QWORD *)a1[5];
  v220 = 0;
  v204 = v18;
  v217 = v18;
  v221 = 0;
  v222 = 0;
  v223 = 0;
  v193 = a1 + 3;
  v218 = a1 + 3;
  v214[0] = v17;
  if ( v17 )
  {
    sub_1623A60((__int64)v214, (__int64)v17, 2);
    if ( v216 )
      sub_161E7C0((__int64)&v216, (__int64)v216);
    v216 = v214[0];
    if ( v214[0] )
      sub_1623210((__int64)v214, v214[0], (__int64)&v216);
  }
  v19 = (unsigned __int8 *)a1[6];
  v214[0] = v19;
  if ( v19 )
  {
    sub_1623A60((__int64)v214, (__int64)v19, 2);
    v20 = v216;
    if ( !v216 )
      goto LABEL_13;
    goto LABEL_12;
  }
  v20 = v216;
  if ( v216 )
  {
LABEL_12:
    sub_161E7C0((__int64)&v216, (__int64)v20);
LABEL_13:
    v20 = v214[0];
    v216 = v214[0];
    if ( v214[0] )
      sub_1623210((__int64)v214, v214[0], (__int64)&v216);
    if ( *(_BYTE *)(v195 + 16) > 0x10u )
      goto LABEL_16;
    goto LABEL_175;
  }
  if ( *(_BYTE *)(v195 + 16) > 0x10u )
    goto LABEL_16;
LABEL_175:
  if ( sub_1596070(v195, (__int64)v20, v15, v16) )
  {
    v215 = 257;
    v153 = sub_1648A60(64, 1u);
    v154 = v153;
    if ( v153 )
      sub_15F9210((__int64)v153, (__int64)(*v192)[3], (__int64)v192, 0, 0, 0);
    if ( v217 )
    {
      v155 = (unsigned __int64 *)v218;
      sub_157E9D0((__int64)(v217 + 5), (__int64)v154);
      v156 = v154[3];
      v157 = *v155;
      v154[4] = v155;
      v157 &= 0xFFFFFFFFFFFFFFF8LL;
      v154[3] = v157 | v156 & 7;
      *(_QWORD *)(v157 + 8) = v154 + 3;
      *v155 = *v155 & 7 | (unsigned __int64)(v154 + 3);
    }
    sub_164B780((__int64)v154, (__int64 *)v214);
    if ( v216 )
    {
      v212[0] = (__int64)v216;
      sub_1623A60((__int64)v212, (__int64)v216, 2);
      v158 = v154[6];
      if ( v158 )
        sub_161E7C0((__int64)(v154 + 6), v158);
      v159 = (unsigned __int8 *)v212[0];
      v154[6] = v212[0];
      if ( v159 )
        sub_1623210((__int64)v212, v159, (__int64)(v154 + 6));
    }
    sub_15F8F50((__int64)v154, (unsigned int)v12);
    sub_164D160((__int64)a1, (__int64)v154, a2, a3, a4, a5, v160, v161, a8, a9);
    result = sub_15F20C0(a1);
    goto LABEL_186;
  }
LABEL_16:
  v21 = (unsigned int)sub_16431D0(v187) >> 3;
  if ( v21 <= (unsigned int)v12 )
    LODWORD(v12) = v21;
  v189 = (unsigned int)v12;
  v22 = (__int64 **)sub_1647190(v190, *((_DWORD *)*v192 + 2) >> 8);
  v23 = 257;
  v213 = 257;
  if ( v22 != *v192 )
  {
    v23 = (__int64)v192;
    if ( *((_BYTE *)v192 + 16) > 0x10u )
    {
      v215 = 257;
      v162 = sub_15FDBD0(47, (__int64)v192, (__int64)v22, (__int64)v214, 0);
      v192 = (__int64 ***)v162;
      v163 = v162;
      if ( v217 )
      {
        v164 = v218;
        sub_157E9D0((__int64)(v217 + 5), v162);
        v165 = *(_QWORD *)(v163 + 24);
        v166 = *v164;
        *(_QWORD *)(v163 + 32) = v164;
        v166 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v163 + 24) = v166 | v165 & 7;
        *(_QWORD *)(v166 + 8) = v163 + 24;
        *v164 = *v164 & 7 | (v163 + 24);
      }
      sub_164B780((__int64)v192, v212);
      v23 = (__int64)v216;
      if ( v216 )
      {
        v211 = v216;
        sub_1623A60((__int64)&v211, (__int64)v216, 2);
        v167 = (__int64)v192[6];
        if ( v167 )
          sub_161E7C0((__int64)(v192 + 6), v167);
        v23 = (__int64)v211;
        v192[6] = (__int64 **)v211;
        if ( v23 )
          sub_1623210((__int64)&v211, (unsigned __int8 *)v23, (__int64)(v192 + 6));
      }
    }
    else
    {
      v192 = (__int64 ***)sub_15A46C0(47, v192, v22, 0);
    }
  }
  v24 = *(_QWORD *)(v187 + 32);
  v206 = sub_1599EF0((__int64 **)v187);
  if ( *(_BYTE *)(v195 + 16) == 8 )
  {
    if ( (_DWORD)v24 )
    {
      v168 = 0;
      v205 = (unsigned int)v24;
      do
      {
        v169 = v168 - (*(_DWORD *)(v195 + 20) & 0xFFFFFFF);
        if ( !sub_1593BB0(*(_QWORD *)(v195 + 24 * v169), v23, v169, v25) )
        {
          v215 = 257;
          v170 = sub_1643350(v219);
          v171 = sub_159C470(v170, v168, 0);
          v202 = sub_17CEC00((__int64 *)&v216, (__int64)v190, v192, v171, (__int64 *)v214);
          v215 = 257;
          v172 = sub_1648A60(64, 1u);
          v173 = v172;
          if ( v172 )
            sub_15F9210((__int64)v172, *(_QWORD *)(*(_QWORD *)v202 + 24LL), v202, 0, 0, 0);
          if ( v217 )
          {
            v203 = (unsigned __int64 *)v218;
            sub_157E9D0((__int64)(v217 + 5), (__int64)v173);
            v174 = *v203;
            v175 = v173[3] & 7LL;
            v173[4] = v203;
            v174 &= 0xFFFFFFFFFFFFFFF8LL;
            v173[3] = v174 | v175;
            *(_QWORD *)(v174 + 8) = v173 + 3;
            *v203 = *v203 & 7 | (unsigned __int64)(v173 + 3);
          }
          sub_164B780((__int64)v173, (__int64 *)v214);
          if ( v216 )
          {
            v212[0] = (__int64)v216;
            sub_1623A60((__int64)v212, (__int64)v216, 2);
            v176 = v173[6];
            v177 = (__int64)(v173 + 6);
            if ( v176 )
            {
              sub_161E7C0((__int64)(v173 + 6), v176);
              v177 = (__int64)(v173 + 6);
            }
            v178 = (unsigned __int8 *)v212[0];
            v173[6] = v212[0];
            if ( v178 )
              sub_1623210((__int64)v212, v178, v177);
          }
          sub_15F8F50((__int64)v173, v189);
          v215 = 257;
          v179 = sub_1643350(v219);
          v180 = sub_159C470(v179, v168, 0);
          v23 = v206;
          v206 = sub_156D8B0((__int64 *)&v216, v206, (__int64)v173, v180, (__int64)v214);
        }
        ++v168;
      }
      while ( v205 != v168 );
    }
    v215 = 257;
    v181 = sub_156B790((__int64 *)&v216, v195, v206, v184, (__int64)v214, 0);
    sub_164D160((__int64)a1, v181, a2, a3, a4, a5, v182, v183, a8, a9);
    result = sub_15F20C0(a1);
LABEL_186:
    if ( v216 )
      return sub_161E7C0((__int64)&v216, (__int64)v216);
    return result;
  }
  if ( (_DWORD)v24 )
  {
    v26 = 0;
    v188 = (unsigned int)(v24 - 1);
    v191 = v206;
    while ( 1 )
    {
      v213 = 257;
      v63 = sub_1643350(v219);
      v64 = sub_159C470(v63, v26, 0);
      v65 = v64;
      if ( *(_BYTE *)(v195 + 16) > 0x10u || *(_BYTE *)(v64 + 16) > 0x10u )
      {
        v215 = 257;
        v109 = sub_1648A60(56, 2u);
        v66 = (__int64)v109;
        if ( v109 )
          sub_15FA320((__int64)v109, (_QWORD *)v195, v65, (__int64)v214, 0);
        if ( v217 )
        {
          v110 = v218;
          sub_157E9D0((__int64)(v217 + 5), v66);
          v111 = *(_QWORD *)(v66 + 24);
          v112 = *v110;
          *(_QWORD *)(v66 + 32) = v110;
          v112 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v66 + 24) = v112 | v111 & 7;
          *(_QWORD *)(v112 + 8) = v66 + 24;
          *v110 = *v110 & 7 | (v66 + 24);
        }
        sub_164B780(v66, v212);
        if ( v216 )
        {
          v211 = v216;
          sub_1623A60((__int64)&v211, (__int64)v216, 2);
          v113 = *(_QWORD *)(v66 + 48);
          if ( v113 )
            sub_161E7C0(v66 + 48, v113);
          v114 = v211;
          *(_QWORD *)(v66 + 48) = v211;
          if ( v114 )
            sub_1623210((__int64)&v211, v114, v66 + 48);
        }
      }
      else
      {
        v66 = sub_15A37D0((_BYTE *)v195, v64, 0);
      }
      v213 = 257;
      v67 = sub_15A0680(*(_QWORD *)v66, 1, 0);
      v68 = v67;
      if ( *(_BYTE *)(v66 + 16) > 0x10u || *(_BYTE *)(v67 + 16) > 0x10u )
      {
        v215 = 257;
        v100 = sub_1648A60(56, 2u);
        v69 = v100;
        if ( v100 )
        {
          v200 = (__int64)v100;
          v101 = *(_QWORD ***)v66;
          if ( *(_BYTE *)(*(_QWORD *)v66 + 8LL) == 16 )
          {
            v186 = v101[4];
            v102 = (__int64 *)sub_1643320(*v101);
            v103 = (__int64)sub_16463B0(v102, (unsigned int)v186);
          }
          else
          {
            v103 = sub_1643320(*v101);
          }
          sub_15FEC10((__int64)v69, v103, 51, 32, v66, v68, (__int64)v214, 0);
        }
        else
        {
          v200 = 0;
        }
        if ( v217 )
        {
          v104 = (unsigned __int64 *)v218;
          sub_157E9D0((__int64)(v217 + 5), (__int64)v69);
          v105 = v69[3];
          v106 = *v104;
          v69[4] = v104;
          v106 &= 0xFFFFFFFFFFFFFFF8LL;
          v69[3] = v106 | v105 & 7;
          *(_QWORD *)(v106 + 8) = v69 + 3;
          *v104 = *v104 & 7 | (unsigned __int64)(v69 + 3);
        }
        sub_164B780(v200, v212);
        if ( v216 )
        {
          v211 = v216;
          sub_1623A60((__int64)&v211, (__int64)v216, 2);
          v107 = v69[6];
          if ( v107 )
            sub_161E7C0((__int64)(v69 + 6), v107);
          v108 = v211;
          v69[6] = v211;
          if ( v108 )
            sub_1623210((__int64)&v211, v108, (__int64)(v69 + 6));
        }
      }
      else
      {
        v69 = (_QWORD *)sub_15A37B0(0x20u, (_QWORD *)v66, (_QWORD *)v67, 0);
      }
      v214[0] = (unsigned __int8 *)"cond.load";
      v215 = 259;
      v70 = sub_157FBF0(v204, v193, (__int64)v214);
      v218 = v193;
      v71 = (_QWORD *)v70;
      v72 = (unsigned __int8 *)a1[6];
      v73 = (_QWORD *)a1[5];
      v214[0] = v72;
      v217 = v73;
      if ( v72 )
      {
        sub_1623A60((__int64)v214, (__int64)v72, 2);
        v74 = v216;
        if ( !v216 )
          goto LABEL_78;
      }
      else
      {
        v74 = v216;
        if ( !v216 )
          goto LABEL_80;
      }
      sub_161E7C0((__int64)&v216, (__int64)v74);
LABEL_78:
      v216 = v214[0];
      if ( v214[0] )
        sub_1623210((__int64)v214, v214[0], (__int64)&v216);
LABEL_80:
      v215 = 257;
      v75 = sub_1643350(v219);
      v76 = sub_159C470(v75, v26, 0);
      v77 = sub_17CEC00((__int64 *)&v216, (__int64)v190, v192, v76, (__int64 *)v214);
      v215 = 257;
      v196 = v77;
      v78 = sub_1648A60(64, 1u);
      v79 = (__int64)v78;
      if ( v78 )
        sub_15F9210((__int64)v78, *(_QWORD *)(*(_QWORD *)v196 + 24LL), v196, 0, 0, 0);
      if ( v217 )
      {
        v197 = v218;
        sub_157E9D0((__int64)(v217 + 5), v79);
        v80 = *v197;
        v81 = *(_QWORD *)(v79 + 24) & 7LL;
        *(_QWORD *)(v79 + 32) = v197;
        v80 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v79 + 24) = v80 | v81;
        *(_QWORD *)(v80 + 8) = v79 + 24;
        *v197 = *v197 & 7 | (v79 + 24);
      }
      sub_164B780(v79, (__int64 *)v214);
      if ( v216 )
      {
        v212[0] = (__int64)v216;
        sub_1623A60((__int64)v212, (__int64)v216, 2);
        v82 = *(_QWORD *)(v79 + 48);
        v83 = v79 + 48;
        if ( v82 )
        {
          sub_161E7C0(v79 + 48, v82);
          v83 = v79 + 48;
        }
        v84 = (unsigned __int8 *)v212[0];
        *(_QWORD *)(v79 + 48) = v212[0];
        if ( v84 )
          sub_1623210((__int64)v212, v84, v83);
      }
      sub_15F8F50(v79, v189);
      v213 = 257;
      v85 = sub_1643350(v219);
      v86 = sub_159C470(v85, v26, 0);
      if ( *(_BYTE *)(v206 + 16) > 0x10u || *(_BYTE *)(v79 + 16) > 0x10u || *(_BYTE *)(v86 + 16) > 0x10u )
      {
        v199 = v86;
        v215 = 257;
        v90 = sub_1648A60(56, 3u);
        v91 = (__int64)v90;
        if ( v90 )
        {
          v92 = (__int64 *)v206;
          v207 = v90;
          sub_15FA480((__int64)v90, v92, v79, v199, (__int64)v214, 0);
          v91 = (__int64)v207;
        }
        if ( v217 )
        {
          v93 = v218;
          v208 = v91;
          sub_157E9D0((__int64)(v217 + 5), v91);
          v91 = v208;
          v94 = *v93;
          v95 = *(_QWORD *)(v208 + 24);
          *(_QWORD *)(v208 + 32) = v93;
          v94 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v208 + 24) = v94 | v95 & 7;
          *(_QWORD *)(v94 + 8) = v208 + 24;
          *v93 = *v93 & 7 | (v208 + 24);
        }
        v209 = v91;
        sub_164B780(v91, v212);
        v96 = v209;
        if ( v216 )
        {
          v211 = v216;
          sub_1623A60((__int64)&v211, (__int64)v216, 2);
          v96 = v209;
          v97 = *(_QWORD *)(v209 + 48);
          v98 = v209 + 48;
          if ( v97 )
          {
            sub_161E7C0(v209 + 48, v97);
            v96 = v209;
          }
          v99 = v211;
          *(_QWORD *)(v96 + 48) = v211;
          if ( v99 )
          {
            v210 = v96;
            sub_1623210((__int64)&v211, v99, v98);
            v96 = v210;
          }
        }
        v206 = v96;
      }
      else
      {
        v206 = sub_15A3890((__int64 *)v206, v79, v86, 0);
      }
      v214[0] = (unsigned __int8 *)"else";
      v215 = 259;
      v87 = sub_157FBF0(v71, v193, (__int64)v214);
      v218 = v193;
      v198 = v87;
      v88 = (unsigned __int8 *)a1[6];
      v89 = (_QWORD *)a1[5];
      v214[0] = v88;
      v217 = v89;
      if ( v88 )
      {
        sub_1623A60((__int64)v214, (__int64)v88, 2);
        v27 = v216;
        if ( !v216 )
          goto LABEL_26;
      }
      else
      {
        v27 = v216;
        if ( !v216 )
          goto LABEL_28;
      }
      sub_161E7C0((__int64)&v216, (__int64)v27);
LABEL_26:
      v216 = v214[0];
      if ( v214[0] )
        sub_1623210((__int64)v214, v214[0], (__int64)&v216);
LABEL_28:
      v28 = (_QWORD *)sub_157EBA0((__int64)v204);
      v29 = sub_1648A60(56, 3u);
      if ( v29 )
        sub_15F83E0((__int64)v29, (__int64)v71, v198, (__int64)v69, (__int64)v28);
      sub_15F20C0(v28);
      if ( v188 == v26 )
      {
        v201 = v71;
        v9 = a1;
        goto LABEL_134;
      }
      if ( (_DWORD)v26 != -1 )
      {
        v212[0] = (__int64)"res.phi.else";
        v213 = 259;
        v215 = 257;
        v30 = sub_1648B60(64);
        v31 = v30;
        if ( v30 )
        {
          v32 = v30;
          sub_15F1EA0(v30, v187, 53, 0, 0, 0);
          *(_DWORD *)(v31 + 56) = 2;
          sub_164B780(v31, (__int64 *)v214);
          sub_1648880(v31, *(_DWORD *)(v31 + 56), 1);
        }
        else
        {
          v32 = 0;
        }
        if ( v217 )
        {
          v185 = v218;
          sub_157E9D0((__int64)(v217 + 5), v31);
          v33 = *v185;
          v34 = *(_QWORD *)(v31 + 24) & 7LL;
          *(_QWORD *)(v31 + 32) = v185;
          v33 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v31 + 24) = v33 | v34;
          *(_QWORD *)(v33 + 8) = v31 + 24;
          *v185 = *v185 & 7 | (v31 + 24);
        }
        sub_164B780(v32, v212);
        v39 = (__int64)v216;
        if ( v216 )
        {
          v211 = v216;
          sub_1623A60((__int64)&v211, (__int64)v216, 2);
          v40 = *(_QWORD *)(v31 + 48);
          v37 = &v211;
          v35 = v31 + 48;
          if ( v40 )
          {
            sub_161E7C0(v31 + 48, v40);
            v37 = &v211;
            v35 = v31 + 48;
          }
          v39 = (__int64)v211;
          *(_QWORD *)(v31 + 48) = v211;
          if ( v39 )
            sub_1623210((__int64)&v211, (unsigned __int8 *)v39, v35);
        }
        v41 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        if ( v41 == *(_DWORD *)(v31 + 56) )
        {
          sub_15F55D0(v31, v39, v35, v36, (__int64)v37, v38);
          v41 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        }
        v42 = (v41 + 1) & 0xFFFFFFF;
        v43 = v42 | *(_DWORD *)(v31 + 20) & 0xF0000000;
        *(_DWORD *)(v31 + 20) = v43;
        if ( (v43 & 0x40000000) != 0 )
          v44 = *(_QWORD *)(v31 - 8);
        else
          v44 = v32 - 24 * v42;
        v45 = (__int64 *)(v44 + 24LL * (unsigned int)(v42 - 1));
        if ( *v45 )
        {
          v46 = v45[1];
          v47 = v45[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v47 = v46;
          if ( v46 )
          {
            v39 = *(_QWORD *)(v46 + 16) & 3LL;
            *(_QWORD *)(v46 + 16) = v39 | v47;
          }
        }
        *v45 = v206;
        if ( v206 )
        {
          v48 = *(_QWORD *)(v206 + 8);
          v39 = v206 + 8;
          v45[1] = v48;
          if ( v48 )
            *(_QWORD *)(v48 + 16) = (unsigned __int64)(v45 + 1) | *(_QWORD *)(v48 + 16) & 3LL;
          v45[2] = v39 | v45[2] & 3;
          *(_QWORD *)(v206 + 8) = v45;
        }
        v49 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        v50 = (unsigned int)(v49 - 1);
        if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
          v51 = *(_QWORD *)(v31 - 8);
        else
          v51 = v32 - 24 * v49;
        v52 = 3LL * *(unsigned int *)(v31 + 56);
        *(_QWORD *)(v51 + 8 * v50 + 24LL * *(unsigned int *)(v31 + 56) + 8) = v71;
        v53 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        if ( v53 == *(_DWORD *)(v31 + 56) )
        {
          sub_15F55D0(v31, v39, v52, v51, (__int64)v37, v38);
          v53 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        }
        v54 = (v53 + 1) & 0xFFFFFFF;
        v55 = v54 | *(_DWORD *)(v31 + 20) & 0xF0000000;
        *(_DWORD *)(v31 + 20) = v55;
        if ( (v55 & 0x40000000) != 0 )
          v56 = *(_QWORD *)(v31 - 8);
        else
          v56 = v32 - 24 * v54;
        v57 = (__int64 *)(v56 + 24LL * (unsigned int)(v54 - 1));
        if ( *v57 )
        {
          v58 = v57[1];
          v59 = v57[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v59 = v58;
          if ( v58 )
            *(_QWORD *)(v58 + 16) = *(_QWORD *)(v58 + 16) & 3LL | v59;
        }
        *v57 = v191;
        if ( v191 )
        {
          v60 = *(_QWORD *)(v191 + 8);
          v57[1] = v60;
          if ( v60 )
            *(_QWORD *)(v60 + 16) = (unsigned __int64)(v57 + 1) | *(_QWORD *)(v60 + 16) & 3LL;
          v57[2] = (v191 + 8) | v57[2] & 3;
          *(_QWORD *)(v191 + 8) = v57;
        }
        v61 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
          v62 = *(_QWORD *)(v31 - 8);
        else
          v62 = v32 - 24 * v61;
        v191 = v31;
        v206 = v31;
        *(_QWORD *)(v62 + 8LL * (unsigned int)(v61 - 1) + 24LL * *(unsigned int *)(v31 + 56) + 8) = v204;
      }
      ++v26;
      v204 = (_QWORD *)v198;
    }
  }
  v201 = 0;
  v191 = v206;
LABEL_134:
  v212[0] = (__int64)"res.phi.select";
  v213 = 259;
  v215 = 257;
  v115 = sub_1648B60(64);
  v116 = v115;
  if ( v115 )
  {
    v117 = v115;
    sub_15F1EA0(v115, v187, 53, 0, 0, 0);
    *(_DWORD *)(v116 + 56) = 2;
    sub_164B780(v116, (__int64 *)v214);
    sub_1648880(v116, *(_DWORD *)(v116 + 56), 1);
  }
  else
  {
    v117 = 0;
  }
  if ( v217 )
  {
    v118 = v218;
    sub_157E9D0((__int64)(v217 + 5), v116);
    v119 = *(_QWORD *)(v116 + 24);
    v120 = *v118;
    *(_QWORD *)(v116 + 32) = v118;
    v120 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v116 + 24) = v120 | v119 & 7;
    *(_QWORD *)(v120 + 8) = v116 + 24;
    *v118 = *v118 & 7 | (v116 + 24);
  }
  sub_164B780(v117, v212);
  v125 = (__int64)v216;
  if ( v216 )
  {
    v211 = v216;
    sub_1623A60((__int64)&v211, (__int64)v216, 2);
    v126 = *(_QWORD *)(v116 + 48);
    v121 = v116 + 48;
    if ( v126 )
    {
      sub_161E7C0(v116 + 48, v126);
      v121 = v116 + 48;
    }
    v125 = (__int64)v211;
    *(_QWORD *)(v116 + 48) = v211;
    if ( v125 )
      sub_1623210((__int64)&v211, (unsigned __int8 *)v125, v121);
  }
  v127 = *(_DWORD *)(v116 + 20) & 0xFFFFFFF;
  if ( v127 == *(_DWORD *)(v116 + 56) )
  {
    sub_15F55D0(v116, v125, v121, v122, v123, v124);
    v127 = *(_DWORD *)(v116 + 20) & 0xFFFFFFF;
  }
  v128 = (v127 + 1) & 0xFFFFFFF;
  v129 = v128 | *(_DWORD *)(v116 + 20) & 0xF0000000;
  *(_DWORD *)(v116 + 20) = v129;
  if ( (v129 & 0x40000000) != 0 )
    v130 = *(_QWORD *)(v116 - 8);
  else
    v130 = v117 - 24 * v128;
  v131 = (__int64 *)(v130 + 24LL * (unsigned int)(v128 - 1));
  if ( *v131 )
  {
    v132 = v131[1];
    v133 = v131[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v133 = v132;
    if ( v132 )
    {
      v125 = *(_QWORD *)(v132 + 16) & 3LL;
      *(_QWORD *)(v132 + 16) = v125 | v133;
    }
  }
  *v131 = v206;
  if ( v206 )
  {
    v134 = *(_QWORD *)(v206 + 8);
    v125 = v206 + 8;
    v131[1] = v134;
    if ( v134 )
      *(_QWORD *)(v134 + 16) = (unsigned __int64)(v131 + 1) | *(_QWORD *)(v134 + 16) & 3LL;
    v131[2] = v125 | v131[2] & 3;
    *(_QWORD *)(v206 + 8) = v131;
  }
  v135 = *(_DWORD *)(v116 + 20) & 0xFFFFFFF;
  v136 = (unsigned int)(v135 - 1);
  if ( (*(_BYTE *)(v116 + 23) & 0x40) != 0 )
    v137 = *(_QWORD *)(v116 - 8);
  else
    v137 = v117 - 24 * v135;
  v138 = 3LL * *(unsigned int *)(v116 + 56);
  *(_QWORD *)(v137 + 8 * v136 + 24LL * *(unsigned int *)(v116 + 56) + 8) = v201;
  v139 = *(_DWORD *)(v116 + 20) & 0xFFFFFFF;
  if ( v139 == *(_DWORD *)(v116 + 56) )
  {
    sub_15F55D0(v116, v125, v138, v137, v123, v124);
    v139 = *(_DWORD *)(v116 + 20) & 0xFFFFFFF;
  }
  v140 = (v139 + 1) & 0xFFFFFFF;
  v141 = v140 | *(_DWORD *)(v116 + 20) & 0xF0000000;
  *(_DWORD *)(v116 + 20) = v141;
  if ( (v141 & 0x40000000) != 0 )
    v142 = *(_QWORD *)(v116 - 8);
  else
    v142 = v117 - 24 * v140;
  v143 = (__int64 *)(v142 + 24LL * (unsigned int)(v140 - 1));
  if ( *v143 )
  {
    v144 = v143[1];
    v145 = v143[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v145 = v144;
    if ( v144 )
      *(_QWORD *)(v144 + 16) = *(_QWORD *)(v144 + 16) & 3LL | v145;
  }
  *v143 = v191;
  if ( v191 )
  {
    v146 = *(_QWORD *)(v191 + 8);
    v143[1] = v146;
    if ( v146 )
      *(_QWORD *)(v146 + 16) = (unsigned __int64)(v143 + 1) | *(_QWORD *)(v146 + 16) & 3LL;
    v143[2] = (v191 + 8) | v143[2] & 3;
    *(_QWORD *)(v191 + 8) = v143;
  }
  v147 = *(_DWORD *)(v116 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v116 + 23) & 0x40) != 0 )
    v148 = *(_QWORD *)(v116 - 8);
  else
    v148 = v117 - 24 * v147;
  *(_QWORD *)(v148 + 8LL * (unsigned int)(v147 - 1) + 24LL * *(unsigned int *)(v116 + 56) + 8) = v204;
  v215 = 257;
  v149 = sub_156B790((__int64 *)&v216, v195, v116, v184, (__int64)v214, 0);
  sub_164D160((__int64)v9, v149, a2, a3, a4, a5, v150, v151, a8, a9);
  result = sub_15F20C0(v9);
  if ( v216 )
    return sub_161E7C0((__int64)&v216, (__int64)v216);
  return result;
}
