// Function: sub_1C637F0
// Address: 0x1c637f0
//
__int64 __fastcall sub_1C637F0(
        __int64 a1,
        __int64 ******a2,
        __int64 a3,
        _DWORD *a4,
        int a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16)
{
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v20; // r14
  __int64 ****v21; // rbx
  __int64 v22; // r15
  __int64 v23; // r12
  __int64 **v24; // r13
  __int64 **v25; // rbx
  __int64 *v26; // r14
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r9
  __int64 v34; // r10
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rax
  bool v38; // al
  bool v39; // cl
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 result; // rax
  __int64 v43; // r13
  __int64 v44; // rax
  void (__fastcall *v45)(_QWORD, __int64); // rax
  unsigned int *v46; // rax
  __int64 **v47; // rbx
  __int64 **v48; // rax
  __int64 **v49; // r15
  __int64 **v50; // rbx
  __int64 v51; // rsi
  __int64 v52; // rsi
  __int64 v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 *v56; // r14
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // r12
  __int64 v60; // rsi
  unsigned __int64 v61; // rax
  __int64 **v62; // rdx
  __int64 v63; // rbx
  __int64 ****v64; // rbx
  __int64 ***v65; // rax
  __int64 **v66; // r12
  __int64 v67; // rdi
  unsigned int v68; // edx
  __int64 *v69; // rcx
  __int64 v70; // r8
  __int64 *v71; // rbx
  __int64 v72; // rsi
  __int64 v73; // r13
  __int64 v74; // r13
  __int64 v75; // rax
  __int64 v76; // rdi
  __int64 v77; // rsi
  __int64 v78; // rdx
  __int64 v79; // rax
  unsigned __int8 *v80; // rax
  unsigned int v81; // esi
  int v82; // r9d
  int v83; // r9d
  __int64 v84; // rsi
  int v85; // edx
  unsigned int v86; // r8d
  unsigned __int8 **v87; // r10
  unsigned int v88; // esi
  __int64 ***v89; // rcx
  __int64 v90; // r8
  unsigned int v91; // edx
  char *v92; // rax
  __int64 v93; // rdi
  __int64 v94; // rdi
  __int64 v95; // rax
  __int64 v96; // r12
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // r14
  double v102; // xmm4_8
  double v103; // xmm5_8
  __int64 v104; // rax
  __int64 v105; // rdx
  char v106; // di
  __int64 v107; // rcx
  __int64 v108; // rdx
  __int64 v109; // rsi
  __int64 v110; // rax
  __int64 v111; // rbx
  __int64 v112; // r13
  __int64 v113; // r12
  __int64 v114; // rax
  __int64 v115; // rax
  unsigned __int8 *v116; // rsi
  __int64 **v117; // rdx
  __int64 v118; // rax
  const char *v119; // rsi
  _QWORD *v120; // r12
  int v121; // eax
  __int64 v122; // rdi
  int v123; // edx
  unsigned int v124; // eax
  __int64 *******v125; // rcx
  __int64 ******v126; // rsi
  int v127; // r11d
  int v128; // edi
  int v129; // r13d
  char *v130; // r11
  int v131; // ecx
  __int64 ***v132; // rdx
  unsigned __int64 v133; // rcx
  __int64 **v134; // rsi
  __int64 v135; // rbx
  __int64 v136; // rax
  __int64 v137; // rax
  unsigned __int8 *v138; // rsi
  __int64 v139; // rax
  __int64 v140; // rax
  int v141; // edx
  __int64 v142; // rdx
  __int64 *v143; // rcx
  __int64 v144; // rdx
  unsigned __int64 v145; // rax
  __int64 v146; // rdx
  __int64 v147; // rdx
  __int64 v148; // rcx
  int v149; // r11d
  unsigned __int8 **v150; // rdi
  int v151; // ecx
  int v152; // r8d
  __int64 v153; // rax
  unsigned __int64 *v154; // r15
  __int64 v155; // rax
  unsigned __int64 v156; // rcx
  __int64 v157; // rsi
  __int64 v158; // rdx
  unsigned __int8 *v159; // rsi
  unsigned __int64 v160; // r12
  __int64 v161; // rax
  __int64 v162; // r13
  __int64 v163; // rax
  __int64 v164; // rcx
  __int64 v165; // r8
  __int64 v166; // r9
  unsigned __int64 v167; // rax
  unsigned __int8 **v168; // [rsp+8h] [rbp-170h]
  __int64 v169; // [rsp+10h] [rbp-168h]
  __int64 v170; // [rsp+20h] [rbp-158h]
  __int64 v171; // [rsp+38h] [rbp-140h]
  __int64 v172; // [rsp+40h] [rbp-138h]
  __int64 v173; // [rsp+48h] [rbp-130h]
  __int64 v174; // [rsp+50h] [rbp-128h]
  bool v175; // [rsp+50h] [rbp-128h]
  __int64 v176; // [rsp+50h] [rbp-128h]
  __int64 v177; // [rsp+50h] [rbp-128h]
  __int64 v178; // [rsp+58h] [rbp-120h]
  __int64 v179; // [rsp+58h] [rbp-120h]
  __int64 v180; // [rsp+58h] [rbp-120h]
  __int64 v181; // [rsp+58h] [rbp-120h]
  __int64 v182; // [rsp+58h] [rbp-120h]
  int v183; // [rsp+58h] [rbp-120h]
  __int64 v186; // [rsp+70h] [rbp-108h]
  __int64 v187; // [rsp+70h] [rbp-108h]
  __int64 v188; // [rsp+70h] [rbp-108h]
  __int64 v189; // [rsp+70h] [rbp-108h]
  __int64 v190; // [rsp+70h] [rbp-108h]
  __int64 v191; // [rsp+70h] [rbp-108h]
  __int64 v194; // [rsp+80h] [rbp-F8h]
  __int64 v195; // [rsp+80h] [rbp-F8h]
  __int64 v196; // [rsp+80h] [rbp-F8h]
  char v197; // [rsp+88h] [rbp-F0h]
  __int64 ****v198; // [rsp+88h] [rbp-F0h]
  __int64 v199; // [rsp+88h] [rbp-F0h]
  __int64 v200; // [rsp+88h] [rbp-F0h]
  __int64 *v201; // [rsp+90h] [rbp-E8h]
  __int64 v202; // [rsp+90h] [rbp-E8h]
  __int64 v203; // [rsp+90h] [rbp-E8h]
  __int64 v204; // [rsp+A0h] [rbp-D8h] BYREF
  __int64 v205; // [rsp+A8h] [rbp-D0h] BYREF
  const char *v206; // [rsp+B0h] [rbp-C8h] BYREF
  const char *v207; // [rsp+B8h] [rbp-C0h] BYREF
  char v208; // [rsp+C8h] [rbp-B0h]
  char v209; // [rsp+C9h] [rbp-AFh]
  unsigned __int8 *v210[2]; // [rsp+D8h] [rbp-A0h] BYREF
  __int16 v211; // [rsp+E8h] [rbp-90h]
  const char *v212; // [rsp+F8h] [rbp-80h] BYREF
  __int64 v213; // [rsp+100h] [rbp-78h]
  unsigned __int64 *v214; // [rsp+108h] [rbp-70h]
  __int64 v215; // [rsp+110h] [rbp-68h]
  __int64 v216; // [rsp+118h] [rbp-60h]
  int v217; // [rsp+120h] [rbp-58h]
  __int64 v218; // [rsp+128h] [rbp-50h]
  __int64 v219; // [rsp+130h] [rbp-48h]
  __int64 v220; // [rsp+190h] [rbp+18h]

  v171 = a3;
  v20 = a16;
  v21 = **a2;
  sub_1C620D0(
    (_QWORD *)a1,
    (unsigned int *)*v21,
    (__int64 **)v21[1],
    &v204,
    (unsigned __int64 *)&v205,
    *(_QWORD *)(a1 + 200),
    0);
  v22 = v204;
  v201 = ***v21;
  if ( *(_BYTE *)(v204 + 16) != 18 )
    goto LABEL_10;
  if ( v204 == *(_QWORD *)((***v21)[2] + 40) )
  {
    v22 = (***v21)[2];
LABEL_10:
    v197 = 1;
    goto LABEL_11;
  }
  v197 = 0;
  v22 = sub_157EBA0(v204);
  if ( *(_WORD *)(a3 + 24) )
  {
    v23 = sub_38767A0(a6, v201[1], 0, v22);
    v172 = sub_145DC80(*(_QWORD *)(a1 + 184), v23);
    v24 = **v21;
    if ( &v24[*((unsigned int *)*v21 + 2)] != v24 )
    {
      v198 = v21;
      v25 = &v24[*((unsigned int *)*v21 + 2)];
      do
      {
        v26 = *v24;
        v27 = v23;
        if ( !sub_14560B0(**v24) )
        {
          v28 = sub_13A5B00(*(_QWORD *)(a1 + 184), v172, *v26, 0, 0);
          v27 = sub_38767A0(a6, v28, 0, v26[2]);
        }
        ++v24;
        sub_1C51F30((__int64 ***)v26[3], v27, v26[2]);
      }
      while ( v24 != v25 );
      v21 = v198;
      v20 = a16;
    }
    v169 = 0;
    v194 = 0;
    v170 = 0;
    goto LABEL_154;
  }
LABEL_11:
  if ( *(_WORD *)(v201[1] + 24) == 7 )
  {
    v174 = v201[1];
    v186 = sub_13A5BC0((_QWORD *)v174, *(_QWORD *)(a1 + 184));
    v29 = (char *)a2[1] - (char *)*a2;
    v178 = *(_QWORD *)(a1 + 184);
    v30 = sub_1456040(v186);
    v31 = sub_145CF80(v178, v30, (v29 >> 3) + 1, 0);
    v32 = sub_13A5B60(*(_QWORD *)(a1 + 184), v31, v171, 0, 0);
    v33 = v186;
    v34 = v174;
    v187 = v32;
    v35 = *(_QWORD *)(v174 + 48);
    if ( v33 == v32 )
    {
      if ( v35 )
      {
        v181 = *(_QWORD *)(a1 + 184);
        v57 = sub_1456040(**(_QWORD **)(v174 + 32));
        v58 = sub_1456E10(v181, v57);
        result = sub_1BF9AF0(v35, 0, v58);
        v34 = v174;
        if ( !result && !a5 )
          return result;
      }
    }
    else
    {
      v179 = sub_1BF8840(v33, *(_QWORD **)(a1 + 184), *(_QWORD *)(v174 + 48), 1u, 0, a7, a8);
      v36 = sub_1BF8840(v187, *(_QWORD **)(a1 + 184), v35, 1u, 0, a7, a8);
      v37 = sub_14806B0(*(_QWORD *)(a1 + 184), v179, v36, 0, 0);
      v38 = sub_14560B0(v37);
      v34 = v174;
      v39 = v38;
      if ( v35 )
      {
        v175 = v38;
        v188 = v34;
        v180 = *(_QWORD *)(a1 + 184);
        v40 = sub_1456040(**(_QWORD **)(v34 + 32));
        v41 = sub_1456E10(v180, v40);
        result = sub_1BF9AF0(v35, 0, v41);
        v34 = v188;
        v39 = v175;
        if ( !a5 && !result )
          return result;
      }
      if ( !v39 )
      {
LABEL_17:
        v43 = 0;
        v169 = 0;
        v194 = 0;
        v201 = ***v21;
        v170 = 0;
        if ( !v197 )
        {
LABEL_18:
          v44 = v201[1];
LABEL_19:
          v172 = v44;
          v23 = sub_38767A0(a6, v44, 0, v22);
          goto LABEL_20;
        }
        goto LABEL_38;
      }
    }
    v189 = v34;
    v182 = **(_QWORD **)(v34 + 32);
    if ( !sub_146CEE0(*(_QWORD *)(a1 + 184), v182, v35) || !sub_146CEE0(*(_QWORD *)(a1 + 184), v171, v35) )
      goto LABEL_17;
    v170 = sub_13FC520(v35);
    if ( v170 && (v170 = sub_13F9E70(v35)) != 0 )
    {
      result = (unsigned int)(*a4 + 2);
      *a4 = result;
      if ( (int)result > a5 && a5 >= 0 )
        return result;
      v195 = v189;
      v190 = sub_13FC520(v35);
      v59 = sub_13F9E70(v35);
      v169 = sub_13FCB50(v35);
      sub_1C620D0(
        (_QWORD *)a1,
        (unsigned int *)**(a2[1] - 1),
        (__int64 **)(*(a2[1] - 1))[1],
        &v204,
        (unsigned __int64 *)&v205,
        *(_QWORD *)(a1 + 200),
        0);
      v60 = v205;
      if ( *(_BYTE *)(v205 + 16) != 18 )
        v60 = *(_QWORD *)(v205 + 40);
      if ( v59 && v169 && sub_15CC8F0(*(_QWORD *)(a1 + 200), v60, v169) && dword_4FBD1E0 > 3 )
      {
        v23 = sub_1649C60(v201[3]);
        if ( *(_BYTE *)(v23 + 16) == 77 )
        {
          v194 = *v201;
        }
        else
        {
          v160 = sub_157EBA0(v190);
          v177 = v195;
          v161 = sub_1456040(**(_QWORD **)(v195 + 32));
          v203 = sub_38767A0(a6, v182, v161, v160);
          v196 = sub_157ED20(**(_QWORD **)(v35 + 32));
          v212 = "baseValue";
          LOWORD(v214) = 259;
          v162 = sub_1456040(**(_QWORD **)(v177 + 32));
          v163 = sub_1648B60(64);
          v23 = v163;
          if ( v163 )
          {
            sub_15F1EA0(v163, v162, 53, 0, 0, v196);
            *(_DWORD *)(v23 + 56) = 2;
            sub_164B780(v23, (__int64 *)&v212);
            sub_1648880(v23, *(_DWORD *)(v23 + 56), 1);
          }
          sub_1704F80(v23, v203, v190, v164, v165, v166);
          v194 = 0;
        }
        v172 = sub_145DC80(*(_QWORD *)(a1 + 184), v23);
        v167 = sub_157EBA0(v190);
        v43 = sub_38767A0(a6, v171, 0, v167);
        if ( v172 )
        {
          v170 = v23;
          goto LABEL_20;
        }
        v170 = v23;
        v201 = ***v21;
      }
      else
      {
        v61 = sub_157EBA0(v190);
        v169 = 0;
        v194 = 0;
        v43 = sub_38767A0(a6, v171, 0, v61);
        v170 = 0;
        v201 = ***v21;
      }
    }
    else
    {
      v43 = 0;
      v169 = 0;
      v194 = 0;
      v201 = ***v21;
    }
  }
  else
  {
    v169 = 0;
    v43 = 0;
    v194 = 0;
    v170 = 0;
  }
  if ( !v197 )
    goto LABEL_18;
LABEL_38:
  if ( !sub_14560B0(*v201) )
  {
    v153 = sub_145DC80(*(_QWORD *)(a1 + 184), v201[3]);
    v44 = sub_14806B0(*(_QWORD *)(a1 + 184), v153, *v201, 0, 0);
    goto LABEL_19;
  }
  v23 = v201[3];
  v172 = sub_145DC80(*(_QWORD *)(a1 + 184), v23);
LABEL_20:
  if ( !v43 )
  {
LABEL_154:
    v133 = v205;
    if ( *(_BYTE *)(v205 + 16) == 18 )
    {
      if ( v205 == *(_QWORD *)((**v21[1])[2] + 40) )
        v133 = (**v21[1])[2];
      else
        v133 = sub_157EBA0(v205);
    }
    v43 = sub_38767A0(a6, v171, 0, v133);
  }
  if ( dword_4FBD1E0 > 5 )
  {
    v45 = *(void (__fastcall **)(_QWORD, __int64))(a1 + 216);
    if ( v45 )
      v45(*(_QWORD *)(a1 + 224), v43);
  }
  if ( *(_WORD *)(v171 + 24) )
    v171 = sub_145DC80(*(_QWORD *)(a1 + 184), v43);
  if ( !v170 )
  {
    sub_1C538E0(a15, (__int64 *)**a2)[1] = v23;
    v62 = (__int64 **)*a2;
    if ( *a2 != a2[1] )
      goto LABEL_59;
    goto LABEL_119;
  }
  v46 = (unsigned int *)*v21;
  v47 = **v21;
  v48 = &v47[v46[2]];
  if ( v47 == v48 )
    goto LABEL_58;
  v199 = v22;
  v49 = v47;
  v50 = v48;
  v220 = v20;
  do
  {
    v56 = *v49;
    if ( v194 )
    {
      v51 = v23;
      if ( **v49 == v194 )
        goto LABEL_32;
      v52 = v172;
      v53 = *(_QWORD *)(a1 + 184);
      v54 = sub_14806B0(v53, **v49, v194, 0, 0);
      goto LABEL_31;
    }
    v51 = v23;
    if ( !sub_14560B0(**v49) )
    {
      v54 = *v56;
      v52 = v172;
      v53 = *(_QWORD *)(a1 + 184);
LABEL_31:
      v55 = sub_13A5B00(v53, v52, v54, 0, 0);
      v51 = sub_38767A0(a6, v55, 0, v56[2]);
    }
LABEL_32:
    ++v49;
    sub_1C51F30((__int64 ***)v56[3], v51, v56[2]);
  }
  while ( v50 != v49 );
  v22 = v199;
  v20 = v220;
LABEL_58:
  sub_1C538E0(a15, (__int64 *)**a2)[1] = v23;
  v62 = (__int64 **)*a2;
  if ( a2[1] == *a2 )
  {
    v95 = sub_13A5B00(*(_QWORD *)(a1 + 184), v172, v171, 0, 0);
LABEL_90:
    v96 = v95;
    if ( *(_QWORD *)(v22 + 40) != v169 )
      v22 = sub_157ED20(v169);
    v101 = sub_38767A0(a6, v96, 0, v22);
    v104 = *(_DWORD *)(v170 + 20) & 0xFFFFFFF;
    if ( (_DWORD)v104 == 1 )
    {
      v134 = *(__int64 ***)v101;
      if ( *(_QWORD *)v170 != *(_QWORD *)v101 && *(_BYTE *)(v101 + 16) > 0x17u )
      {
        v135 = *(_QWORD *)(v101 + 32);
        if ( v135 == *(_QWORD *)(v101 + 40) + 40LL || !v135 )
        {
          v16 = sub_16498A0(0);
          v212 = 0;
          v214 = 0;
          v215 = v16;
          v216 = 0;
          v217 = 0;
          v218 = 0;
          v219 = 0;
          v213 = 0;
          BUG();
        }
        v136 = sub_16498A0(v135 - 24);
        v212 = 0;
        v215 = v136;
        v216 = 0;
        v217 = 0;
        v218 = 0;
        v219 = 0;
        v137 = *(_QWORD *)(v135 + 16);
        v214 = (unsigned __int64 *)v135;
        v213 = v137;
        v138 = *(unsigned __int8 **)(v135 + 24);
        v210[0] = v138;
        if ( v138 )
        {
          sub_1623A60((__int64)v210, (__int64)v138, 2);
          if ( v212 )
            sub_161E7C0((__int64)&v212, (__int64)v212);
          v212 = (const char *)v210[0];
          if ( v210[0] )
            sub_1623210((__int64)v210, v210[0], (__int64)&v212);
        }
        v210[0] = "bitCastEnd";
        v211 = 259;
        v139 = sub_12AA3B0((__int64 *)&v212, 0x2Fu, v101, *(_QWORD *)v170, (__int64)v210);
        v134 = (__int64 **)v212;
        v101 = v139;
        if ( v212 )
          sub_161E7C0((__int64)&v212, (__int64)v212);
        LODWORD(v104) = *(_DWORD *)(v170 + 20) & 0xFFFFFFF;
      }
      if ( (_DWORD)v104 == *(_DWORD *)(v170 + 56) )
      {
        sub_15F55D0(v170, (__int64)v134, v97, v98, v99, v100);
        LODWORD(v104) = *(_DWORD *)(v170 + 20) & 0xFFFFFFF;
      }
      v140 = ((_DWORD)v104 + 1) & 0xFFFFFFF;
      v141 = v140 | *(_DWORD *)(v170 + 20) & 0xF0000000;
      *(_DWORD *)(v170 + 20) = v141;
      if ( (v141 & 0x40000000) != 0 )
        v142 = *(_QWORD *)(v170 - 8);
      else
        v142 = v170 - 24 * v140;
      v143 = (__int64 *)(v142 + 24LL * (unsigned int)(v140 - 1));
      if ( *v143 )
      {
        v144 = v143[1];
        v145 = v143[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v145 = v144;
        if ( v144 )
          *(_QWORD *)(v144 + 16) = *(_QWORD *)(v144 + 16) & 3LL | v145;
      }
      *v143 = v101;
      if ( v101 )
      {
        v146 = *(_QWORD *)(v101 + 8);
        v143[1] = v146;
        if ( v146 )
          *(_QWORD *)(v146 + 16) = (unsigned __int64)(v143 + 1) | *(_QWORD *)(v146 + 16) & 3LL;
        v143[2] = v143[2] & 3 | (v101 + 8);
        *(_QWORD *)(v101 + 8) = v143;
      }
      v147 = *(_DWORD *)(v170 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v170 + 23) & 0x40) != 0 )
        v148 = *(_QWORD *)(v170 - 8);
      else
        v148 = v170 - 24 * v147;
      *(_QWORD *)(v148 + 8LL * (unsigned int)(v147 - 1) + 24LL * *(unsigned int *)(v170 + 56) + 8) = v169;
    }
    else
    {
      v105 = 0x17FFFFFFE8LL;
      v106 = *(_BYTE *)(v170 + 23) & 0x40;
      if ( (_DWORD)v104 )
      {
        v107 = 24LL * *(unsigned int *)(v170 + 56) + 8;
        v108 = 0;
        do
        {
          v109 = v170 - 24LL * (unsigned int)v104;
          if ( v106 )
            v109 = *(_QWORD *)(v170 - 8);
          if ( *(_QWORD *)(v109 + v107) == v169 )
          {
            v105 = 24 * v108;
            goto LABEL_100;
          }
          ++v108;
          v107 += 8;
        }
        while ( (_DWORD)v104 != (_DWORD)v108 );
        v105 = 0x17FFFFFFE8LL;
      }
LABEL_100:
      if ( v106 )
        v110 = *(_QWORD *)(v170 - 8);
      else
        v110 = v170 - 24 * v104;
      v111 = *(_QWORD *)(v110 + v105);
      if ( v101 != v111 )
      {
        v112 = *(_QWORD *)(a1 + 200);
        while ( 1 )
        {
          if ( *(_QWORD *)v111 == *(_QWORD *)v101 || *(_BYTE *)(v101 + 16) <= 0x17u )
          {
            v120 = (_QWORD *)v101;
          }
          else
          {
            v113 = *(_QWORD *)(v101 + 32);
            if ( v113 == *(_QWORD *)(v101 + 40) + 40LL || !v113 )
            {
              v17 = sub_16498A0(0);
              v212 = 0;
              v214 = 0;
              v215 = v17;
              v216 = 0;
              v217 = 0;
              v218 = 0;
              v219 = 0;
              v213 = 0;
              BUG();
            }
            v114 = sub_16498A0(v113 - 24);
            v212 = 0;
            v215 = v114;
            v216 = 0;
            v217 = 0;
            v218 = 0;
            v219 = 0;
            v115 = *(_QWORD *)(v113 + 16);
            v214 = (unsigned __int64 *)v113;
            v213 = v115;
            v116 = *(unsigned __int8 **)(v113 + 24);
            v210[0] = v116;
            if ( v116 )
            {
              sub_1623A60((__int64)v210, (__int64)v116, 2);
              if ( v212 )
                sub_161E7C0((__int64)&v212, (__int64)v212);
              v212 = (const char *)v210[0];
              if ( v210[0] )
                sub_1623210((__int64)v210, v210[0], (__int64)&v212);
            }
            v209 = 1;
            v207 = "bitCastEnd";
            v208 = 3;
            v117 = *(__int64 ***)v111;
            if ( *(_QWORD *)v111 == *(_QWORD *)v101 )
            {
              v119 = v212;
              v120 = (_QWORD *)v101;
            }
            else if ( *(_BYTE *)(v101 + 16) > 0x10u )
            {
              v211 = 257;
              v120 = (_QWORD *)sub_15FDBD0(47, v101, (__int64)v117, (__int64)v210, 0);
              if ( v213 )
              {
                v154 = v214;
                sub_157E9D0(v213 + 40, (__int64)v120);
                v155 = v120[3];
                v156 = *v154;
                v120[4] = v154;
                v156 &= 0xFFFFFFFFFFFFFFF8LL;
                v120[3] = v156 | v155 & 7;
                *(_QWORD *)(v156 + 8) = v120 + 3;
                *v154 = *v154 & 7 | (unsigned __int64)(v120 + 3);
              }
              sub_164B780((__int64)v120, (__int64 *)&v207);
              if ( !v212 )
                goto LABEL_118;
              v206 = v212;
              sub_1623A60((__int64)&v206, (__int64)v212, 2);
              v157 = v120[6];
              v158 = (__int64)(v120 + 6);
              if ( v157 )
              {
                sub_161E7C0((__int64)(v120 + 6), v157);
                v158 = (__int64)(v120 + 6);
              }
              v159 = (unsigned __int8 *)v206;
              v120[6] = v206;
              if ( v159 )
                sub_1623210((__int64)&v206, v159, v158);
              v119 = v212;
            }
            else
            {
              v118 = sub_15A46C0(47, (__int64 ***)v101, v117, 0);
              v119 = v212;
              v120 = (_QWORD *)v118;
            }
            if ( v119 )
              sub_161E7C0((__int64)&v212, (__int64)v119);
          }
LABEL_118:
          sub_164D160(v111, (__int64)v120, (__m128)a7, *(double *)a8.m128i_i64, a9, a10, v102, v103, a13, a14);
          if ( *(_BYTE *)(v111 + 16) == 71 )
          {
            v111 = *(_QWORD *)(v111 - 24);
            if ( *(_BYTE *)(v111 + 16) > 0x17u )
            {
              if ( sub_15CCEE0(v112, v101, v111) )
                continue;
            }
          }
          goto LABEL_119;
        }
      }
    }
    goto LABEL_119;
  }
LABEL_59:
  v63 = 0;
  v183 = 0;
  v176 = v171;
  while ( 2 )
  {
    v173 = v63;
    sub_1C620D0(
      (_QWORD *)a1,
      (unsigned int *)*v62[v63],
      (__int64 **)v62[v63][1],
      &v204,
      (unsigned __int64 *)&v205,
      *(_QWORD *)(a1 + 200),
      0);
    v202 = 0;
    v191 = sub_13A5B00(*(_QWORD *)(a1 + 184), v172, v176, 0, 0);
    v64 = (*a2)[v63];
    v65 = v64[1];
    v66 = *v65;
    v200 = (__int64)&(*v65)[*((unsigned int *)v65 + 2)];
    if ( *v65 != (__int64 **)v200 )
    {
      while ( 1 )
      {
        v22 = v205;
        v71 = *v66;
        if ( *(_BYTE *)(v205 + 16) == 18 )
        {
          if ( v205 == *(_QWORD *)(v71[2] + 40) )
            v22 = v71[2];
          else
            v22 = sub_157EBA0(v205);
        }
        if ( !v202 )
          v202 = sub_38767A0(a6, v191, 0, v22);
        v72 = *v71;
        if ( v194 )
          break;
        v73 = v202;
        if ( !sub_14560B0(v72) )
        {
          v78 = *v71;
          v77 = v191;
          v76 = *(_QWORD *)(a1 + 184);
LABEL_72:
          v79 = sub_13A5B00(v76, v77, v78, 0, 0);
          v73 = sub_38767A0(a6, v79, 0, v71[2]);
        }
LABEL_73:
        v80 = (unsigned __int8 *)v71[2];
        v81 = *(_DWORD *)(v20 + 24);
        v210[0] = v80;
        if ( !v81 )
        {
          ++*(_QWORD *)v20;
          goto LABEL_75;
        }
        v67 = *(_QWORD *)(v20 + 8);
        v68 = (v81 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
        v69 = (__int64 *)(v67 + 8LL * v68);
        v70 = *v69;
        if ( v80 != (unsigned __int8 *)*v69 )
        {
          v127 = 1;
          v87 = 0;
          while ( v70 != -8 )
          {
            if ( v87 || v70 != -16 )
              v69 = (__int64 *)v87;
            v68 = (v81 - 1) & (v127 + v68);
            v70 = *(_QWORD *)(v67 + 8LL * v68);
            if ( v80 == (unsigned __int8 *)v70 )
              goto LABEL_63;
            ++v127;
            v87 = (unsigned __int8 **)v69;
            v69 = (__int64 *)(v67 + 8LL * v68);
          }
          v128 = *(_DWORD *)(v20 + 16);
          if ( !v87 )
            v87 = (unsigned __int8 **)v69;
          ++*(_QWORD *)v20;
          v85 = v128 + 1;
          if ( 4 * (v128 + 1) >= 3 * v81 )
          {
LABEL_75:
            sub_1467110(v20, 2 * v81);
            v82 = *(_DWORD *)(v20 + 24);
            if ( !v82 )
            {
              ++*(_DWORD *)(v20 + 16);
              BUG();
            }
            v83 = v82 - 1;
            v84 = *(_QWORD *)(v20 + 8);
            v85 = *(_DWORD *)(v20 + 16) + 1;
            v86 = v83 & ((LODWORD(v210[0]) >> 9) ^ (LODWORD(v210[0]) >> 4));
            v87 = (unsigned __int8 **)(v84 + 8LL * v86);
            v80 = *v87;
            if ( v210[0] != *v87 )
            {
              v149 = 1;
              v150 = 0;
              while ( v80 != (unsigned __int8 *)-8LL )
              {
                if ( v150 || v80 != (unsigned __int8 *)-16LL )
                  v87 = v150;
                v86 = v83 & (v149 + v86);
                v168 = (unsigned __int8 **)(v84 + 8LL * v86);
                v80 = *v168;
                if ( v210[0] == *v168 )
                {
                  v87 = (unsigned __int8 **)(v84 + 8LL * v86);
                  goto LABEL_77;
                }
                ++v149;
                v150 = v87;
                v87 = (unsigned __int8 **)(v84 + 8LL * v86);
              }
              v80 = v210[0];
              if ( v150 )
                v87 = v150;
            }
          }
          else if ( v81 - *(_DWORD *)(v20 + 20) - v85 <= v81 >> 3 )
          {
            sub_1467110(v20, v81);
            sub_1463A20(v20, (__int64 *)v210, &v212);
            v87 = (unsigned __int8 **)v212;
            v80 = v210[0];
            v85 = *(_DWORD *)(v20 + 16) + 1;
          }
LABEL_77:
          *(_DWORD *)(v20 + 16) = v85;
          if ( *v87 != (unsigned __int8 *)-8LL )
            --*(_DWORD *)(v20 + 20);
          *v87 = v80;
          v70 = v71[2];
        }
LABEL_63:
        ++v66;
        sub_1C51F30((__int64 ***)v71[3], v73, v70);
        if ( (__int64 **)v200 == v66 )
        {
          v64 = (*a2)[v173];
          goto LABEL_83;
        }
      }
      v73 = v202;
      if ( v72 == v194 )
        goto LABEL_73;
      v74 = *(_QWORD *)(a1 + 184);
      v75 = sub_14806B0(v74, v72, v194, 0, 0);
      v76 = v74;
      v77 = v191;
      v78 = v75;
      goto LABEL_72;
    }
LABEL_83:
    v88 = *(_DWORD *)(a15 + 24);
    if ( v88 )
    {
      v89 = v64[1];
      v90 = *(_QWORD *)(a15 + 8);
      v91 = (v88 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
      v92 = (char *)(v90 + 16LL * v91);
      v93 = *(_QWORD *)v92;
      if ( v89 == *(__int64 ****)v92 )
        goto LABEL_85;
      v129 = 1;
      v130 = 0;
      while ( v93 != -8 )
      {
        if ( !v130 && v93 == -16 )
          v130 = v92;
        v91 = (v88 - 1) & (v129 + v91);
        v92 = (char *)(v90 + 16LL * v91);
        v93 = *(_QWORD *)v92;
        if ( v89 == *(__int64 ****)v92 )
          goto LABEL_85;
        ++v129;
      }
      if ( v130 )
        v92 = v130;
      ++*(_QWORD *)a15;
      v131 = *(_DWORD *)(a15 + 16) + 1;
      if ( 4 * v131 < 3 * v88 )
      {
        if ( v88 - *(_DWORD *)(a15 + 20) - v131 <= v88 >> 3 )
        {
          sub_1C53730(a15, v88);
          sub_1C502F0(a15, (__int64 *)v64 + 1, &v212);
          v92 = (char *)v212;
          v131 = *(_DWORD *)(a15 + 16) + 1;
        }
        goto LABEL_141;
      }
    }
    else
    {
      ++*(_QWORD *)a15;
    }
    sub_1C53730(a15, 2 * v88);
    sub_1C502F0(a15, (__int64 *)v64 + 1, &v212);
    v92 = (char *)v212;
    v131 = *(_DWORD *)(a15 + 16) + 1;
LABEL_141:
    *(_DWORD *)(a15 + 16) = v131;
    if ( *(_QWORD *)v92 != -8 )
      --*(_DWORD *)(a15 + 20);
    v132 = v64[1];
    *((_QWORD *)v92 + 1) = 0;
    *(_QWORD *)v92 = v132;
LABEL_85:
    *((_QWORD *)v92 + 1) = v202;
    v94 = *(_QWORD *)(a1 + 184);
    if ( *(_WORD *)(v171 + 24) )
      v172 = sub_145DC80(v94, v202);
    else
      v176 = sub_13A5B00(v94, v176, v171, 0, 0);
    v63 = (unsigned int)++v183;
    v62 = (__int64 **)*a2;
    if ( v183 != a2[1] - *a2 )
      continue;
    break;
  }
  if ( v170 )
  {
    v95 = sub_13A5B00(*(_QWORD *)(a1 + 184), v172, v176, 0, 0);
    goto LABEL_90;
  }
LABEL_119:
  v121 = *(_DWORD *)(a1 + 256);
  if ( v121 )
  {
    v122 = *(_QWORD *)(a1 + 240);
    v123 = v121 - 1;
    v124 = (v121 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v125 = (__int64 *******)(v122 + 8LL * v124);
    v126 = *v125;
    if ( a2 == *v125 )
    {
LABEL_121:
      *v125 = (__int64 ******)-16LL;
      --*(_DWORD *)(a1 + 248);
      ++*(_DWORD *)(a1 + 252);
    }
    else
    {
      v151 = 1;
      while ( v126 != (__int64 ******)-8LL )
      {
        v152 = v151 + 1;
        v124 = v123 & (v151 + v124);
        v125 = (__int64 *******)(v122 + 8LL * v124);
        v126 = *v125;
        if ( *v125 == a2 )
          goto LABEL_121;
        v151 = v152;
      }
    }
  }
  if ( *a2 )
    j_j___libc_free_0(*a2, (char *)a2[2] - (char *)*a2);
  return j_j___libc_free_0(a2, 24);
}
