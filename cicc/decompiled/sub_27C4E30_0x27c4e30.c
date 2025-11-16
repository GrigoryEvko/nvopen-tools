// Function: sub_27C4E30
// Address: 0x27c4e30
//
__int64 __fastcall sub_27C4E30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _BYTE *a4,
        bool a5,
        __int64 *a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  _QWORD *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r12
  __int64 v18; // rdx
  unsigned __int8 v19; // dl
  __int64 v20; // rdi
  __int64 *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r15
  __int64 v26; // rcx
  _BYTE *v27; // rdi
  __int64 v28; // r13
  __int64 *v29; // rax
  unsigned int v30; // esi
  __int64 *v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // r12
  __int64 v35; // r13
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // rax
  int v38; // edx
  unsigned __int64 v39; // rax
  bool v40; // cf
  __int64 v41; // rdx
  __int64 *v42; // r12
  __int64 *v43; // rbx
  unsigned __int16 v44; // ax
  char v45; // dl
  unsigned __int8 *v46; // r12
  __int64 v47; // rdx
  unsigned __int64 *v48; // rbx
  __int64 v49; // rdx
  unsigned __int64 *v50; // r12
  __int64 v51; // r9
  int v52; // eax
  unsigned __int64 *v53; // rdi
  unsigned __int64 v54; // rax
  __int64 *v55; // rdx
  __int64 *v56; // rax
  __int64 v58; // r13
  const char *v59; // rcx
  unsigned __int64 v60; // r8
  _QWORD *v61; // r13
  __int64 v62; // r9
  __int64 *v63; // r15
  __int64 *v64; // rbx
  __int64 v65; // r12
  __int64 v66; // rax
  _QWORD *v67; // rax
  _QWORD *v68; // rax
  __int64 v69; // r9
  __int64 v70; // rdx
  _BYTE *v71; // r12
  __int64 v72; // r13
  __int64 *v73; // rax
  __int64 *v74; // rdx
  __int64 *v75; // rax
  __int64 v76; // rax
  _QWORD *v77; // rax
  __int64 v78; // rbx
  unsigned __int64 v79; // rax
  int v80; // edx
  __int64 v81; // rsi
  __int64 v82; // rax
  unsigned __int64 v83; // rax
  int v84; // edx
  __int64 v85; // rdi
  __int64 v86; // rax
  __int64 v87; // rsi
  _QWORD *v88; // rax
  _QWORD *v89; // rdx
  unsigned __int64 v90; // rax
  int v91; // edx
  __int64 v92; // rbx
  unsigned __int64 v93; // rax
  __int64 v94; // r12
  unsigned __int64 v95; // rax
  const char *v96; // rax
  __int64 v97; // rdx
  _QWORD *v98; // rax
  _BYTE *v99; // rax
  __int64 v100; // rdi
  _BYTE *v101; // rdi
  __int64 v102; // rbx
  int v103; // eax
  int v104; // eax
  __int64 v105; // rbx
  __int64 *v106; // r12
  __int64 *v107; // r12
  __int64 v108; // r13
  __int64 v109; // rbx
  __int64 v110; // rax
  _QWORD *v111; // rbx
  __int64 *v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // r9
  char v116; // dl
  __int64 v117; // rax
  unsigned __int64 v118; // rdx
  _QWORD **v119; // rdx
  int v120; // ecx
  __int64 *v121; // rax
  __int64 v122; // rsi
  unsigned __int64 v123; // r13
  const char *v124; // rbx
  __int64 v125; // rdx
  unsigned int v126; // esi
  __int64 v127; // rax
  unsigned __int64 v128; // rdx
  __int64 *v129; // rax
  unsigned __int64 v130; // [rsp+8h] [rbp-2C8h]
  __int64 v131; // [rsp+10h] [rbp-2C0h]
  _QWORD *v132; // [rsp+18h] [rbp-2B8h]
  __int64 *v133; // [rsp+20h] [rbp-2B0h]
  unsigned __int64 v134; // [rsp+38h] [rbp-298h]
  __int64 *v135; // [rsp+38h] [rbp-298h]
  unsigned __int64 v136; // [rsp+38h] [rbp-298h]
  unsigned __int8 v137; // [rsp+40h] [rbp-290h]
  _QWORD *v138; // [rsp+40h] [rbp-290h]
  __int64 *v139; // [rsp+48h] [rbp-288h]
  __int64 v140; // [rsp+50h] [rbp-280h]
  unsigned __int8 v143; // [rsp+67h] [rbp-269h]
  __int64 v144; // [rsp+68h] [rbp-268h]
  __int64 v145; // [rsp+68h] [rbp-268h]
  _QWORD *v146; // [rsp+68h] [rbp-268h]
  __int64 v147; // [rsp+68h] [rbp-268h]
  __int64 v148; // [rsp+68h] [rbp-268h]
  bool v150; // [rsp+78h] [rbp-258h]
  int v151; // [rsp+78h] [rbp-258h]
  __int64 v152; // [rsp+80h] [rbp-250h]
  unsigned __int32 v153; // [rsp+80h] [rbp-250h]
  char *v154; // [rsp+80h] [rbp-250h]
  int v155; // [rsp+80h] [rbp-250h]
  unsigned __int64 v157; // [rsp+90h] [rbp-240h]
  __int64 *v158; // [rsp+98h] [rbp-238h]
  __int64 *v159; // [rsp+98h] [rbp-238h]
  __int64 v160; // [rsp+A8h] [rbp-228h]
  __m128i v161; // [rsp+B0h] [rbp-220h] BYREF
  _BYTE *v162; // [rsp+C0h] [rbp-210h]
  char v163; // [rsp+C8h] [rbp-208h]
  _QWORD v164[4]; // [rsp+D0h] [rbp-200h] BYREF
  __int16 v165; // [rsp+F0h] [rbp-1E0h]
  unsigned __int64 v166[4]; // [rsp+100h] [rbp-1D0h] BYREF
  __int16 v167; // [rsp+120h] [rbp-1B0h]
  __int64 *v168; // [rsp+130h] [rbp-1A0h] BYREF
  __int64 v169; // [rsp+138h] [rbp-198h]
  _BYTE v170[32]; // [rsp+140h] [rbp-190h] BYREF
  _QWORD *v171; // [rsp+160h] [rbp-170h] BYREF
  __int64 v172; // [rsp+168h] [rbp-168h]
  _QWORD v173[4]; // [rsp+170h] [rbp-160h] BYREF
  __int64 v174; // [rsp+190h] [rbp-140h] BYREF
  __int64 *v175; // [rsp+198h] [rbp-138h]
  __int64 v176; // [rsp+1A0h] [rbp-130h]
  int v177; // [rsp+1A8h] [rbp-128h]
  char v178; // [rsp+1ACh] [rbp-124h]
  __int64 v179; // [rsp+1B0h] [rbp-120h] BYREF
  __int64 v180; // [rsp+1D0h] [rbp-100h] BYREF
  __int64 *v181; // [rsp+1D8h] [rbp-F8h]
  __int64 v182; // [rsp+1E0h] [rbp-F0h]
  int v183; // [rsp+1E8h] [rbp-E8h]
  char v184; // [rsp+1ECh] [rbp-E4h]
  char v185; // [rsp+1F0h] [rbp-E0h] BYREF
  unsigned __int64 v186; // [rsp+210h] [rbp-C0h] BYREF
  __int64 v187; // [rsp+218h] [rbp-B8h]
  __int64 v188[2]; // [rsp+220h] [rbp-B0h] BYREF
  __int64 *v189; // [rsp+230h] [rbp-A0h]
  __int64 v190; // [rsp+240h] [rbp-90h] BYREF
  __int64 v191; // [rsp+248h] [rbp-88h]
  __int64 v192; // [rsp+250h] [rbp-80h]
  __int64 v193; // [rsp+258h] [rbp-78h]
  void **v194; // [rsp+260h] [rbp-70h]
  void **v195; // [rsp+268h] [rbp-68h]
  __int64 v196; // [rsp+270h] [rbp-60h]
  int v197; // [rsp+278h] [rbp-58h]
  __int16 v198; // [rsp+27Ch] [rbp-54h]
  char v199; // [rsp+27Eh] [rbp-52h]
  __int64 v200; // [rsp+280h] [rbp-50h]
  __int64 v201; // [rsp+288h] [rbp-48h]
  void *v202; // [rsp+290h] [rbp-40h] BYREF
  void *v203; // [rsp+298h] [rbp-38h] BYREF

  v131 = a1 + 56;
  v11 = *(_QWORD *)(a2 - 64);
  v143 = *(_BYTE *)(a1 + 84);
  if ( v143 )
  {
    v12 = *(_QWORD **)(a1 + 64);
    v13 = &v12[*(unsigned int *)(a1 + 76)];
    if ( v12 == v13 )
    {
LABEL_161:
      v143 = 0;
    }
    else
    {
      while ( v11 != *v12 )
      {
        if ( v13 == ++v12 )
          goto LABEL_161;
      }
    }
  }
  else
  {
    v143 = sub_C8CA60(v131, v11) != 0;
  }
  v14 = v173;
  v177 = 0;
  v168 = (__int64 *)v170;
  v169 = 0x400000000LL;
  v175 = &v179;
  v179 = *(_QWORD *)(a2 - 96);
  v173[0] = v179;
  v172 = 0x400000001LL;
  LODWORD(v15) = 1;
  v171 = v173;
  v176 = 0x100000004LL;
  v178 = 1;
  v174 = 1;
  while ( 1 )
  {
    v16 = (unsigned int)v15;
    LODWORD(v15) = v15 - 1;
    v17 = v14[v16 - 1];
    LODWORD(v172) = v15;
    v18 = *(_QWORD *)(v17 + 16);
    if ( !v18 || *(_QWORD *)(v18 + 8) )
      goto LABEL_7;
    v19 = *(_BYTE *)v17;
    if ( v143 )
    {
      if ( v19 <= 0x1Cu )
        goto LABEL_7;
      v20 = *(_QWORD *)(v17 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
        v20 = **(_QWORD **)(v20 + 16);
      if ( !sub_BCAC40(v20, 1) )
        goto LABEL_149;
      if ( *(_BYTE *)v17 == 58 )
        break;
      if ( *(_BYTE *)v17 != 86 )
        goto LABEL_149;
      v25 = *(_QWORD *)(v17 - 96);
      LODWORD(v15) = v172;
      v26 = *(_QWORD *)(v17 + 8);
      if ( *(_QWORD *)(v25 + 8) == v26 )
      {
        v27 = *(_BYTE **)(v17 - 64);
        if ( *v27 <= 0x15u )
        {
          v28 = *(_QWORD *)(v17 - 32);
          if ( !sub_AD7A80(v27, 1, (__int64)v21, v26, v23) )
            goto LABEL_149;
LABEL_21:
          if ( !v28 )
            goto LABEL_149;
          goto LABEL_22;
        }
      }
      goto LABEL_7;
    }
    if ( v19 <= 0x1Cu )
      goto LABEL_7;
    v100 = *(_QWORD *)(v17 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v100 + 8) - 17 <= 1 )
      v100 = **(_QWORD **)(v100 + 16);
    if ( !sub_BCAC40(v100, 1) )
    {
LABEL_149:
      if ( *(_BYTE *)v17 == 82 )
      {
        v127 = (unsigned int)v169;
        v128 = (unsigned int)v169 + 1LL;
        if ( v128 > HIDWORD(v169) )
        {
          sub_C8D5F0((__int64)&v168, v170, v128, 8u, v23, v24);
          v127 = (unsigned int)v169;
        }
        v168[v127] = v17;
        LODWORD(v169) = v169 + 1;
      }
LABEL_150:
      LODWORD(v15) = v172;
      goto LABEL_7;
    }
    if ( *(_BYTE *)v17 == 57 )
      break;
    if ( *(_BYTE *)v17 != 86 )
      goto LABEL_149;
    v25 = *(_QWORD *)(v17 - 96);
    LODWORD(v15) = v172;
    if ( *(_QWORD *)(v25 + 8) == *(_QWORD *)(v17 + 8) )
    {
      v101 = *(_BYTE **)(v17 - 32);
      if ( *v101 <= 0x15u )
      {
        v28 = *(_QWORD *)(v17 - 64);
        if ( !sub_AC30F0((__int64)v101) )
          goto LABEL_149;
        goto LABEL_21;
      }
    }
LABEL_7:
    if ( !(_DWORD)v15 )
      goto LABEL_35;
LABEL_8:
    v14 = v171;
  }
  if ( (*(_BYTE *)(v17 + 7) & 0x40) != 0 )
    v106 = *(__int64 **)(v17 - 8);
  else
    v106 = (__int64 *)(v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF));
  v25 = *v106;
  if ( !*v106 )
    goto LABEL_150;
  v28 = v106[4];
  if ( !v28 )
    goto LABEL_150;
LABEL_22:
  if ( !v178 )
    goto LABEL_176;
  v29 = v175;
  v30 = HIDWORD(v176);
  v22 = (__int64)&v175[HIDWORD(v176)];
  v21 = v175;
  if ( v175 == (__int64 *)v22 )
  {
LABEL_180:
    if ( HIDWORD(v176) < (unsigned int)v176 )
    {
      ++HIDWORD(v176);
      *(_QWORD *)v22 = v25;
      ++v174;
LABEL_182:
      v117 = (unsigned int)v172;
      v22 = HIDWORD(v172);
      v118 = (unsigned int)v172 + 1LL;
      if ( v118 > HIDWORD(v172) )
      {
        sub_C8D5F0((__int64)&v171, v173, v118, 8u, v23, v24);
        v117 = (unsigned int)v172;
      }
      v31 = v171;
      v171[v117] = v25;
      LODWORD(v172) = v172 + 1;
      if ( v178 )
        goto LABEL_185;
LABEL_178:
      sub_C8CC70((__int64)&v174, v28, (__int64)v31, v22, v23, v24);
      v15 = (unsigned int)v172;
      if ( v116 )
        goto LABEL_32;
      goto LABEL_7;
    }
LABEL_176:
    sub_C8CC70((__int64)&v174, v25, (__int64)v21, v22, v23, v24);
    if ( (_BYTE)v31 )
      goto LABEL_182;
    if ( !v178 )
      goto LABEL_178;
LABEL_185:
    v29 = v175;
    v30 = HIDWORD(v176);
  }
  else
  {
    while ( v25 != *v21 )
    {
      if ( (__int64 *)v22 == ++v21 )
        goto LABEL_180;
    }
  }
  v31 = &v29[v30];
  if ( v29 != v31 )
  {
    while ( v28 != *v29 )
    {
      if ( v31 == ++v29 )
        goto LABEL_30;
    }
    goto LABEL_150;
  }
LABEL_30:
  if ( v30 >= (unsigned int)v176 )
    goto LABEL_178;
  HIDWORD(v176) = v30 + 1;
  *v31 = v28;
  v15 = (unsigned int)v172;
  ++v174;
LABEL_32:
  if ( v15 + 1 > (unsigned __int64)HIDWORD(v172) )
  {
    sub_C8D5F0((__int64)&v171, v173, v15 + 1, 8u, v23, v24);
    v15 = (unsigned int)v172;
  }
  v171[v15] = v28;
  LODWORD(v15) = v172 + 1;
  LODWORD(v172) = v15;
  if ( (_DWORD)v15 )
    goto LABEL_8;
LABEL_35:
  v180 = 0;
  v181 = (__int64 *)&v185;
  v182 = 4;
  v183 = 0;
  v184 = 1;
  if ( !a5 )
  {
    if ( (unsigned int)v169 <= 1 )
    {
      v32 = (unsigned __int64)v168;
      v139 = &v168[(unsigned int)v169];
      if ( v139 == v168 )
      {
        v137 = 0;
        goto LABEL_77;
      }
LABEL_41:
      v158 = (__int64 *)v32;
      v137 = 0;
      v140 = a3 + 48;
      while ( 2 )
      {
        v33 = *v158;
        v150 = a5;
        if ( !a5 )
        {
          v150 = 1;
          if ( (unsigned int)(HIDWORD(v182) - v183) <= 1 )
          {
            v150 = 0;
            if ( HIDWORD(v182) - v183 == 1 )
            {
              v150 = v184;
              if ( v184 )
              {
                v74 = v181;
                v75 = &v181[HIDWORD(v182)];
                if ( v181 != v75 )
                {
                  while ( v33 != *v74 )
                  {
                    if ( v75 == ++v74 )
                      goto LABEL_44;
                  }
                  v150 = 0;
                }
              }
              else
              {
                v150 = sub_C8CA60((__int64)&v180, v33) == 0;
              }
            }
          }
        }
LABEL_44:
        v34 = *(_QWORD *)(v33 - 64);
        v35 = *(_QWORD *)(v33 - 32);
        v36 = ((unsigned __int64)((*(_BYTE *)(v33 + 1) & 2) != 0) << 32)
            | *(_WORD *)(v33 + 2) & 0x3FLL
            | v157 & 0xFFFFFF0000000000LL;
        v157 = v36;
        v37 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v140 == v37 )
        {
          v152 = 0;
        }
        else
        {
          if ( !v37 )
            BUG();
          v38 = *(unsigned __int8 *)(v37 - 24);
          v39 = v37 - 24;
          v40 = (unsigned int)(v38 - 30) < 0xB;
          v41 = 0;
          if ( v40 )
            v41 = v39;
          v152 = v41;
        }
        if ( v143 )
          v157 = (unsigned int)sub_B52870(*(_WORD *)(v33 + 2) & 0x3F) | v36 & 0xFFFFFFFF00000000LL;
        v42 = sub_DDFBA0((__int64)a6, v34, (char *)a1);
        v43 = sub_DDFBA0((__int64)a6, v35, (char *)a1);
        v44 = sub_DDCA80(a6, v157, v42, v43, v152);
        if ( HIBYTE(v44) )
        {
          v45 = v44 ^ 1;
          goto LABEL_53;
        }
        v58 = sub_D95540((__int64)v42);
        v144 = sub_D95540((__int64)a4);
        v134 = sub_D97050((__int64)a6, v58);
        if ( v134 <= sub_D97050((__int64)a6, v144) )
        {
          v136 = sub_D97050((__int64)a6, v58);
          if ( v136 < sub_D97050((__int64)a6, v144)
            && (v98 = sub_DA2C50((__int64)a6, v58, -1, 1u),
                v99 = sub_DC2B70((__int64)a6, (__int64)v98, v144, 0),
                v130 = v130 & 0xFFFFFF0000000000LL | 0x25,
                (unsigned __int8)sub_DDCB50(a6, v130, a4, v99, v152)) )
          {
            v61 = sub_DC5200((__int64)a6, (__int64)a4, v58, 0);
          }
          else
          {
            v61 = a4;
          }
        }
        else
        {
          v61 = sub_DC2B70((__int64)a6, (__int64)a4, v58, 0);
        }
        if ( v150 )
        {
          v186 = (unsigned __int64)v188;
          v187 = 0x400000000LL;
          if ( *((_WORD *)v61 + 12) == 11 )
          {
            v62 = v61[4];
            if ( v62 != v62 + 8LL * v61[5] )
            {
              v135 = v42;
              v145 = v33;
              v63 = (__int64 *)(v62 + 8LL * v61[5]);
              v133 = v43;
              v64 = (__int64 *)v61[4];
              do
              {
                v65 = *v64;
                v66 = sub_D95540(*v64);
                v67 = sub_DA2C50((__int64)a6, v66, 1, 0);
                v68 = sub_DCC810(a6, v65, (__int64)v67, 0, 0);
                v70 = (unsigned int)v187;
                v60 = (unsigned int)v187 + 1LL;
                if ( v60 > HIDWORD(v187) )
                {
                  v132 = v68;
                  sub_C8D5F0((__int64)&v186, v188, (unsigned int)v187 + 1LL, 8u, v60, v69);
                  v70 = (unsigned int)v187;
                  v68 = v132;
                }
                v59 = (const char *)v186;
                ++v64;
                *(_QWORD *)(v186 + 8 * v70) = v68;
                LODWORD(v187) = v187 + 1;
              }
              while ( v63 != v64 );
              v33 = v145;
              v42 = v135;
              v43 = v133;
            }
            v61 = sub_DCEEE0(a6, (__int64)&v186, 0, (__int64)v59, v60);
          }
          else
          {
            v76 = sub_D95540((__int64)v61);
            v77 = sub_DA2C50((__int64)a6, v76, 1, 0);
            v61 = sub_DCC810(a6, (__int64)v61, (__int64)v77, 0, 0);
          }
          if ( (__int64 *)v186 != v188 )
            _libc_free(v186);
        }
        sub_DDE7B0(&v161, a6, v157, (__int64)v42, (__int64)v43, a1, v152, (__int64)v61);
        if ( !v163 )
          goto LABEL_74;
        v71 = v162;
        v72 = v161.m128i_i64[1];
        if ( (unsigned __int8)sub_DDCB50(a6, v161.m128i_i64[0], (_BYTE *)v161.m128i_i64[1], v162, v152) )
        {
          v45 = 0;
LABEL_53:
          v46 = (unsigned __int8 *)sub_27C1150(a1, a3, v45);
LABEL_54:
          if ( *v46 > 0x1Cu )
          {
            v186 = (unsigned __int64)sub_BD5D20(v33);
            LOWORD(v189) = 773;
            v187 = v47;
            v188[0] = (__int64)".first_iter";
            sub_BD6B50(v46, (const char **)&v186);
          }
          sub_BD84D0(v33, (__int64)v46);
          v188[0] = v33;
          v186 = 6;
          v187 = 0;
          if ( v33 != -8192 && v33 != -4096 )
            sub_BD73F0((__int64)&v186);
          v48 = &v186;
          v49 = *(unsigned int *)(a8 + 8);
          v50 = *(unsigned __int64 **)a8;
          v51 = v49 + 1;
          v52 = *(_DWORD *)(a8 + 8);
          if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(a8 + 12) )
          {
            v102 = a8 + 16;
            if ( v50 > &v186 || &v186 >= &v50[3 * v49] )
            {
              v50 = (unsigned __int64 *)sub_C8D7D0(a8, a8 + 16, v49 + 1, 0x18u, v166, v51);
              sub_F17F80(a8, v50);
              v104 = v166[0];
              if ( *(_QWORD *)a8 == v102 )
              {
                *(_QWORD *)a8 = v50;
                v105 = a8;
              }
              else
              {
                v155 = v166[0];
                _libc_free(*(_QWORD *)a8);
                *(_QWORD *)a8 = v50;
                v105 = a8;
                v104 = v155;
              }
              v49 = *(unsigned int *)(v105 + 8);
              *(_DWORD *)(v105 + 12) = v104;
              v48 = &v186;
              v52 = v49;
            }
            else
            {
              v154 = (char *)((char *)&v186 - (char *)v50);
              v50 = (unsigned __int64 *)sub_C8D7D0(a8, a8 + 16, v49 + 1, 0x18u, v166, v51);
              sub_F17F80(a8, v50);
              v103 = v166[0];
              if ( v102 == *(_QWORD *)a8 )
              {
                *(_QWORD *)a8 = v50;
                *(_DWORD *)(a8 + 12) = v103;
              }
              else
              {
                v151 = v166[0];
                _libc_free(*(_QWORD *)a8);
                *(_QWORD *)a8 = v50;
                *(_DWORD *)(a8 + 12) = v151;
              }
              v49 = *(unsigned int *)(a8 + 8);
              v48 = (unsigned __int64 *)&v154[(_QWORD)v50];
              v52 = *(_DWORD *)(a8 + 8);
            }
          }
          v53 = &v50[3 * v49];
          if ( v53 )
          {
            *v53 = 6;
            v54 = v48[2];
            v53[1] = 0;
            v53[2] = v54;
            if ( v54 != 0 && v54 != -4096 && v54 != -8192 )
              sub_BD6050(v53, *v48 & 0xFFFFFFFFFFFFFFF8LL);
            v52 = *(_DWORD *)(a8 + 8);
          }
          *(_DWORD *)(a8 + 8) = v52 + 1;
          if ( v188[0] != -4096 && v188[0] != 0 && v188[0] != -8192 )
            sub_BD60C0(&v186);
          v137 = v184;
          if ( v184 )
          {
            v55 = &v181[HIDWORD(v182)];
            v56 = v181;
            if ( v181 == v55 )
              goto LABEL_102;
            while ( v33 != *v56 )
            {
              if ( v55 == ++v56 )
                goto LABEL_102;
            }
            --HIDWORD(v182);
            *v56 = v181[HIDWORD(v182)];
            ++v180;
          }
          else
          {
            v73 = sub_C8CA60((__int64)&v180, v33);
            if ( v73 )
            {
              *v73 = -2;
              ++v183;
              ++v180;
            }
LABEL_102:
            v137 = 1;
          }
LABEL_74:
          if ( v139 == ++v158 )
            goto LABEL_75;
          continue;
        }
        break;
      }
      v153 = v161.m128i_i32[0];
      v78 = sub_D4B130(a1);
      v79 = *(_QWORD *)(v78 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v78 + 48 == v79 )
      {
        v81 = 0;
      }
      else
      {
        if ( !v79 )
          BUG();
        v80 = *(unsigned __int8 *)(v79 - 24);
        v81 = 0;
        v82 = v79 - 24;
        if ( (unsigned int)(v80 - 30) < 0xB )
          v81 = v82;
      }
      sub_D5F1F0((__int64)(a7 + 65), v81);
      v146 = sub_F8DB50(a7, v72, 0);
      v138 = sub_F8DB50(a7, (__int64)v71, 0);
      v83 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v140 == v83 )
      {
        v85 = 0;
      }
      else
      {
        if ( !v83 )
          BUG();
        v84 = *(unsigned __int8 *)(v83 - 24);
        v85 = 0;
        v86 = v83 - 24;
        if ( (unsigned int)(v84 - 30) < 0xB )
          v85 = v86;
      }
      v87 = sub_B46EC0(v85, 0);
      if ( *(_BYTE *)(a1 + 84) )
      {
        v88 = *(_QWORD **)(a1 + 64);
        v89 = &v88[*(unsigned int *)(a1 + 76)];
        if ( v88 == v89 )
        {
LABEL_152:
          v153 = sub_B52870(v153);
        }
        else
        {
          while ( v87 != *v88 )
          {
            if ( v89 == ++v88 )
              goto LABEL_152;
          }
        }
      }
      else if ( !sub_C8CA60(v131, v87) )
      {
        goto LABEL_152;
      }
      v90 = *(_QWORD *)(v78 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v78 + 48 == v90 )
      {
        v94 = 0;
      }
      else
      {
        if ( !v90 )
          BUG();
        v91 = *(unsigned __int8 *)(v90 - 24);
        v92 = 0;
        v93 = v90 - 24;
        if ( (unsigned int)(v91 - 30) < 0xB )
          v92 = v93;
        v94 = v92;
      }
      v193 = sub_BD5C60(v94);
      v194 = &v202;
      v195 = &v203;
      v186 = (unsigned __int64)v188;
      v202 = &unk_49DA100;
      v187 = 0x200000000LL;
      v196 = 0;
      v203 = &unk_49DA0B0;
      v197 = 0;
      v198 = 512;
      v199 = 7;
      v200 = 0;
      v201 = 0;
      v190 = 0;
      v191 = 0;
      LOWORD(v192) = 0;
      sub_D5F1F0((__int64)&v186, v94);
      v95 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v140 == v95 )
        goto LABEL_215;
      if ( !v95 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v95 - 24) - 30 > 0xA )
LABEL_215:
        BUG();
      v96 = sub_BD5D20(*(_QWORD *)(v95 - 120));
      v165 = 261;
      v164[1] = v97;
      v164[0] = v96;
      v46 = (unsigned __int8 *)(*((__int64 (__fastcall **)(void **, _QWORD, _QWORD *, _QWORD *))*v194 + 7))(
                                 v194,
                                 v153,
                                 v146,
                                 v138);
      if ( !v46 )
      {
        v167 = 257;
        v46 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
        if ( v46 )
        {
          v119 = (_QWORD **)v146[1];
          v120 = *((unsigned __int8 *)v119 + 8);
          if ( (unsigned int)(v120 - 17) > 1 )
          {
            v122 = sub_BCB2A0(*v119);
          }
          else
          {
            BYTE4(v160) = (_BYTE)v120 == 18;
            LODWORD(v160) = *((_DWORD *)v119 + 8);
            v121 = (__int64 *)sub_BCB2A0(*v119);
            v122 = sub_BCE1B0(v121, v160);
          }
          sub_B523C0((__int64)v46, v122, 53, v153, (__int64)v146, (__int64)v138, (__int64)v166, 0, 0, 0);
        }
        (*((void (__fastcall **)(void **, unsigned __int8 *, _QWORD *, __int64, __int64))*v195 + 2))(
          v195,
          v46,
          v164,
          v191,
          v192);
        if ( v186 != v186 + 16LL * (unsigned int)v187 )
        {
          v123 = v186;
          v124 = (const char *)(v186 + 16LL * (unsigned int)v187);
          do
          {
            v125 = *(_QWORD *)(v123 + 8);
            v126 = *(_DWORD *)v123;
            v123 += 16LL;
            sub_B99FD0((__int64)v46, v126, v125);
          }
          while ( v124 != (const char *)v123 );
        }
      }
      nullsub_61();
      v202 = &unk_49DA100;
      nullsub_63();
      if ( (__int64 *)v186 != v188 )
        _libc_free(v186);
      goto LABEL_54;
    }
    if ( a4 == (_BYTE *)sub_DBA6E0((__int64)a6, a1, a3, 2) )
    {
      v107 = v168;
      v159 = &v168[(unsigned int)v169];
      if ( v159 == v168 )
        goto LABEL_196;
      do
      {
        v108 = *v107;
        sub_DB8CC0((__int64)&v186, (__int64)a6, a1, *v107, v143, 0, 0);
        v109 = v188[0];
        if ( !sub_D96A50(v188[0]) )
        {
          v147 = sub_D95540((__int64)a4);
          v110 = sub_D95540(v109);
          v148 = sub_D970B0((__int64)a6, v110, v147);
          v111 = sub_DC2CB0((__int64)a6, v109, v148);
          if ( v111 == sub_DC2CB0((__int64)a6, (__int64)a4, v148) )
          {
            if ( !v184 )
              goto LABEL_211;
            v129 = v181;
            v113 = HIDWORD(v182);
            v112 = &v181[HIDWORD(v182)];
            if ( v181 != v112 )
            {
              while ( v108 != *v129 )
              {
                if ( v112 == ++v129 )
                  goto LABEL_209;
              }
              goto LABEL_172;
            }
LABEL_209:
            if ( HIDWORD(v182) < (unsigned int)v182 )
            {
              ++HIDWORD(v182);
              *v112 = v108;
              ++v180;
            }
            else
            {
LABEL_211:
              sub_C8CC70((__int64)&v180, v108, (__int64)v112, v113, v114, v115);
            }
          }
        }
LABEL_172:
        if ( v189 != &v190 )
          _libc_free((unsigned __int64)v189);
        ++v107;
      }
      while ( v159 != v107 );
    }
  }
  v32 = (unsigned __int64)v168;
  v139 = &v168[(unsigned int)v169];
  if ( v139 != v168 )
    goto LABEL_41;
LABEL_196:
  v137 = 0;
LABEL_75:
  if ( !v184 )
    _libc_free((unsigned __int64)v181);
LABEL_77:
  if ( !v178 )
    _libc_free((unsigned __int64)v175);
  if ( v171 != v173 )
    _libc_free((unsigned __int64)v171);
  if ( v168 != (__int64 *)v170 )
    _libc_free((unsigned __int64)v168);
  return v137;
}
