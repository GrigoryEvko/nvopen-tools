// Function: sub_289C020
// Address: 0x289c020
//
void __fastcall sub_289C020(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // r15
  __int64 v5; // r14
  _QWORD *v6; // rax
  _QWORD *i; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  _BYTE *v10; // rax
  unsigned __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // eax
  __int64 v14; // r15
  unsigned __int8 **v15; // rdx
  __int64 v16; // rcx
  unsigned __int8 *v17; // rbx
  unsigned __int8 **v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  unsigned int v21; // r12d
  __int64 v22; // rax
  unsigned int v23; // r10d
  __int64 v24; // r13
  int v25; // edx
  int v26; // ebx
  __int64 *v27; // rdi
  __int64 v28; // rax
  unsigned int v29; // r10d
  int v30; // edx
  int v31; // r8d
  int v32; // edx
  bool v33; // of
  __int64 v34; // rax
  unsigned __int64 v35; // r13
  signed __int64 v36; // rax
  int v37; // edx
  int v38; // r12d
  __int64 v39; // rbx
  unsigned __int64 v40; // r13
  signed __int64 v41; // rsi
  unsigned __int64 v42; // rax
  int v43; // edx
  int v44; // ecx
  __int64 v45; // rax
  int v46; // edx
  unsigned __int64 v47; // rdx
  __int64 v48; // rdx
  unsigned __int8 **v49; // rbx
  __int64 *v50; // rax
  int v51; // ecx
  unsigned __int8 *v52; // r12
  int v53; // eax
  __int64 v54; // rdx
  __int64 v55; // rdx
  unsigned __int8 *v56; // r13
  __int64 v57; // rdx
  __int64 v58; // rax
  char v59; // dl
  __int64 v60; // r12
  unsigned __int8 v61; // al
  int v62; // r13d
  signed __int64 v63; // rax
  __int64 v64; // r12
  int v65; // eax
  unsigned __int64 v66; // rax
  __int64 v67; // r13
  unsigned __int8 *v68; // r12
  __int64 v69; // rdx
  __int64 v70; // r13
  unsigned __int8 **v71; // rax
  __int64 v72; // r13
  _BYTE *v73; // rsi
  __int64 v74; // r12
  unsigned __int8 *v75; // rax
  _BYTE *v76; // r12
  _BYTE *v77; // r13
  unsigned __int8 *v78; // rbx
  __int64 (__fastcall *v79)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v80; // rdi
  __int64 v81; // r10
  __int64 v82; // r12
  __int64 *v83; // r11
  __int64 v84; // r14
  unsigned __int64 v85; // r13
  signed __int64 v86; // rax
  int v87; // edx
  __int64 v88; // r14
  __int64 v89; // rax
  bool v90; // zf
  unsigned __int64 v91; // rdx
  int v92; // edx
  __int64 v93; // rax
  __int64 v94; // rdx
  int v95; // r14d
  int v96; // ebx
  __int64 *v97; // r13
  __int64 *v98; // r15
  __int64 v99; // rax
  __int64 v100; // rax
  int v101; // edx
  __int64 v102; // rdx
  unsigned __int8 *v103; // r14
  __int64 v104; // rdx
  __int64 v105; // rax
  char v106; // dl
  __int64 v107; // rax
  signed __int64 v108; // rax
  int v109; // edx
  int v110; // r13d
  unsigned __int64 v111; // rdx
  __int64 v112; // rax
  __int64 v113; // rdx
  __int64 v114; // r13
  __int64 v115; // rax
  char v116; // al
  char v117; // di
  _QWORD *v118; // rax
  __int64 v119; // r9
  __int64 v120; // r15
  __int64 v121; // r13
  __int64 *v122; // r13
  __int64 *v123; // rbx
  __int64 v124; // rdx
  unsigned int v125; // esi
  __int64 v126; // rax
  int v127; // edx
  _QWORD *v128; // rax
  __int64 v129; // r9
  _QWORD *v130; // r10
  __int64 v131; // rax
  int v132; // r14d
  __int64 *v133; // r13
  __int64 *v134; // r15
  __int64 v135; // rax
  __int64 v136; // rax
  int v137; // edx
  int v138; // eax
  unsigned __int8 *v139; // rax
  unsigned __int8 *v140; // rax
  unsigned __int64 v141; // rdx
  __int64 v142; // [rsp+0h] [rbp-2B0h]
  __int64 v143; // [rsp+8h] [rbp-2A8h]
  unsigned __int8 *v144; // [rsp+18h] [rbp-298h]
  int v145; // [rsp+18h] [rbp-298h]
  _BYTE *v146; // [rsp+20h] [rbp-290h]
  __int64 v147; // [rsp+30h] [rbp-280h]
  __int64 v148; // [rsp+50h] [rbp-260h]
  __int64 v149; // [rsp+50h] [rbp-260h]
  __int64 v150; // [rsp+58h] [rbp-258h]
  unsigned int v151; // [rsp+60h] [rbp-250h]
  __int64 v152; // [rsp+60h] [rbp-250h]
  __int64 v153; // [rsp+60h] [rbp-250h]
  __int64 v154; // [rsp+60h] [rbp-250h]
  unsigned __int8 **v155; // [rsp+60h] [rbp-250h]
  __int64 *v156; // [rsp+60h] [rbp-250h]
  __int64 *v157; // [rsp+60h] [rbp-250h]
  int v158; // [rsp+60h] [rbp-250h]
  __int64 *v159; // [rsp+68h] [rbp-248h]
  int v160; // [rsp+68h] [rbp-248h]
  unsigned int v161; // [rsp+70h] [rbp-240h]
  __int64 v162; // [rsp+70h] [rbp-240h]
  unsigned int v163; // [rsp+70h] [rbp-240h]
  int v164; // [rsp+70h] [rbp-240h]
  __int64 v165; // [rsp+78h] [rbp-238h]
  __int64 v166; // [rsp+78h] [rbp-238h]
  char v167; // [rsp+83h] [rbp-22Dh]
  int v169; // [rsp+88h] [rbp-228h]
  int v170; // [rsp+88h] [rbp-228h]
  unsigned __int8 **v171; // [rsp+88h] [rbp-228h]
  _QWORD *v173; // [rsp+90h] [rbp-220h]
  __int64 v174; // [rsp+90h] [rbp-220h]
  __int64 v175; // [rsp+A8h] [rbp-208h] BYREF
  __int64 v176; // [rsp+B0h] [rbp-200h]
  int v177; // [rsp+B8h] [rbp-1F8h] BYREF
  unsigned int v178; // [rsp+BCh] [rbp-1F4h]
  char v179[4]; // [rsp+C4h] [rbp-1ECh] BYREF
  int v180; // [rsp+C8h] [rbp-1E8h]
  _DWORD v181[8]; // [rsp+D0h] [rbp-1E0h] BYREF
  __int16 v182; // [rsp+F0h] [rbp-1C0h]
  __int64 *v183; // [rsp+100h] [rbp-1B0h] BYREF
  int v184; // [rsp+108h] [rbp-1A8h]
  __int64 *v185; // [rsp+110h] [rbp-1A0h]
  __int16 v186; // [rsp+120h] [rbp-190h]
  __int64 v187; // [rsp+130h] [rbp-180h] BYREF
  unsigned __int8 **v188; // [rsp+138h] [rbp-178h]
  __int64 v189; // [rsp+140h] [rbp-170h]
  int v190; // [rsp+148h] [rbp-168h]
  char v191; // [rsp+14Ch] [rbp-164h]
  char v192; // [rsp+150h] [rbp-160h] BYREF
  unsigned __int8 **v193; // [rsp+170h] [rbp-140h] BYREF
  __int64 v194; // [rsp+178h] [rbp-138h]
  _BYTE v195[48]; // [rsp+180h] [rbp-130h] BYREF
  unsigned __int8 **v196; // [rsp+1B0h] [rbp-100h] BYREF
  __int64 v197; // [rsp+1B8h] [rbp-F8h]
  _BYTE v198[48]; // [rsp+1C0h] [rbp-F0h] BYREF
  __int64 *v199; // [rsp+1F0h] [rbp-C0h] BYREF
  unsigned int v200; // [rsp+1F8h] [rbp-B8h]
  __int64 v201; // [rsp+200h] [rbp-B0h]
  __int64 v202; // [rsp+220h] [rbp-90h]
  __int64 v203; // [rsp+228h] [rbp-88h]
  __int64 v204; // [rsp+230h] [rbp-80h]
  _QWORD *v205; // [rsp+238h] [rbp-78h]
  __int64 v206; // [rsp+240h] [rbp-70h]
  __int64 v207; // [rsp+248h] [rbp-68h]

  v4 = a3;
  v5 = a2;
  if ( *(_BYTE *)(a3 + 28) )
  {
    v6 = *(_QWORD **)(a3 + 8);
    for ( i = &v6[*(unsigned int *)(a3 + 20)]; i != v6; ++v6 )
    {
      if ( a2 == *v6 )
        return;
    }
  }
  else if ( sub_C8CA60(a3, a2) )
  {
    return;
  }
  if ( !dword_5003CC8 )
  {
    sub_28940A0(
      (__int64)&v177,
      *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
      *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
    sub_28940A0(
      (__int64)v179,
      *(_QWORD *)(a2 + 32 * (v8 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
      *(_QWORD *)(a2 + 32 * (4LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
    if ( v177 == 1 && v180 == 1 )
    {
      v9 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v150 = *(_QWORD *)(a2 - 32 * v9);
      v148 = *(_QWORD *)(*(_QWORD *)(v150 + 8) + 24LL);
      v167 = *(_BYTE *)(v148 + 8);
      if ( v167 == 12 || (a4 & 1) != 0 )
      {
        v187 = 0;
        v196 = (unsigned __int8 **)v198;
        v10 = *(_BYTE **)(a2 + 32 * (1 - v9));
        v191 = 1;
        v189 = 4;
        v146 = v10;
        v188 = (unsigned __int8 **)&v192;
        v193 = (unsigned __int8 **)v195;
        v194 = 0x600000000LL;
        v197 = 0x600000000LL;
        v190 = 0;
        sub_94F890((__int64)&v193, v150);
        v13 = v194;
        v169 = 0;
        v165 = 0;
        v159 = (__int64 *)(a1 + 64);
        if ( !(_DWORD)v194 )
          goto LABEL_23;
        v147 = v4;
        v14 = a1;
        while ( 1 )
        {
          v15 = v193;
          v16 = v13;
          v17 = v193[v13 - 1];
          LODWORD(v194) = v13 - 1;
          if ( !v191 )
            goto LABEL_66;
          v18 = v188;
          v16 = HIDWORD(v189);
          v15 = &v188[HIDWORD(v189)];
          if ( v188 != v15 )
          {
            while ( v17 != *v18 )
            {
              if ( v15 == ++v18 )
                goto LABEL_88;
            }
            goto LABEL_20;
          }
LABEL_88:
          if ( HIDWORD(v189) < (unsigned int)v189 )
          {
            ++HIDWORD(v189);
            *v15 = v17;
            ++v187;
          }
          else
          {
LABEL_66:
            sub_C8CC70((__int64)&v187, (__int64)v17, (__int64)v15, v16, v11, v12);
            if ( !v59 )
              goto LABEL_20;
          }
          v163 = v178;
          v60 = *(_QWORD *)(v14 + 72) + 24LL * *(unsigned int *)(v14 + 88);
          sub_2895280(&v199, v159, (__int64)v17);
          if ( v60 == v201 )
          {
            v62 = 1;
LABEL_70:
            if ( v169 == 1 )
              goto LABEL_20;
            v63 = v165;
            v64 = 0;
LABEL_72:
            if ( v169 != v62 )
            {
              if ( v62 >= v169 )
                goto LABEL_20;
              goto LABEL_74;
            }
            goto LABEL_132;
          }
          v61 = *v17;
          if ( *v17 <= 0x1Cu )
            goto LABEL_69;
          v82 = *((_QWORD *)v17 + 1);
          v83 = *(__int64 **)(v82 + 24);
          if ( (unsigned int)v61 - 42 > 0x11 )
          {
            v94 = *((_QWORD *)v17 + 2);
            if ( !v94 || *(_QWORD *)(v94 + 8) )
            {
LABEL_123:
              if ( v163 > 1 )
              {
                v153 = v14;
                v95 = 1;
                v64 = 0;
                v144 = v17;
                v96 = 0;
                v97 = v83;
                do
                {
                  v98 = *(__int64 **)(v153 + 16);
                  v99 = sub_BCDA70(v97, 1);
                  v100 = sub_DFBC30(v98, 8, v99, 0, 0, 0, 0, 0, 0, 0, 0);
                  if ( v101 == 1 )
                    v96 = 1;
                  v33 = __OFADD__(v100, v64);
                  v64 += v100;
                  if ( v33 )
                  {
                    v64 = 0x8000000000000000LL;
                    if ( v100 > 0 )
                      v64 = 0x7FFFFFFFFFFFFFFFLL;
                  }
                  ++v95;
                }
                while ( v163 != v95 );
                goto LABEL_129;
              }
              goto LABEL_155;
            }
            if ( v61 == 61 )
              goto LABEL_178;
            if ( v61 != 85 )
              goto LABEL_123;
            v102 = *((_QWORD *)v17 - 4);
            if ( !v102 )
              goto LABEL_123;
            if ( !*(_BYTE *)v102 && *(_QWORD *)(v102 + 24) == *((_QWORD *)v17 + 10) && *(_DWORD *)(v102 + 36) == 234 )
            {
              v106 = 85;
              goto LABEL_148;
            }
            if ( *(_BYTE *)v102 )
              goto LABEL_123;
            if ( *(_QWORD *)(v102 + 24) != *((_QWORD *)v17 + 10) )
              goto LABEL_123;
            if ( *(_DWORD *)(v102 + 36) != 231 )
              goto LABEL_123;
            v103 = *(unsigned __int8 **)&v17[32 * (1LL - (*((_DWORD *)v17 + 1) & 0x7FFFFFF))];
            v104 = *v103;
            if ( (_BYTE)v104 != 17 )
            {
              if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v103 + 1) + 8LL) - 17 > 1 )
                goto LABEL_123;
              if ( (unsigned __int8)v104 > 0x15u )
                goto LABEL_123;
              v157 = *(__int64 **)(v82 + 24);
              v139 = sub_AD7630((__int64)v103, 0, v104);
              v83 = v157;
              v103 = v139;
              if ( !v139 || *v139 != 17 )
                goto LABEL_123;
            }
            if ( *((_DWORD *)v103 + 8) > 0x40u )
            {
              v145 = *((_DWORD *)v103 + 8);
              v156 = v83;
              v138 = sub_C444A0((__int64)(v103 + 24));
              v83 = v156;
              if ( (unsigned int)(v145 - v138) > 0x40 )
                goto LABEL_123;
              v105 = **((_QWORD **)v103 + 3);
            }
            else
            {
              v105 = *((_QWORD *)v103 + 3);
            }
            if ( v105 != 1 )
              goto LABEL_123;
            v61 = *v17;
            v106 = *v17;
            if ( *v17 <= 0x1Cu )
              goto LABEL_148;
            if ( (unsigned int)v61 - 42 > 0x11 )
            {
LABEL_178:
              v106 = v61;
              goto LABEL_148;
            }
          }
          v152 = (__int64)v83;
          v84 = *(_QWORD *)(v14 + 72) + 24LL * *(unsigned int *)(v14 + 88);
          sub_2895280(&v199, v159, (__int64)v17);
          v83 = (__int64 *)v152;
          if ( v84 != v201 )
          {
            v85 = v163;
            v86 = sub_DFD800(*(_QWORD *)(v14 + 16), (unsigned int)*v17 - 29, v152, 0, 0, 0, 0, 0, 0, 0);
            v164 = v87;
            v88 = v85 * v86;
            if ( !is_mul_ok(v85, v86) )
            {
              if ( !v85 || (v88 = 0x7FFFFFFFFFFFFFFFLL, v86 <= 0) )
                v88 = 0x8000000000000000LL;
            }
            v142 = 0;
            v143 = 0;
            v89 = sub_DFD800(*(_QWORD *)(v14 + 16), (unsigned int)*v17 - 29, v82, 0, 0, 0, 0, 0, 0, 0);
            v90 = v164 == 1;
            v11 = v91;
            goto LABEL_114;
          }
          v106 = *v17;
LABEL_148:
          if ( v106 == 85 )
          {
            v107 = *((_QWORD *)v17 - 4);
            if ( v107 )
            {
              if ( !*(_BYTE *)v107 && *(_QWORD *)(v107 + 24) == *((_QWORD *)v17 + 10) && *(_DWORD *)(v107 + 36) == 234 )
              {
                if ( v163 > 1 )
                {
                  v153 = v14;
                  v132 = 1;
                  v64 = 0;
                  v144 = v17;
                  v96 = 0;
                  v133 = v83;
                  do
                  {
                    v134 = *(__int64 **)(v153 + 16);
                    v135 = sub_BCDA70(v133, 1);
                    v136 = sub_DFBC30(v134, 8, v135, 0, 0, 0, 0, 0, 0, 0, 0);
                    if ( v137 == 1 )
                      v96 = 1;
                    v33 = __OFSUB__(v64, v136);
                    v64 -= v136;
                    if ( v33 )
                    {
                      v64 = 0x7FFFFFFFFFFFFFFFLL;
                      if ( v136 > 0 )
                        v64 = 0x8000000000000000LL;
                    }
                    ++v132;
                  }
                  while ( v163 != v132 );
LABEL_129:
                  v62 = v96;
                  v14 = v153;
                  v17 = v144;
                  goto LABEL_118;
                }
LABEL_155:
                v62 = 0;
                v64 = 0;
                goto LABEL_118;
              }
            }
          }
          if ( v163 == 1 )
          {
LABEL_69:
            v62 = 0;
            goto LABEL_70;
          }
          v108 = sub_DFD4A0(*(__int64 **)(v14 + 16));
          v110 = v109;
          v88 = v108 * v163;
          if ( !is_mul_ok(v108, v163) )
          {
            if ( v108 <= 0 || (v88 = 0x7FFFFFFFFFFFFFFFLL, !v163) )
              v88 = 0x8000000000000000LL;
          }
          v89 = sub_DFD4A0(*(__int64 **)(v14 + 16));
          v90 = v110 == 1;
          v12 = 0;
          v11 = v111;
LABEL_114:
          v92 = 1;
          if ( !v90 )
            v92 = v11;
          v33 = __OFSUB__(v89, v88);
          v93 = v89 - v88;
          v62 = v92;
          if ( v33 )
          {
            v64 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v88 > 0 )
              v64 = 0x8000000000000000LL;
          }
          else
          {
            v64 = v93;
          }
LABEL_118:
          if ( v169 != 1 )
          {
            v63 = v64 + v165;
            if ( __OFADD__(v64, v165) )
            {
              v63 = 0x8000000000000000LL;
              if ( v165 > 0 )
                v63 = 0x7FFFFFFFFFFFFFFFLL;
            }
            goto LABEL_72;
          }
          if ( !__OFADD__(v64, v165) )
          {
            if ( v64 + v165 >= v165 )
              goto LABEL_20;
            goto LABEL_74;
          }
          if ( v165 > 0 )
            goto LABEL_20;
          v63 = 0x8000000000000000LL;
LABEL_132:
          if ( v63 >= v165 )
            goto LABEL_20;
LABEL_74:
          v65 = 1;
          if ( v62 != 1 )
            v65 = v169;
          v169 = v65;
          v66 = v64 + v165;
          if ( __OFADD__(v64, v165) )
          {
            v66 = 0x8000000000000000LL;
            if ( v64 > 0 )
              v66 = 0x7FFFFFFFFFFFFFFFLL;
          }
          v165 = v66;
          sub_94F890((__int64)&v196, (__int64)v17);
          if ( *v17 > 0x1Cu )
          {
            v67 = 32LL * (*((_DWORD *)v17 + 1) & 0x7FFFFFF);
            if ( (v17[7] & 0x40) != 0 )
            {
              v68 = (unsigned __int8 *)*((_QWORD *)v17 - 1);
              v17 = &v68[v67];
            }
            else
            {
              v68 = &v17[-v67];
            }
            v69 = (unsigned int)v194;
            v70 = v67 >> 5;
            v11 = v70 + (unsigned int)v194;
            if ( v11 > HIDWORD(v194) )
            {
              sub_C8D5F0((__int64)&v193, v195, v70 + (unsigned int)v194, 8u, v11, v12);
              v69 = (unsigned int)v194;
            }
            v71 = &v193[v69];
            if ( v17 != v68 )
            {
              do
              {
                if ( v71 )
                  *v71 = *(unsigned __int8 **)v68;
                v68 += 32;
                ++v71;
              }
              while ( v17 != v68 );
              LODWORD(v69) = v194;
            }
            LODWORD(v194) = v69 + v70;
            v13 = v69 + v70;
            goto LABEL_21;
          }
LABEL_20:
          v13 = v194;
LABEL_21:
          if ( !v13 )
          {
            v5 = a2;
            v4 = v147;
LABEL_23:
            v19 = *(_QWORD *)(a1 + 16);
            v20 = *(_QWORD *)(v150 + 8);
            if ( v167 == 12 )
            {
              v21 = 17;
              v126 = sub_DFD800(v19, 0x11u, v20, 0, 0, 0, 0, 0, 0, 0);
              BYTE4(v176) = 0;
              v23 = 13;
              v24 = v126;
              v26 = v127;
              v27 = *(__int64 **)(a1 + 16);
            }
            else
            {
              v21 = 18;
              v22 = sub_DFD800(v19, 0x12u, v20, 0, 0, 0, 0, 0, 0, 0);
              BYTE4(v176) = 1;
              v23 = 14;
              v24 = v22;
              v26 = v25;
              v27 = *(__int64 **)(a1 + 16);
              LODWORD(v176) = a4;
            }
            v161 = v23;
            v28 = sub_DFDC10(v27, v23, *(_QWORD *)(v150 + 8), v176);
            v29 = v161;
            v31 = v30;
            v32 = 1;
            if ( v26 != 1 )
              v32 = v31;
            v33 = __OFADD__(v24, v28);
            v34 = v24 + v28;
            v160 = v32;
            if ( v33 )
            {
              v141 = 0x8000000000000000LL;
              if ( v24 > 0 )
                v141 = 0x7FFFFFFFFFFFFFFFLL;
              v162 = v141;
            }
            else
            {
              v162 = v34;
            }
            v151 = v29;
            v35 = v178;
            v36 = sub_DFD800(*(_QWORD *)(a1 + 16), v21, v148, 0, 0, 0, 0, 0, 0, 0);
            v38 = v37;
            v39 = v35 * v36;
            if ( !is_mul_ok(v35, v36) )
            {
              if ( v36 <= 0 || (v39 = 0x7FFFFFFFFFFFFFFFLL, !v35) )
                v39 = 0x8000000000000000LL;
            }
            v40 = v178 - 1;
            v41 = sub_DFD800(*(_QWORD *)(a1 + 16), v151, v148, 0, 0, 0, 0, 0, 0, 0);
            v42 = v40 * v41;
            v44 = v43;
            if ( !is_mul_ok(v40, v41) )
            {
              if ( v41 <= 0 || (v42 = 0x7FFFFFFFFFFFFFFFLL, !v40) )
                v42 = 0x8000000000000000LL;
            }
            if ( v38 == 1 )
              v44 = 1;
            v33 = __OFADD__(v39, v42);
            v45 = v39 + v42;
            if ( v33 )
            {
              v45 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v39 <= 0 )
                v45 = 0x8000000000000000LL;
            }
            v46 = 1;
            if ( v160 != 1 )
              v46 = v169;
            v170 = v46;
            v47 = v162 + v165;
            if ( __OFADD__(v162, v165) )
            {
              v47 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v162 <= 0 )
                v47 = 0x8000000000000000LL;
            }
            if ( v44 == 1 )
              goto LABEL_103;
            v33 = __OFSUB__(v47, v45);
            v48 = v47 - v45;
            if ( v33 )
            {
              if ( v45 <= 0 || v170 )
                goto LABEL_103;
            }
            else if ( v170 || v48 > 0 )
            {
              goto LABEL_103;
            }
            sub_BED950((__int64)&v199, v4, v5);
            sub_23D0AB0((__int64)&v199, v5, 0, 0, 0);
            v49 = v196;
            v149 = a1 + 96;
            v171 = &v196[(unsigned int)v197];
            if ( v171 == v196 )
              goto LABEL_92;
            v166 = v4;
            while ( 2 )
            {
              v52 = *v49;
              v53 = **v49;
              if ( (unsigned __int8)(v53 - 42) > 0x11u )
              {
                v54 = *((_QWORD *)v52 + 2);
                if ( !v54 || *(_QWORD *)(v54 + 8) )
                  goto LABEL_46;
                if ( (_BYTE)v53 == 61 )
                  goto LABEL_63;
                if ( (_BYTE)v53 != 85 || (v55 = *((_QWORD *)v52 - 4)) == 0 )
                {
LABEL_46:
                  if ( v171 == ++v49 )
                  {
                    v4 = v166;
LABEL_92:
                    v72 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
                    if ( v167 == 12 )
                    {
                      v186 = 257;
                      v131 = sub_A81850((unsigned int **)&v199, (_BYTE *)v72, v146, (__int64)&v183, 0, 0);
                      v76 = (_BYTE *)sub_B34850((__int64)&v199, v131);
                    }
                    else
                    {
                      v73 = *(_BYTE **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
                      v186 = 257;
                      v181[1] = 0;
                      v74 = sub_A826E0((unsigned int **)&v199, v73, v146, v181[0], (__int64)&v183, 0);
                      v75 = sub_AD8DD0(*(_QWORD *)(*(_QWORD *)(v72 + 8) + 24LL), 0.0);
                      v76 = (_BYTE *)sub_B348A0((__int64)&v199, (__int64)v75, v74);
                      sub_B45150((__int64)v76, a4);
                    }
                    v182 = 257;
                    v77 = (_BYTE *)sub_ACADE0(*(__int64 ***)(v5 + 8));
                    v78 = (unsigned __int8 *)sub_2894BB0(v205, 0);
                    v79 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v206
                                                                                                 + 104LL);
                    if ( v79 == sub_948040 )
                    {
                      v80 = 0;
                      if ( *v77 <= 0x15u )
                        v80 = (__int64)v77;
                      if ( *v76 <= 0x15u && *v78 <= 0x15u && v80 )
                      {
                        v81 = sub_AD5A90(v80, v76, v78, 0);
                        goto LABEL_101;
                      }
                      goto LABEL_173;
                    }
                    v81 = v79(v206, v77, v76, v78);
LABEL_101:
                    if ( !v81 )
                    {
LABEL_173:
                      v186 = 257;
                      v128 = sub_BD2C40(72, 3u);
                      v129 = 0;
                      v130 = v128;
                      if ( v128 )
                      {
                        v173 = v128;
                        sub_B4DFA0((__int64)v128, (__int64)v77, (__int64)v76, (__int64)v78, (__int64)&v183, 0, 0, 0);
                        v130 = v173;
                      }
                      v174 = (__int64)v130;
                      (*(void (__fastcall **)(__int64, _QWORD *, _DWORD *, __int64, __int64, __int64))(*(_QWORD *)v207 + 16LL))(
                        v207,
                        v130,
                        v181,
                        v203,
                        v204,
                        v129);
                      sub_94AAF0((unsigned int **)&v199, v174);
                      v81 = v174;
                    }
                    sub_BD84D0(v5, v81);
                    sub_BED950((__int64)&v183, v4, v5);
                    sub_9C95B0(v149, v5);
                    sub_F94A20(&v199, v5);
LABEL_103:
                    if ( v196 != (unsigned __int8 **)v198 )
                      _libc_free((unsigned __int64)v196);
                    if ( v193 != (unsigned __int8 **)v195 )
                      _libc_free((unsigned __int64)v193);
                    if ( !v191 )
                      _libc_free((unsigned __int64)v188);
                    return;
                  }
                  continue;
                }
                if ( *(_BYTE *)v55 || *(_QWORD *)(v55 + 24) != *((_QWORD *)v52 + 10) || *(_DWORD *)(v55 + 36) != 234 )
                {
                  if ( *(_BYTE *)v55 )
                    goto LABEL_46;
                  if ( *(_QWORD *)(v55 + 24) != *((_QWORD *)v52 + 10) )
                    goto LABEL_46;
                  if ( *(_DWORD *)(v55 + 36) != 231 )
                    goto LABEL_46;
                  v56 = *(unsigned __int8 **)&v52[32 * (1LL - (*((_DWORD *)v52 + 1) & 0x7FFFFFF))];
                  v57 = *v56;
                  if ( (_BYTE)v57 != 17 )
                  {
                    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v56 + 1) + 8LL) - 17 > 1 )
                      goto LABEL_46;
                    if ( (unsigned __int8)v57 > 0x15u )
                      goto LABEL_46;
                    v140 = sub_AD7630((__int64)v56, 0, v57);
                    v56 = v140;
                    if ( !v140 || *v140 != 17 )
                      goto LABEL_46;
                  }
                  if ( *((_DWORD *)v56 + 8) > 0x40u )
                  {
                    v158 = *((_DWORD *)v56 + 8);
                    if ( v158 - (unsigned int)sub_C444A0((__int64)(v56 + 24)) > 0x40 )
                      goto LABEL_46;
                    v58 = **((_QWORD **)v56 + 3);
                  }
                  else
                  {
                    v58 = *((_QWORD *)v56 + 3);
                  }
                  if ( v58 != 1 )
                    goto LABEL_46;
                  v53 = *v52;
                  if ( (unsigned __int8)v53 <= 0x1Cu )
                  {
LABEL_63:
                    sub_BED950((__int64)&v183, v166, (__int64)v52);
                    if ( *v52 == 85
                      && (v112 = *((_QWORD *)v52 - 4)) != 0
                      && !*(_BYTE *)v112
                      && *(_QWORD *)(v112 + 24) == *((_QWORD *)v52 + 10)
                      && *(_DWORD *)(v112 + 36) == 231
                      && (v113 = *(_QWORD *)&v52[-32 * (*((_DWORD *)v52 + 1) & 0x7FFFFFF)]) != 0 )
                    {
                      v175 = *(_QWORD *)&v52[-32 * (*((_DWORD *)v52 + 1) & 0x7FFFFFF)];
                      v182 = 257;
                      v114 = *((_QWORD *)v52 + 1);
                      v154 = v113;
                      v115 = sub_AA4E30(v202);
                      v116 = sub_AE5020(v115, v114);
                      v186 = 257;
                      v117 = v116;
                      v118 = sub_BD2C40(80, unk_3F10A14);
                      v120 = (__int64)v118;
                      if ( v118 )
                        sub_B4D190((__int64)v118, v114, v154, (__int64)&v183, 0, v117, 0, 0);
                      (*(void (__fastcall **)(__int64, __int64, _DWORD *, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v207 + 16LL))(
                        v207,
                        v120,
                        v181,
                        v203,
                        v204,
                        v119,
                        v142,
                        v143);
                      v121 = 2LL * v200;
                      if ( v199 != &v199[v121] )
                      {
                        v155 = v49;
                        v122 = &v199[v121];
                        v123 = v199;
                        do
                        {
                          v124 = v123[1];
                          v125 = *(_DWORD *)v123;
                          v123 += 2;
                          sub_B99FD0(v120, v125, v124);
                        }
                        while ( v122 != v123 );
                        v49 = v155;
                      }
                      sub_BD84D0((__int64)v52, v120);
                      sub_28957D0(a1, v52);
                    }
                    else
                    {
                      LODWORD(v183) = 234;
                      v184 = 0;
                      v185 = &v175;
                      if ( (unsigned __int8)sub_10E25C0((__int64)&v183, (__int64)v52) )
                      {
                        sub_9C95B0(v149, (__int64)v52);
                        sub_BD84D0((__int64)v52, v175);
                      }
                    }
                    goto LABEL_46;
                  }
                }
              }
              break;
            }
            if ( (unsigned int)(v53 - 42) <= 0x11 )
            {
              sub_2895280(&v183, (__int64 *)(a1 + 64), (__int64)v52);
              v50 = v185;
              if ( v185 != (__int64 *)(*(_QWORD *)(a1 + 72) + 24LL * *(unsigned int *)(a1 + 88)) )
              {
                v90 = dword_5003CC8 == 0;
                v51 = *((_DWORD *)v185 + 3);
                *((_DWORD *)v185 + 3) = *((_DWORD *)v185 + 2);
                *((_BYTE *)v50 + 16) = v90;
                *((_DWORD *)v50 + 2) = v51;
                goto LABEL_46;
              }
            }
            goto LABEL_63;
          }
        }
      }
    }
  }
}
