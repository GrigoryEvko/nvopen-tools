// Function: sub_2E92D10
// Address: 0x2e92d10
//
void __fastcall sub_2E92D10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r14
  _QWORD *v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // r15
  __int64 v18; // rbx
  unsigned __int8 **v19; // r14
  __m128i *v20; // r13
  __int64 v21; // r12
  unsigned __int32 v22; // r11d
  unsigned __int8 *v23; // rax
  unsigned __int8 *v24; // rdx
  unsigned __int8 v25; // al
  unsigned __int64 v26; // rbx
  unsigned __int8 **v27; // r14
  __int64 v28; // rax
  __int32 v29; // r15d
  unsigned __int64 v30; // rdx
  unsigned __int32 v31; // edx
  char v32; // al
  __int64 v33; // r12
  char *v34; // rdi
  char *v35; // rdx
  _DWORD *v36; // rdi
  _BYTE *v37; // rax
  __int16 *v38; // rax
  int v39; // r13d
  __int16 *v40; // r12
  int v41; // ebx
  unsigned __int8 *v42; // r14
  unsigned __int8 *v43; // rax
  int v44; // eax
  __int64 v45; // rsi
  unsigned __int64 v46; // rax
  __int64 *v47; // rsi
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  unsigned __int8 *v50; // r8
  unsigned __int8 *v51; // r13
  unsigned int *v52; // r15
  _DWORD *v53; // rax
  _DWORD *v54; // rsi
  __int32 *v55; // r13
  __int32 *v56; // r12
  unsigned __int32 v57; // edx
  _DWORD *v58; // rax
  _DWORD *v59; // rsi
  char v60; // si
  _DWORD *v61; // rax
  _DWORD *v62; // rdi
  bool v63; // al
  __int64 v64; // rax
  int v65; // edx
  unsigned __int64 v66; // rbx
  unsigned __int64 v67; // rdi
  unsigned __int64 v68; // rbx
  unsigned __int64 v69; // rdi
  unsigned __int64 v70; // rbx
  unsigned __int64 v71; // rdi
  unsigned __int64 v72; // rbx
  unsigned __int64 v73; // rdi
  __int64 v74; // rax
  unsigned __int64 v75; // rdx
  char *v76; // rax
  unsigned __int64 v77; // rdx
  char v78; // dl
  __int64 v79; // rax
  unsigned __int64 v80; // rdx
  unsigned __int64 v81; // rdx
  unsigned __int64 v82; // rax
  int *v83; // r9
  __int64 v84; // rdi
  __int64 v85; // rcx
  unsigned __int64 v86; // rax
  int *v87; // r8
  __int64 v88; // rsi
  __int64 v89; // rcx
  char *v90; // rax
  unsigned __int64 v91; // rax
  int *v92; // r9
  unsigned __int64 v93; // rax
  __int64 *v94; // rdi
  __int64 v95; // rsi
  __int64 v96; // rcx
  unsigned int *v97; // rbx
  char v98; // r12
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rdx
  __int64 v102; // rax
  __int64 v103; // rdx
  bool v104; // r8
  __int64 v105; // rax
  __int64 v106; // rax
  unsigned int *v107; // r12
  unsigned int *v108; // r13
  char v109; // r14
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rdx
  __int64 v113; // [rsp+0h] [rbp-5D0h]
  __int16 *v114; // [rsp+10h] [rbp-5C0h]
  int v115; // [rsp+1Ch] [rbp-5B4h]
  unsigned __int64 v118; // [rsp+88h] [rbp-548h]
  __m128i *v119; // [rsp+88h] [rbp-548h]
  __int64 v120; // [rsp+90h] [rbp-540h]
  unsigned __int8 **v121; // [rsp+90h] [rbp-540h]
  _QWORD *v122; // [rsp+98h] [rbp-538h]
  __int64 v123; // [rsp+A0h] [rbp-530h]
  __int64 v124; // [rsp+B8h] [rbp-518h]
  unsigned int *v125; // [rsp+C0h] [rbp-510h]
  unsigned int v126; // [rsp+D0h] [rbp-500h]
  _QWORD *v127; // [rsp+D0h] [rbp-500h]
  __int64 v128; // [rsp+D0h] [rbp-500h]
  _BYTE *v130; // [rsp+E8h] [rbp-4E8h]
  unsigned __int32 v131; // [rsp+E8h] [rbp-4E8h]
  char v132; // [rsp+E8h] [rbp-4E8h]
  _QWORD *v133; // [rsp+E8h] [rbp-4E8h]
  unsigned __int32 v134; // [rsp+E8h] [rbp-4E8h]
  unsigned __int32 v135; // [rsp+E8h] [rbp-4E8h]
  unsigned int v136; // [rsp+FCh] [rbp-4D4h] BYREF
  __m128i v137; // [rsp+100h] [rbp-4D0h] BYREF
  __int64 v138; // [rsp+110h] [rbp-4C0h]
  __int64 v139; // [rsp+118h] [rbp-4B8h]
  __int64 v140; // [rsp+120h] [rbp-4B0h]
  __int32 *v141; // [rsp+130h] [rbp-4A0h] BYREF
  __int64 v142; // [rsp+138h] [rbp-498h]
  _BYTE v143[32]; // [rsp+140h] [rbp-490h] BYREF
  _BYTE *v144; // [rsp+160h] [rbp-470h] BYREF
  __int64 v145; // [rsp+168h] [rbp-468h]
  _BYTE v146[32]; // [rsp+170h] [rbp-460h] BYREF
  _BYTE *v147; // [rsp+190h] [rbp-440h] BYREF
  __int64 v148; // [rsp+198h] [rbp-438h]
  _BYTE v149[32]; // [rsp+1A0h] [rbp-430h] BYREF
  __int64 v150; // [rsp+1C0h] [rbp-410h] BYREF
  int v151; // [rsp+1C8h] [rbp-408h] BYREF
  unsigned __int64 v152; // [rsp+1D0h] [rbp-400h]
  int *v153; // [rsp+1D8h] [rbp-3F8h]
  int *v154; // [rsp+1E0h] [rbp-3F0h]
  __int64 v155; // [rsp+1E8h] [rbp-3E8h]
  unsigned __int64 v156[2]; // [rsp+1F0h] [rbp-3E0h] BYREF
  _BYTE v157[40]; // [rsp+200h] [rbp-3D0h] BYREF
  int v158; // [rsp+228h] [rbp-3A8h] BYREF
  unsigned __int64 v159; // [rsp+230h] [rbp-3A0h]
  int *v160; // [rsp+238h] [rbp-398h]
  int *v161; // [rsp+240h] [rbp-390h]
  __int64 v162; // [rsp+248h] [rbp-388h]
  _BYTE *v163; // [rsp+250h] [rbp-380h] BYREF
  __int64 v164; // [rsp+258h] [rbp-378h]
  _BYTE v165[40]; // [rsp+260h] [rbp-370h] BYREF
  int v166; // [rsp+288h] [rbp-348h] BYREF
  unsigned __int64 v167; // [rsp+290h] [rbp-340h]
  int *v168; // [rsp+298h] [rbp-338h]
  int *v169; // [rsp+2A0h] [rbp-330h]
  __int64 v170; // [rsp+2A8h] [rbp-328h]
  _BYTE *v171; // [rsp+2B0h] [rbp-320h] BYREF
  __int64 v172; // [rsp+2B8h] [rbp-318h]
  _BYTE v173[40]; // [rsp+2C0h] [rbp-310h] BYREF
  int v174; // [rsp+2E8h] [rbp-2E8h] BYREF
  unsigned __int64 v175; // [rsp+2F0h] [rbp-2E0h]
  int *v176; // [rsp+2F8h] [rbp-2D8h]
  int *v177; // [rsp+300h] [rbp-2D0h]
  __int64 v178; // [rsp+308h] [rbp-2C8h]
  char *v179; // [rsp+310h] [rbp-2C0h] BYREF
  __int64 v180; // [rsp+318h] [rbp-2B8h]
  _BYTE v181[64]; // [rsp+320h] [rbp-2B0h] BYREF
  __int64 v182; // [rsp+360h] [rbp-270h] BYREF
  __int64 v183; // [rsp+368h] [rbp-268h] BYREF
  unsigned __int64 v184; // [rsp+370h] [rbp-260h]
  __int64 *v185; // [rsp+378h] [rbp-258h]
  __int64 *v186; // [rsp+380h] [rbp-250h]
  __int64 v187; // [rsp+388h] [rbp-248h]
  unsigned __int8 *v188; // [rsp+390h] [rbp-240h] BYREF
  __int64 v189; // [rsp+398h] [rbp-238h]
  _BYTE v190[128]; // [rsp+3A0h] [rbp-230h] BYREF
  unsigned __int8 *v191; // [rsp+420h] [rbp-1B0h] BYREF
  __int64 v192; // [rsp+428h] [rbp-1A8h]
  _BYTE v193[128]; // [rsp+430h] [rbp-1A0h] BYREF
  __int64 v194; // [rsp+4B0h] [rbp-120h] BYREF
  __int64 v195; // [rsp+4B8h] [rbp-118h] BYREF
  unsigned __int64 v196; // [rsp+4C0h] [rbp-110h]
  __int64 *v197; // [rsp+4C8h] [rbp-108h]
  __int64 *v198; // [rsp+4D0h] [rbp-100h]
  __int64 v199; // [rsp+4D8h] [rbp-F8h]
  unsigned __int8 *v200; // [rsp+4E0h] [rbp-F0h] BYREF
  __int64 v201; // [rsp+4E8h] [rbp-E8h]
  __int64 v202[17]; // [rsp+4F0h] [rbp-E0h] BYREF
  int v203; // [rsp+578h] [rbp-58h] BYREF
  unsigned __int64 v204; // [rsp+580h] [rbp-50h]
  int *v205; // [rsp+588h] [rbp-48h]
  int *v206; // [rsp+590h] [rbp-40h]
  __int64 v207; // [rsp+598h] [rbp-38h]

  if ( !a2 )
    BUG();
  v4 = a2;
  if ( (*(_BYTE *)a2 & 4) == 0 && (*(_BYTE *)(a2 + 44) & 8) != 0 )
  {
    do
      v4 = *(_QWORD *)(v4 + 8);
    while ( (*(_BYTE *)(v4 + 44) & 8) != 0 );
  }
  v5 = *(_QWORD *)(v4 + 8);
  if ( a3 != v5 )
  {
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      v6 = v5;
      if ( (*(_BYTE *)v5 & 4) == 0 && (*(_BYTE *)(v5 + 44) & 8) != 0 )
      {
        do
          v6 = *(_QWORD *)(v6 + 8);
        while ( (*(_BYTE *)(v6 + 44) & 8) != 0 );
      }
      v7 = *(_QWORD *)(v6 + 8);
      sub_2E89030((__int64 *)v5);
      if ( a3 == v7 )
        break;
      v5 = v7;
    }
  }
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
  v123 = *(_QWORD *)(a1 + 32);
  v9 = *(__int64 (**)())(*(_QWORD *)v8 + 128LL);
  if ( v9 == sub_2DAC790 )
  {
    (*(void (**)(void))(*(_QWORD *)v8 + 200LL))();
    BUG();
  }
  v10 = v9();
  v113 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v123 + 16) + 200LL))(*(_QWORD *)(v123 + 16));
  v11 = a2;
  v12 = *(_QWORD *)(v10 + 8) - 840LL;
  if ( a2 == a3 )
  {
LABEL_12:
    v188 = 0;
    v200 = 0;
  }
  else
  {
    while ( 1 )
    {
      v45 = *(_QWORD *)(v11 + 56);
      if ( v45 )
        break;
      v11 = *(_QWORD *)(v11 + 8);
      if ( a3 == v11 )
        goto LABEL_12;
    }
    v188 = *(unsigned __int8 **)(v11 + 56);
    sub_B96E90((__int64)&v188, v45, 1);
    v200 = v188;
    if ( v188 )
    {
      sub_B976B0((__int64)&v188, v188, (__int64)&v200);
      v188 = 0;
      v201 = 0;
      v202[0] = 0;
      v191 = v200;
      if ( v200 )
        sub_B96E90((__int64)&v191, (__int64)v200, 1);
      goto LABEL_14;
    }
  }
  v201 = 0;
  v202[0] = 0;
  v191 = 0;
LABEL_14:
  v13 = sub_2E7B380((_QWORD *)v123, v12, &v191, 0);
  v124 = (__int64)v13;
  if ( v201 )
    sub_2E882B0((__int64)v13, v123, v201);
  if ( v202[0] )
    sub_2E88680(v124, v123, v202[0]);
  if ( v191 )
    sub_B91220((__int64)&v191, (__int64)v191);
  if ( v200 )
    sub_B91220((__int64)&v200, (__int64)v200);
  if ( v188 )
    sub_B91220((__int64)&v188, (__int64)v188);
  sub_2E326B0(a1, (__int64 *)a2, v124);
  if ( a2 != a3 )
    sub_2E89040(v124);
  LODWORD(v195) = 0;
  v153 = &v151;
  v154 = &v151;
  v188 = v190;
  v179 = v181;
  v191 = v193;
  v180 = 0x1000000000LL;
  v197 = &v195;
  v198 = &v195;
  v185 = &v183;
  v186 = &v183;
  v147 = v149;
  v141 = (__int32 *)v143;
  v148 = 0x800000000LL;
  v189 = 0x2000000000LL;
  v192 = 0x2000000000LL;
  v196 = 0;
  v199 = 0;
  v151 = 0;
  v152 = 0;
  v155 = 0;
  LODWORD(v183) = 0;
  v184 = 0;
  v187 = 0;
  v142 = 0x800000000LL;
  v156[0] = (unsigned __int64)v157;
  v156[1] = 0x800000000LL;
  v164 = 0x800000000LL;
  v172 = 0x800000000LL;
  v160 = &v158;
  v161 = &v158;
  v176 = &v174;
  v177 = &v174;
  v163 = v165;
  v144 = v146;
  v168 = &v166;
  v169 = &v166;
  v16 = a3;
  v145 = 0x400000000LL;
  v158 = 0;
  v159 = 0;
  v162 = 0;
  v166 = 0;
  v167 = 0;
  v170 = 0;
  v171 = v173;
  v174 = 0;
  v175 = 0;
  v178 = 0;
  if ( a2 == a3 )
  {
    v201 = 0x2000000000LL;
    v200 = (unsigned __int8 *)v202;
    v203 = 0;
    v204 = 0;
    v205 = &v203;
    v206 = &v203;
    v207 = 0;
    goto LABEL_114;
  }
  v17 = a2;
  do
  {
    while ( (unsigned __int16)(*(_WORD *)(v17 + 68) - 14) <= 4u )
    {
      v17 = *(_QWORD *)(v17 + 8);
      if ( v17 == a3 )
        goto LABEL_103;
    }
    v18 = *(_QWORD *)(v17 + 32);
    v19 = &v200;
    v20 = &v137;
    v21 = v18 + 40LL * (*(_DWORD *)(v17 + 40) & 0xFFFFFF);
    while ( v21 != v18 )
    {
      if ( !*(_BYTE *)v18 )
      {
        if ( (*(_BYTE *)(v18 + 3) & 0x10) != 0 )
        {
          v48 = (unsigned int)v145;
          v16 = HIDWORD(v145);
          v49 = (unsigned int)v145 + 1LL;
          if ( v49 > HIDWORD(v145) )
          {
            sub_C8D5F0((__int64)&v144, v146, v49, 8u, v14, v15);
            v48 = (unsigned int)v145;
          }
          *(_QWORD *)&v144[8 * v48] = v18;
          LODWORD(v145) = v145 + 1;
          goto LABEL_41;
        }
        v22 = *(_DWORD *)(v18 + 8);
        v137.m128i_i32[0] = v22;
        if ( v22 )
        {
          if ( v199 )
          {
            v46 = v196;
            if ( !v196 )
              goto LABEL_89;
            v47 = &v195;
            do
            {
              v16 = *(_QWORD *)(v46 + 16);
              if ( v22 > *(_DWORD *)(v46 + 32) )
              {
                v46 = *(_QWORD *)(v46 + 24);
              }
              else
              {
                v47 = (__int64 *)v46;
                v46 = *(_QWORD *)(v46 + 16);
              }
            }
            while ( v46 );
            if ( v47 == &v195 || v22 < *((_DWORD *)v47 + 8) )
              goto LABEL_89;
          }
          else
          {
            v23 = v191;
            v24 = &v191[4 * (unsigned int)v192];
            if ( v191 == v24 )
              goto LABEL_89;
            while ( v22 != *(_DWORD *)v23 )
            {
              v23 += 4;
              if ( v24 == v23 )
                goto LABEL_89;
            }
            if ( v24 == v23 )
            {
LABEL_89:
              sub_2E34820((__int64)v19, (__int64)v156, (unsigned int *)v20, v16, v14);
              if ( LOBYTE(v202[0]) )
              {
                v74 = (unsigned int)v142;
                v16 = HIDWORD(v142);
                v14 = v137.m128i_u32[0];
                v75 = (unsigned int)v142 + 1LL;
                if ( v75 > HIDWORD(v142) )
                {
                  v134 = v137.m128i_i32[0];
                  sub_C8D5F0((__int64)&v141, v143, v75, 4u, v137.m128i_u32[0], v15);
                  v74 = (unsigned int)v142;
                  v14 = v134;
                }
                v141[v74] = v14;
                LODWORD(v142) = v142 + 1;
                if ( (*(_BYTE *)(v18 + 4) & 1) != 0 )
                  sub_2E34820((__int64)v19, (__int64)&v171, (unsigned int *)v20, v16, v14);
              }
              if ( (((*(_BYTE *)(v18 + 3) & 0x40) != 0) & ((*(_BYTE *)(v18 + 3) >> 4) ^ 1)) != 0 )
                sub_2E34820((__int64)v19, (__int64)&v163, (unsigned int *)v20, v16, v14);
              goto LABEL_41;
            }
          }
          v25 = *(_BYTE *)(v18 + 3);
          *(_BYTE *)(v18 + 4) |= 2u;
          if ( (((v25 & 0x40) != 0) & ((v25 >> 4) ^ 1)) != 0 )
          {
            if ( v187 )
            {
              v131 = v22;
              v102 = sub_2DCBDB0((__int64)&v182, (unsigned int *)v20);
              if ( v103 )
              {
                if ( v102 || (__int64 *)v103 == &v183 )
LABEL_249:
                  v104 = 1;
                else
                  v104 = v131 < *(_DWORD *)(v103 + 32);
LABEL_250:
                v127 = (_QWORD *)v103;
                v132 = v104;
                v106 = sub_22077B0(0x28u);
                *(_DWORD *)(v106 + 32) = v137.m128i_i32[0];
                sub_220F040(v132, v106, v127, &v183);
                ++v187;
              }
            }
            else
            {
              v15 = (__int64)v179;
              v14 = (__int64)&v179[4 * (unsigned int)v180];
              if ( v179 == (char *)v14 )
              {
                if ( (unsigned int)v180 <= 0xFuLL )
                  goto LABEL_177;
              }
              else
              {
                v76 = v179;
                while ( v22 != *(_DWORD *)v76 )
                {
                  v76 += 4;
                  if ( (char *)v14 == v76 )
                    goto LABEL_176;
                }
                if ( (char *)v14 != v76 )
                  goto LABEL_41;
LABEL_176:
                if ( (unsigned int)v180 <= 0xFuLL )
                {
LABEL_177:
                  v77 = (unsigned int)v180 + 1LL;
                  if ( v77 > HIDWORD(v180) )
                  {
                    v135 = v22;
                    sub_C8D5F0((__int64)&v179, v181, v77, 4u, v14, (__int64)v179);
                    v22 = v135;
                    v14 = (__int64)&v179[4 * (unsigned int)v180];
                  }
                  *(_DWORD *)v14 = v22;
                  LODWORD(v180) = v180 + 1;
                  goto LABEL_41;
                }
                v128 = v21;
                v107 = (unsigned int *)&v179[4 * (unsigned int)v180];
                v119 = v20;
                v108 = (unsigned int *)v179;
                v121 = v19;
                do
                {
                  v111 = sub_2DCC990(&v182, (__int64)&v183, v108);
                  if ( v112 )
                  {
                    v109 = v111 || (__int64 *)v112 == &v183 || *v108 < *(_DWORD *)(v112 + 32);
                    v133 = (_QWORD *)v112;
                    v110 = sub_22077B0(0x28u);
                    *(_DWORD *)(v110 + 32) = *v108;
                    sub_220F040(v109, v110, v133, &v183);
                    ++v187;
                  }
                  ++v108;
                }
                while ( v107 != v108 );
                v21 = v128;
                v19 = v121;
                v20 = v119;
              }
              LODWORD(v180) = 0;
              v105 = sub_2DCBDB0((__int64)&v182, (unsigned int *)v20);
              if ( v103 )
              {
                if ( v105 || (__int64 *)v103 == &v183 )
                  goto LABEL_249;
                v104 = v137.m128i_i32[0] < *(_DWORD *)(v103 + 32);
                goto LABEL_250;
              }
            }
          }
        }
      }
LABEL_41:
      v18 += 40;
    }
    v26 = (unsigned __int64)v144;
    v130 = &v144[8 * (unsigned int)v145];
    if ( v130 == v144 )
      goto LABEL_102;
    v120 = v17;
    v27 = &v200;
    do
    {
      v33 = *(_QWORD *)v26;
      v137.m128i_i32[0] = *(_DWORD *)(*(_QWORD *)v26 + 8LL);
      if ( !v137.m128i_i32[0] )
        goto LABEL_48;
      sub_2E92A20((__int64)v27, (__int64)&v191, (unsigned int *)&v137, v16, v14);
      if ( LOBYTE(v202[0]) )
      {
        v28 = (unsigned int)v189;
        v29 = v137.m128i_i32[0];
        v30 = (unsigned int)v189 + 1LL;
        if ( v30 > HIDWORD(v189) )
        {
          sub_C8D5F0((__int64)&v188, v190, v30, 4u, v14, v15);
          v28 = (unsigned int)v189;
        }
        *(_DWORD *)&v188[4 * v28] = v29;
        v31 = v137.m128i_i32[0];
        LODWORD(v189) = v189 + 1;
        v16 = *(unsigned __int8 *)(v33 + 3);
        v32 = (unsigned __int8)v16 >> 4;
        LOBYTE(v16) = (unsigned __int8)v16 >> 6;
        if ( (v32 & 1 & (unsigned __int8)v16) == 0 )
          goto LABEL_47;
        sub_2E34820((__int64)v27, (__int64)&v147, (unsigned int *)&v137, v16, v14);
      }
      else
      {
        if ( v187 )
        {
          sub_2E91E90(&v182, (unsigned int *)&v137);
        }
        else
        {
          v34 = v179;
          v35 = &v179[4 * (unsigned int)v180];
          v16 = (unsigned int)v180;
          if ( v179 != v35 )
          {
            while ( *(_DWORD *)v34 != v137.m128i_i32[0] )
            {
              v34 += 4;
              if ( v35 == v34 )
                goto LABEL_60;
            }
            if ( v35 != v34 )
            {
              if ( v35 != v34 + 4 )
              {
                memmove(v34, v34 + 4, v35 - (v34 + 4));
                LODWORD(v16) = v180;
              }
              v16 = (unsigned int)(v16 - 1);
              LODWORD(v180) = v16;
            }
          }
        }
LABEL_60:
        if ( (((*(_BYTE *)(v33 + 3) & 0x10) != 0) & (*(_BYTE *)(v33 + 3) >> 6)) != 0 )
          goto LABEL_48;
        if ( v155 )
        {
          sub_2E91E90(&v150, (unsigned int *)&v137);
          goto LABEL_70;
        }
        v36 = v147;
        v31 = v137.m128i_i32[0];
        v16 = (unsigned int)v148;
        v37 = &v147[4 * (unsigned int)v148];
        if ( v147 == v37 )
          goto LABEL_47;
        while ( v137.m128i_i32[0] != *v36 )
        {
          if ( v37 == (_BYTE *)++v36 )
            goto LABEL_47;
        }
        if ( v37 == (_BYTE *)v36 )
        {
LABEL_47:
          if ( v31 - 1 > 0x3FFFFFFE )
            goto LABEL_48;
LABEL_72:
          v38 = (__int16 *)(*(_QWORD *)(v113 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v113 + 8) + 24LL * v31 + 4));
          v39 = *v38;
          v40 = v38 + 1;
          v16 = v39 + v31;
          v126 = v39 + v31;
          if ( !*v38 )
            goto LABEL_48;
          v118 = v26;
          v14 = (unsigned int)v16;
          v125 = (unsigned int *)v27;
          while ( 2 )
          {
            v41 = (unsigned __int16)v14;
            LODWORD(v200) = (unsigned __int16)v14;
            if ( !v199 )
            {
              v42 = &v191[4 * (unsigned int)v192];
              if ( v191 == v42 )
              {
                if ( (unsigned int)v192 <= 0x1FuLL )
                  goto LABEL_186;
              }
              else
              {
                v43 = v191;
                while ( (unsigned __int16)v14 != *(_DWORD *)v43 )
                {
                  v43 += 4;
                  if ( v42 == v43 )
                    goto LABEL_185;
                }
                if ( v42 != v43 )
                  goto LABEL_80;
LABEL_185:
                if ( (unsigned int)v192 <= 0x1FuLL )
                {
LABEL_186:
                  v81 = (unsigned int)v192 + 1LL;
                  if ( v81 > HIDWORD(v192) )
                  {
                    sub_C8D5F0((__int64)&v191, v193, v81, 4u, v14, v15);
                    v42 = &v191[4 * (unsigned int)v192];
                  }
                  *(_DWORD *)v42 = v41;
                  LODWORD(v192) = v192 + 1;
LABEL_182:
                  v79 = (unsigned int)v189;
                  v16 = HIDWORD(v189);
                  v80 = (unsigned int)v189 + 1LL;
                  if ( v80 > HIDWORD(v189) )
                  {
                    sub_C8D5F0((__int64)&v188, v190, v80, 4u, v14, v15);
                    v79 = (unsigned int)v189;
                  }
                  *(_DWORD *)&v188[4 * v79] = v41;
                  LODWORD(v189) = v189 + 1;
LABEL_80:
                  v44 = *v40++;
                  if ( !(_WORD)v44 )
                  {
                    v26 = v118;
                    v27 = (unsigned __int8 **)v125;
                    goto LABEL_48;
                  }
                  v126 += v44;
                  v16 = v126;
                  v14 = v126;
                  continue;
                }
                v115 = (unsigned __int16)v14;
                v97 = (unsigned int *)v191;
                v114 = v40;
                do
                {
                  v100 = sub_2DCC990(&v194, (__int64)&v195, v97);
                  if ( v101 )
                  {
                    v98 = v100 || (__int64 *)v101 == &v195 || *v97 < *(_DWORD *)(v101 + 32);
                    v122 = (_QWORD *)v101;
                    v99 = sub_22077B0(0x28u);
                    *(_DWORD *)(v99 + 32) = *v97;
                    sub_220F040(v98, v99, v122, &v195);
                    ++v199;
                  }
                  ++v97;
                }
                while ( v42 != (unsigned __int8 *)v97 );
                v41 = v115;
                v40 = v114;
              }
              LODWORD(v192) = 0;
              sub_2DCBE50((__int64)&v194, v125);
              goto LABEL_182;
            }
            break;
          }
          sub_2DCBE50((__int64)&v194, v125);
          if ( v78 )
            goto LABEL_182;
          goto LABEL_80;
        }
        if ( v37 != (_BYTE *)(v36 + 1) )
        {
          memmove(v36, v36 + 1, v37 - (_BYTE *)(v36 + 1));
          LODWORD(v16) = v148;
        }
        v16 = (unsigned int)(v16 - 1);
        LODWORD(v148) = v16;
      }
LABEL_70:
      if ( (((*(_BYTE *)(v33 + 3) & 0x10) != 0) & (*(_BYTE *)(v33 + 3) >> 6)) == 0 )
      {
        v31 = v137.m128i_i32[0];
        if ( (unsigned int)(v137.m128i_i32[0] - 1) <= 0x3FFFFFFE )
          goto LABEL_72;
      }
LABEL_48:
      v26 += 8LL;
    }
    while ( v130 != (_BYTE *)v26 );
    v17 = v120;
LABEL_102:
    LODWORD(v145) = 0;
    v17 = *(_QWORD *)(v17 + 8);
  }
  while ( v17 != a3 );
LABEL_103:
  v50 = v188;
  v203 = 0;
  v204 = 0;
  v51 = &v188[4 * (unsigned int)v189];
  v207 = 0;
  v200 = (unsigned __int8 *)v202;
  v201 = 0x2000000000LL;
  v205 = &v203;
  v206 = &v203;
  if ( v51 != v188 )
  {
    v52 = (unsigned int *)v188;
    while ( 1 )
    {
      v136 = *v52;
      sub_2E92A20((__int64)&v137, (__int64)&v200, &v136, v16, (__int64)v50);
      v16 = (unsigned __int8)v138;
      if ( (_BYTE)v138 )
        break;
LABEL_113:
      if ( v51 == (unsigned __int8 *)++v52 )
        goto LABEL_114;
    }
    if ( v155 )
    {
      v91 = v152;
      if ( !v152 )
        goto LABEL_207;
      v92 = &v151;
      do
      {
        if ( v136 > *(_DWORD *)(v91 + 32) )
        {
          v91 = *(_QWORD *)(v91 + 24);
        }
        else
        {
          v92 = (int *)v91;
          v91 = *(_QWORD *)(v91 + 16);
        }
      }
      while ( v91 );
      if ( v92 == &v151 || v136 < v92[8] )
        goto LABEL_207;
    }
    else
    {
      v53 = v147;
      v54 = &v147[4 * (unsigned int)v148];
      if ( v147 != (_BYTE *)v54 )
      {
        while ( v136 != *v53 )
        {
          if ( v54 == ++v53 )
            goto LABEL_207;
        }
        if ( v53 != v54 )
          goto LABEL_112;
      }
LABEL_207:
      if ( v187 )
      {
        v93 = v184;
        if ( !v184 )
          goto LABEL_237;
        v94 = &v183;
        do
        {
          while ( 1 )
          {
            v95 = *(_QWORD *)(v93 + 16);
            v96 = *(_QWORD *)(v93 + 24);
            if ( v136 <= *(_DWORD *)(v93 + 32) )
              break;
            v93 = *(_QWORD *)(v93 + 24);
            if ( !v96 )
              goto LABEL_224;
          }
          v94 = (__int64 *)v93;
          v93 = *(_QWORD *)(v93 + 16);
        }
        while ( v95 );
LABEL_224:
        LOBYTE(v16) = 0;
        if ( v94 != &v183 )
          LOBYTE(v16) = v136 >= *((_DWORD *)v94 + 8);
      }
      else
      {
        v90 = v179;
        v16 = (__int64)&v179[4 * (unsigned int)v180];
        if ( v179 == (char *)v16 )
        {
LABEL_237:
          LOBYTE(v16) = 0;
        }
        else
        {
          while ( v136 != *(_DWORD *)v90 )
          {
            v90 += 4;
            if ( (char *)v16 == v90 )
              goto LABEL_237;
          }
          LOBYTE(v16) = v16 != (_QWORD)v90;
        }
      }
    }
LABEL_112:
    v137.m128i_i32[2] = v136;
    v137.m128i_i64[0] = 805306368;
    v138 = 0;
    *(__int32 *)((char *)v137.m128i_i32 + 3) = ((v16 & 1) << 6) | 0x30;
    *(__int32 *)((char *)v137.m128i_i32 + 2) = v137.m128i_i16[1] & 0xF00F;
    v139 = 0;
    v137.m128i_i32[0] &= 0xFFF000FF;
    v140 = 0;
    sub_2E8EAD0(v124, v123, &v137);
    goto LABEL_113;
  }
LABEL_114:
  v55 = v141;
  v56 = &v141[(unsigned int)v142];
  if ( v56 != v141 )
  {
    while ( 2 )
    {
      v57 = *v55;
      if ( v170 )
      {
        v86 = v167;
        if ( !v167 )
          goto LABEL_204;
        v87 = &v166;
        do
        {
          while ( 1 )
          {
            v88 = *(_QWORD *)(v86 + 16);
            v89 = *(_QWORD *)(v86 + 24);
            if ( v57 <= *(_DWORD *)(v86 + 32) )
              break;
            v86 = *(_QWORD *)(v86 + 24);
            if ( !v89 )
              goto LABEL_201;
          }
          v87 = (int *)v86;
          v86 = *(_QWORD *)(v86 + 16);
        }
        while ( v88 );
LABEL_201:
        v60 = 0;
        if ( v87 != &v166 )
          v60 = v57 >= v87[8];
      }
      else
      {
        v58 = v163;
        v59 = &v163[4 * (unsigned int)v164];
        if ( v163 == (_BYTE *)v59 )
        {
LABEL_204:
          v60 = 0;
        }
        else
        {
          while ( v57 != *v58 )
          {
            if ( v59 == ++v58 )
              goto LABEL_204;
          }
          v60 = v59 != v58;
        }
      }
      if ( v178 )
      {
        v82 = v175;
        if ( !v175 )
          goto LABEL_203;
        v83 = &v174;
        do
        {
          while ( 1 )
          {
            v84 = *(_QWORD *)(v82 + 16);
            v85 = *(_QWORD *)(v82 + 24);
            if ( v57 <= *(_DWORD *)(v82 + 32) )
              break;
            v82 = *(_QWORD *)(v82 + 24);
            if ( !v85 )
              goto LABEL_194;
          }
          v83 = (int *)v82;
          v82 = *(_QWORD *)(v82 + 16);
        }
        while ( v84 );
LABEL_194:
        v63 = 0;
        if ( v83 != &v174 )
          v63 = v57 >= v83[8];
      }
      else
      {
        v61 = v171;
        v62 = &v171[4 * (unsigned int)v172];
        if ( v171 == (_BYTE *)v62 )
        {
LABEL_203:
          v63 = 0;
        }
        else
        {
          while ( v57 != *v61 )
          {
            if ( v62 == ++v61 )
              goto LABEL_203;
          }
          v63 = v62 != v61;
        }
      }
      v137.m128i_i32[2] = *v55++;
      v137.m128i_i64[0] = 0x20000000;
      *(__int32 *)((char *)v137.m128i_i32 + 3) = (unsigned __int8)(v60 << 6) | 0x20;
      *(__int32 *)((char *)v137.m128i_i32 + 2) = v137.m128i_i16[1] & 0xF00F;
      v138 = 0;
      v137.m128i_i32[0] &= 0xFFF000FF;
      v137.m128i_i8[4] = v63;
      v139 = 0;
      v140 = 0;
      sub_2E8EAD0(v124, v123, &v137);
      if ( v56 == v55 )
        break;
      continue;
    }
  }
  if ( a2 != a3 )
  {
    v64 = a2;
    do
    {
      v65 = *(_DWORD *)(v64 + 44) & 0xFFFFFF;
      if ( (*(_DWORD *)(v64 + 44) & 1) != 0 )
      {
        *(_DWORD *)(v124 + 44) |= 1u;
        v65 = *(_DWORD *)(v64 + 44) & 0xFFFFFF;
      }
      if ( (v65 & 2) != 0 )
        *(_DWORD *)(v124 + 44) |= 2u;
      v64 = *(_QWORD *)(v64 + 8);
    }
    while ( a3 != v64 );
  }
  sub_2E91CC0(v204);
  if ( v200 != (unsigned __int8 *)v202 )
    _libc_free((unsigned __int64)v200);
  if ( v144 != v146 )
    _libc_free((unsigned __int64)v144);
  v66 = v175;
  while ( v66 )
  {
    sub_2E91CC0(*(_QWORD *)(v66 + 24));
    v67 = v66;
    v66 = *(_QWORD *)(v66 + 16);
    j_j___libc_free_0(v67);
  }
  if ( v171 != v173 )
    _libc_free((unsigned __int64)v171);
  v68 = v167;
  while ( v68 )
  {
    sub_2E91CC0(*(_QWORD *)(v68 + 24));
    v69 = v68;
    v68 = *(_QWORD *)(v68 + 16);
    j_j___libc_free_0(v69);
  }
  if ( v163 != v165 )
    _libc_free((unsigned __int64)v163);
  v70 = v159;
  while ( v70 )
  {
    sub_2E91CC0(*(_QWORD *)(v70 + 24));
    v71 = v70;
    v70 = *(_QWORD *)(v70 + 16);
    j_j___libc_free_0(v71);
  }
  if ( (_BYTE *)v156[0] != v157 )
    _libc_free(v156[0]);
  if ( v141 != (__int32 *)v143 )
    _libc_free((unsigned __int64)v141);
  sub_2E91CC0(v184);
  if ( v179 != v181 )
    _libc_free((unsigned __int64)v179);
  v72 = v152;
  while ( v72 )
  {
    sub_2E91CC0(*(_QWORD *)(v72 + 24));
    v73 = v72;
    v72 = *(_QWORD *)(v72 + 16);
    j_j___libc_free_0(v73);
  }
  if ( v147 != v149 )
    _libc_free((unsigned __int64)v147);
  sub_2E91CC0(v196);
  if ( v191 != v193 )
    _libc_free((unsigned __int64)v191);
  if ( v188 != v190 )
    _libc_free((unsigned __int64)v188);
}
