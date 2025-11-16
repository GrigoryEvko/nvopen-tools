// Function: sub_28FC5C0
// Address: 0x28fc5c0
//
__int64 __fastcall sub_28FC5C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  int v8; // eax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // rsi
  _BYTE *v22; // rax
  __int64 *v23; // r15
  unsigned __int64 *v24; // r14
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // r13
  unsigned __int8 *v30; // r15
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 v36; // rax
  _QWORD *v37; // rdx
  unsigned __int64 *v38; // r13
  __int64 v39; // rax
  __int64 v40; // rbx
  _QWORD *v41; // r15
  __int64 i; // r14
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rsi
  _QWORD *v49; // rax
  unsigned __int64 *v50; // r13
  __int64 v51; // r15
  _QWORD *v52; // rcx
  unsigned __int64 *v53; // r14
  _QWORD *v54; // rbx
  unsigned __int64 **v55; // r9
  unsigned __int64 v56; // rdx
  __int64 v57; // rax
  unsigned __int8 *v58; // r13
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  unsigned __int64 v62; // rdi
  unsigned __int64 *v63; // rax
  _QWORD *v64; // r9
  __int64 v65; // rcx
  __int64 v66; // rcx
  __int64 v67; // rsi
  unsigned int v68; // edx
  __int64 *v69; // r13
  __int64 v70; // r9
  __int64 v71; // rax
  char *v72; // rdx
  unsigned __int64 *v73; // rdi
  unsigned __int64 v74; // rax
  int v75; // r10d
  __int64 j; // rax
  int v77; // edx
  unsigned __int8 *v78; // r13
  unsigned __int64 *v79; // rsi
  int v80; // r9d
  __int64 v81; // rdi
  unsigned int v82; // edx
  _QWORD *v83; // rcx
  unsigned __int8 *v84; // r10
  __int64 v85; // rdx
  unsigned __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // rax
  __int64 *v92; // r13
  __int64 *v93; // rbx
  __int64 v94; // rax
  int v95; // eax
  __int64 v96; // rdx
  _QWORD *v97; // rax
  _QWORD *k; // rdx
  int v99; // r14d
  unsigned int v100; // eax
  _QWORD *v101; // r13
  __int64 v102; // rdx
  __int64 v103; // r15
  _QWORD *v104; // r12
  __int64 m; // r14
  __int64 v106; // rax
  __int64 v107; // r14
  __int64 v108; // r13
  int v109; // edx
  _QWORD *v110; // r12
  unsigned int v111; // eax
  __int64 v112; // rbx
  _QWORD *v113; // r15
  __int64 v114; // rax
  __int64 v115; // rax
  __int64 v116; // rax
  __int64 v117; // rax
  __int64 v118; // r8
  int v119; // r12d
  unsigned int v120; // edx
  unsigned int v121; // eax
  _QWORD *v122; // rdi
  unsigned __int64 v123; // rax
  __int64 v124; // rax
  _QWORD *v125; // rax
  __int64 v126; // rdx
  _QWORD *n; // rdx
  void *v128; // r13
  void *v129; // r12
  int v131; // ecx
  int v132; // r8d
  unsigned int v133; // ecx
  unsigned int v134; // eax
  int v135; // eax
  unsigned __int64 v136; // r12
  unsigned int v137; // eax
  _QWORD *v138; // rax
  __int64 v139; // rax
  int v140; // edx
  int v141; // r13d
  unsigned int v142; // eax
  unsigned int v143; // eax
  __int64 v145; // [rsp+18h] [rbp-AF8h]
  unsigned __int64 *v146; // [rsp+28h] [rbp-AE8h]
  __int64 *v147; // [rsp+30h] [rbp-AE0h]
  __int64 v148; // [rsp+38h] [rbp-AD8h]
  unsigned __int64 *v149; // [rsp+48h] [rbp-AC8h]
  unsigned __int64 **v150; // [rsp+48h] [rbp-AC8h]
  __int64 v151; // [rsp+48h] [rbp-AC8h]
  unsigned __int64 *v152; // [rsp+50h] [rbp-AC0h]
  __int64 *v153; // [rsp+50h] [rbp-AC0h]
  _QWORD *v154; // [rsp+50h] [rbp-AC0h]
  __int64 v155; // [rsp+50h] [rbp-AC0h]
  __int64 *v156; // [rsp+58h] [rbp-AB8h]
  __int64 v157; // [rsp+58h] [rbp-AB8h]
  _QWORD *v158; // [rsp+58h] [rbp-AB8h]
  _QWORD *v159; // [rsp+58h] [rbp-AB8h]
  unsigned __int64 v160; // [rsp+60h] [rbp-AB0h]
  _BYTE *v161; // [rsp+60h] [rbp-AB0h]
  __int64 v162; // [rsp+60h] [rbp-AB0h]
  _QWORD *v163; // [rsp+68h] [rbp-AA8h]
  int v164; // [rsp+68h] [rbp-AA8h]
  int v165; // [rsp+68h] [rbp-AA8h]
  _BYTE *v166; // [rsp+70h] [rbp-AA0h] BYREF
  __int64 v167; // [rsp+78h] [rbp-A98h]
  _BYTE v168[64]; // [rsp+80h] [rbp-A90h] BYREF
  unsigned __int64 v169[54]; // [rsp+C0h] [rbp-A50h] BYREF
  __int64 v170; // [rsp+270h] [rbp-8A0h] BYREF
  __int64 *v171; // [rsp+278h] [rbp-898h]
  int v172; // [rsp+280h] [rbp-890h]
  int v173; // [rsp+284h] [rbp-88Ch]
  int v174; // [rsp+288h] [rbp-888h]
  char v175; // [rsp+28Ch] [rbp-884h]
  __int64 v176; // [rsp+290h] [rbp-880h] BYREF
  unsigned __int64 *v177; // [rsp+2D0h] [rbp-840h]
  __int64 v178; // [rsp+2D8h] [rbp-838h]
  unsigned __int64 v179; // [rsp+2E0h] [rbp-830h] BYREF
  int v180; // [rsp+2E8h] [rbp-828h]
  unsigned __int64 v181; // [rsp+2F0h] [rbp-820h]
  int v182; // [rsp+2F8h] [rbp-818h]
  __int64 v183; // [rsp+300h] [rbp-810h]
  char v184[8]; // [rsp+420h] [rbp-6F0h] BYREF
  unsigned __int64 v185; // [rsp+428h] [rbp-6E8h]
  char v186; // [rsp+43Ch] [rbp-6D4h]
  char *v187; // [rsp+480h] [rbp-690h]
  char v188; // [rsp+490h] [rbp-680h] BYREF
  __int64 v189; // [rsp+5D0h] [rbp-540h] BYREF
  unsigned __int64 v190; // [rsp+5D8h] [rbp-538h]
  __int64 v191; // [rsp+5E0h] [rbp-530h]
  char v192; // [rsp+5ECh] [rbp-524h]
  char *v193; // [rsp+630h] [rbp-4E0h]
  char v194; // [rsp+640h] [rbp-4D0h] BYREF
  __int64 v195; // [rsp+780h] [rbp-390h] BYREF
  unsigned __int64 v196; // [rsp+788h] [rbp-388h]
  __int64 v197; // [rsp+790h] [rbp-380h]
  unsigned __int64 *v198; // [rsp+798h] [rbp-378h]
  char *v199; // [rsp+7E0h] [rbp-330h]
  char v200; // [rsp+7F0h] [rbp-320h] BYREF
  __int64 v201; // [rsp+930h] [rbp-1E0h] BYREF
  __int64 *v202; // [rsp+938h] [rbp-1D8h]
  __int64 v203; // [rsp+940h] [rbp-1D0h]
  __int64 v204; // [rsp+948h] [rbp-1C8h]
  __int64 v205[2]; // [rsp+950h] [rbp-1C0h] BYREF
  unsigned __int64 *v206; // [rsp+960h] [rbp-1B0h] BYREF
  unsigned __int64 **v207; // [rsp+968h] [rbp-1A8h]
  __int64 v208; // [rsp+970h] [rbp-1A0h]
  __int64 v209; // [rsp+978h] [rbp-198h]
  unsigned __int64 *v210; // [rsp+980h] [rbp-190h] BYREF
  unsigned __int64 *v211; // [rsp+988h] [rbp-188h]
  char *v212; // [rsp+990h] [rbp-180h]
  _QWORD *v213; // [rsp+998h] [rbp-178h]
  char v214; // [rsp+9A0h] [rbp-170h] BYREF

  v4 = a2;
  v5 = *(_QWORD *)(a3 + 80);
  v166 = v168;
  memset(v169, 0, sizeof(v169));
  v178 = 0x800000000LL;
  v169[1] = (unsigned __int64)&v169[4];
  v169[12] = (unsigned __int64)&v169[14];
  if ( v5 )
    v5 -= 24;
  HIDWORD(v169[13]) = 8;
  v171 = &v176;
  v177 = &v179;
  v6 = *(_QWORD *)(v5 + 48);
  v167 = 0x800000000LL;
  v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
  LODWORD(v169[2]) = 8;
  BYTE4(v169[3]) = 1;
  v172 = 8;
  v174 = 0;
  v175 = 1;
  v173 = 1;
  v176 = v5;
  v170 = 1;
  if ( v7 == v5 + 48 )
    goto LABEL_225;
  if ( !v7 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
  {
LABEL_225:
    v8 = 0;
    v10 = 0;
    v9 = 0;
  }
  else
  {
    v160 = v7 - 24;
    v8 = sub_B46E30(v7 - 24);
    v9 = v160;
    v10 = v160;
  }
  v179 = v10;
  v181 = v9;
  v180 = v8;
  v183 = v5;
  v182 = 0;
  LODWORD(v178) = 1;
  sub_CE27D0((__int64)&v170);
  sub_CE3710((__int64)&v195, (__int64)v169, v11, v12, v13, v14);
  sub_CE35F0((__int64)&v201, (__int64)&v195);
  sub_CE3710((__int64)v184, (__int64)&v170, v15, v16, (__int64)v184, (__int64)&v170);
  sub_CE35F0((__int64)&v189, (__int64)v184);
  sub_CE37E0((__int64)&v189, (__int64)&v201, (__int64)&v166, v17, v18, v19);
  if ( v193 != &v194 )
    _libc_free((unsigned __int64)v193);
  if ( !v192 )
    _libc_free(v190);
  if ( v187 != &v188 )
    _libc_free((unsigned __int64)v187);
  if ( !v186 )
    _libc_free(v185);
  if ( v212 != &v214 )
    _libc_free((unsigned __int64)v212);
  if ( !BYTE4(v204) )
    _libc_free((unsigned __int64)v202);
  if ( v199 != &v200 )
    _libc_free((unsigned __int64)v199);
  if ( !BYTE4(v198) )
    _libc_free(v196);
  if ( v177 != &v179 )
    _libc_free((unsigned __int64)v177);
  if ( !v175 )
    _libc_free((unsigned __int64)v171);
  if ( (unsigned __int64 *)v169[12] != &v169[14] )
    _libc_free(v169[12]);
  if ( !BYTE4(v169[3]) )
    _libc_free(v169[1]);
  sub_28EFD40(a2, a3, (__int64)&v166, v20);
  sub_28F0920(a2, (__int64 *)&v166);
  v21 = (__int64)v166;
  v22 = &v166[8 * (unsigned int)v167];
  *(_BYTE *)(v4 + 752) = 0;
  v145 = v21;
  v161 = v22;
  v163 = (_QWORD *)(v4 + 96);
  if ( (_BYTE *)v21 != v22 )
  {
    v23 = &v195;
    v24 = (unsigned __int64 *)&v189;
    do
    {
      v25 = *((_QWORD *)v161 - 1);
      v26 = *(_QWORD *)(v25 + 56);
      v27 = v25 + 48;
      if ( v27 != v26 )
      {
        v152 = v24;
        v28 = v26;
        v29 = v27;
        v156 = v23;
        do
        {
          while ( 1 )
          {
            v30 = (unsigned __int8 *)(v28 - 24);
            if ( !v28 )
              v30 = 0;
            if ( !sub_F50EE0(v30, 0) )
              break;
            v28 = *(_QWORD *)(v28 + 8);
            sub_28FB6F0(v4, (__int64 *)v30, v31, v32, v33, v34);
            if ( v29 == v28 )
              goto LABEL_40;
          }
          sub_28F9970(v4, (__int64)v30);
          v28 = *(_QWORD *)(v28 + 8);
        }
        while ( v29 != v28 );
LABEL_40:
        v23 = v156;
        v24 = v152;
      }
      v201 = 0;
      v202 = 0;
      v203 = 0;
      v204 = 0;
      sub_C7D6A0(0, 0, 8);
      v35 = *(unsigned int *)(v4 + 88);
      LODWORD(v204) = v35;
      if ( (_DWORD)v35 )
      {
        v36 = sub_C7D670(24 * v35, 8);
        v37 = *(_QWORD **)(v4 + 72);
        v202 = (__int64 *)v36;
        v38 = (unsigned __int64 *)v36;
        v39 = *(_QWORD *)(v4 + 80);
        v189 = 0;
        v203 = v39;
        v190 = 0;
        v191 = -4096;
        v195 = 0;
        v196 = 0;
        v197 = -8192;
        if ( (_DWORD)v204 )
        {
          v157 = v4;
          v40 = (unsigned int)v204;
          v153 = v23;
          v41 = v37;
          v149 = v24;
          for ( i = 0; i != v40; ++i )
          {
            *v38 = 0;
            v38[1] = 0;
            v43 = v41[2];
            v38[2] = v43;
            if ( v43 != 0 && v43 != -4096 && v43 != -8192 )
              sub_BD6050(v38, *v41 & 0xFFFFFFFFFFFFFFF8LL);
            v38 += 3;
            v41 += 3;
          }
          v4 = v157;
          v23 = v153;
          v24 = v149;
        }
        sub_D68D70(v23);
        sub_D68D70(v24);
      }
      else
      {
        v202 = 0;
        v203 = 0;
      }
      v44 = *(_QWORD *)(v4 + 168) - *(_QWORD *)(v4 + 136);
      v205[0] = 0;
      v205[1] = 0;
      v206 = 0;
      v207 = 0;
      v45 = 21 * ((v44 >> 3) - 1);
      v46 = *(_QWORD *)(v4 + 144) - *(_QWORD *)(v4 + 152);
      v208 = 0;
      v209 = 0;
      v210 = 0;
      v211 = 0;
      v212 = 0;
      v47 = v45 - 0x5555555555555555LL * (v46 >> 3);
      v48 = *(_QWORD *)(v4 + 128) - *(_QWORD *)(v4 + 112);
      v213 = 0;
      sub_2350260(v205, v47 - 0x5555555555555555LL * (v48 >> 3));
      v49 = *(_QWORD **)(v4 + 112);
      v148 = v4;
      v147 = v23;
      v50 = v206;
      v51 = *(_QWORD *)(v4 + 136);
      v146 = v24;
      v52 = *(_QWORD **)(v4 + 128);
      v53 = (unsigned __int64 *)v208;
      v54 = *(_QWORD **)(v4 + 144);
      v55 = (unsigned __int64 **)(v209 + 8);
      while ( v54 != v49 )
      {
        while ( 1 )
        {
          if ( v50 )
          {
            *v50 = 0;
            v50[1] = 0;
            v56 = v49[2];
            v50[2] = v56;
            if ( v56 != -4096 && v56 != 0 && v56 != -8192 )
            {
              v150 = v55;
              v154 = v52;
              v158 = v49;
              sub_BD6050(v50, *v49 & 0xFFFFFFFFFFFFFFF8LL);
              v55 = v150;
              v52 = v154;
              v49 = v158;
            }
          }
          v49 += 3;
          if ( v49 == v52 )
          {
            v49 = *(_QWORD **)(v51 + 8);
            v51 += 8;
            v52 = v49 + 63;
          }
          v50 += 3;
          if ( v53 != v50 )
            break;
          v50 = *v55++;
          v53 = v50 + 63;
          if ( v54 == v49 )
            goto LABEL_60;
        }
      }
LABEL_60:
      v4 = v148;
      v23 = v147;
      v24 = v146;
      while ( 1 )
      {
        v62 = (unsigned __int64)v210;
        if ( v210 == v206 )
          break;
        v63 = v211;
        v64 = v213;
        v65 = (__int64)v210;
        if ( v210 == v211 )
          v65 = *(v213 - 1) + 504LL;
        v189 = 0;
        v190 = 0;
        v191 = *(_QWORD *)(v65 - 8);
        if ( v191 != 0 && v191 != -4096 && v191 != -8192 )
        {
          sub_BD6050(v146, *(_QWORD *)(v65 - 24) & 0xFFFFFFFFFFFFFFF8LL);
          v62 = (unsigned __int64)v210;
          v63 = v211;
          v64 = v213;
        }
        v66 = v62;
        if ( v63 == (unsigned __int64 *)v62 )
          v66 = *(v64 - 1) + 504LL;
        if ( (_DWORD)v204 )
        {
          v67 = *(_QWORD *)(v66 - 8);
          v68 = (v204 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
          v69 = &v202[3 * v68];
          v70 = v69[2];
          if ( v67 == v70 )
          {
LABEL_79:
            v195 = 0;
            v196 = 0;
            v197 = -8192;
            v71 = v69[2];
            if ( v71 != -8192 )
            {
              if ( v71 != -4096 && v71 )
                sub_BD60C0(v69);
              v69[2] = -8192;
              if ( v197 != 0 && v197 != -4096 && v197 != -8192 )
                sub_BD60C0(v147);
            }
            LODWORD(v203) = v203 - 1;
            v62 = (unsigned __int64)v210;
            ++HIDWORD(v203);
            v63 = v211;
          }
          else
          {
            v75 = 1;
            while ( v70 != -4096 )
            {
              v68 = (v204 - 1) & (v75 + v68);
              v69 = &v202[3 * v68];
              v70 = v69[2];
              if ( v67 == v70 )
                goto LABEL_79;
              ++v75;
            }
          }
        }
        if ( (unsigned __int64 *)v62 == v63 )
        {
          j_j___libc_free_0(v62);
          v72 = (char *)(*--v213 + 504LL);
          v73 = (unsigned __int64 *)(*v213 + 480LL);
          v211 = (unsigned __int64 *)*v213;
          v212 = v72;
          v210 = v73;
          v74 = v211[62];
          if ( v74 != 0 && v74 != -4096 && v74 != -8192 )
            sub_BD60C0(v73);
        }
        else
        {
          v210 = (unsigned __int64 *)(v62 - 24);
          v57 = *(_QWORD *)(v62 - 8);
          if ( v57 != 0 && v57 != -4096 && v57 != -8192 )
            sub_BD60C0((_QWORD *)(v62 - 24));
        }
        v58 = (unsigned __int8 *)v191;
        if ( v191 != 0 && v191 != -4096 && v191 != -8192 )
          sub_BD60C0(v146);
        if ( sub_F50EE0(v58, 0) )
        {
          sub_28FBEB0(v148, (__int64)v58, (__int64)&v201, v59, v60, v61);
          *(_BYTE *)(v148 + 752) = 1;
        }
      }
      for ( j = *(_QWORD *)(v148 + 112); *(_QWORD *)(v148 + 144) != j; j = *(_QWORD *)(v148 + 112) )
      {
        while ( 1 )
        {
          v77 = *(_DWORD *)(v148 + 88);
          v78 = *(unsigned __int8 **)(j + 16);
          v79 = *(unsigned __int64 **)(v148 + 136);
          if ( v77 )
          {
            v80 = v77 - 1;
            v81 = *(_QWORD *)(v148 + 72);
            v82 = (v77 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
            v83 = (_QWORD *)(v81 + 24LL * v82);
            v84 = (unsigned __int8 *)v83[2];
            if ( v78 == v84 )
            {
LABEL_100:
              v195 = 0;
              v196 = 0;
              v197 = -8192;
              v85 = v83[2];
              if ( v85 != -8192 )
              {
                if ( v85 != -4096 && v85 )
                {
                  v151 = j;
                  v159 = v83;
                  sub_BD60C0(v83);
                  j = v151;
                  v83 = v159;
                }
                v83[2] = -8192;
                if ( v197 != 0 && v197 != -4096 && v197 != -8192 )
                {
                  v155 = j;
                  sub_BD60C0(v147);
                  j = v155;
                }
              }
              --*(_DWORD *)(v148 + 80);
              ++*(_DWORD *)(v148 + 84);
            }
            else
            {
              v131 = 1;
              while ( v84 != (unsigned __int8 *)-4096LL )
              {
                v132 = v131 + 1;
                v82 = v80 & (v131 + v82);
                v83 = (_QWORD *)(v81 + 24LL * v82);
                v84 = (unsigned __int8 *)v83[2];
                if ( v78 == v84 )
                  goto LABEL_100;
                v131 = v132;
              }
            }
          }
          v195 = j;
          v86 = *v79;
          v198 = v79;
          v196 = v86;
          v197 = v86 + 504;
          sub_28FB170((__int64 *)v146, v163, (__int64)v147);
          if ( !sub_F50EE0(v78, 0) )
            break;
          sub_28FB6F0(v148, (__int64 *)v78, v87, v88, v89, v90);
          j = *(_QWORD *)(v148 + 112);
          if ( *(_QWORD *)(v148 + 144) == j )
            goto LABEL_110;
        }
        sub_28F9970(v148, (__int64)v78);
      }
LABEL_110:
      sub_28EDAB0((unsigned __int64 *)v205);
      v91 = (unsigned int)v204;
      if ( (_DWORD)v204 )
      {
        v92 = v202;
        v189 = 0;
        v190 = 0;
        v191 = -4096;
        v195 = 0;
        v196 = 0;
        v197 = -8192;
        v93 = &v202[3 * (unsigned int)v204];
        do
        {
          v94 = v92[2];
          if ( v94 != 0 && v94 != -4096 && v94 != -8192 )
            sub_BD60C0(v92);
          v92 += 3;
        }
        while ( v93 != v92 );
        v4 = v148;
        if ( v197 != 0 && v197 != -4096 && v197 != -8192 )
          sub_BD60C0(v147);
        if ( v191 != 0 && v191 != -4096 && v191 != -8192 )
          sub_BD60C0(v146);
        v91 = (unsigned int)v204;
      }
      sub_C7D6A0((__int64)v202, 24 * v91, 8);
      v161 -= 8;
    }
    while ( (_BYTE *)v145 != v161 );
  }
  v95 = *(_DWORD *)(v4 + 16);
  ++*(_QWORD *)v4;
  if ( v95 )
  {
    v133 = 4 * v95;
    v96 = *(unsigned int *)(v4 + 24);
    if ( (unsigned int)(4 * v95) < 0x40 )
      v133 = 64;
    if ( (unsigned int)v96 <= v133 )
      goto LABEL_127;
    v134 = v95 - 1;
    if ( v134 )
    {
      _BitScanReverse(&v134, v134);
      v135 = 1 << (33 - (v134 ^ 0x1F));
      if ( v135 < 64 )
        v135 = 64;
      if ( v135 == (_DWORD)v96 )
        goto LABEL_218;
      v136 = 4 * v135 / 3u + 1;
    }
    else
    {
      v136 = 86;
    }
    sub_C7D6A0(*(_QWORD *)(v4 + 8), 16LL * *(unsigned int *)(v4 + 24), 8);
    v137 = sub_AF1560(v136);
    *(_DWORD *)(v4 + 24) = v137;
    if ( !v137 )
      goto LABEL_250;
    *(_QWORD *)(v4 + 8) = sub_C7D670(16LL * v137, 8);
LABEL_218:
    sub_28EEF70(v4);
    goto LABEL_130;
  }
  if ( *(_DWORD *)(v4 + 20) )
  {
    v96 = *(unsigned int *)(v4 + 24);
    if ( (unsigned int)v96 <= 0x40 )
    {
LABEL_127:
      v97 = *(_QWORD **)(v4 + 8);
      for ( k = &v97[2 * v96]; k != v97; v97 += 2 )
        *v97 = -4096;
      goto LABEL_129;
    }
    sub_C7D6A0(*(_QWORD *)(v4 + 8), 16LL * *(unsigned int *)(v4 + 24), 8);
    *(_DWORD *)(v4 + 24) = 0;
LABEL_250:
    *(_QWORD *)(v4 + 8) = 0;
LABEL_129:
    *(_QWORD *)(v4 + 16) = 0;
  }
LABEL_130:
  v99 = *(_DWORD *)(v4 + 48);
  ++*(_QWORD *)(v4 + 32);
  if ( !v99 && !*(_DWORD *)(v4 + 52) )
    goto LABEL_147;
  v100 = 4 * v99;
  v101 = *(_QWORD **)(v4 + 40);
  v102 = *(unsigned int *)(v4 + 56);
  v103 = 32 * v102;
  if ( (unsigned int)(4 * v99) < 0x40 )
    v100 = 64;
  v104 = &v101[(unsigned __int64)v103 / 8];
  if ( (unsigned int)v102 <= v100 )
  {
    v201 = 0;
    v202 = 0;
    v203 = -4096;
    if ( v101 != v104 )
    {
      for ( m = -4096; ; m = v203 )
      {
        v106 = v101[2];
        if ( m != v106 )
        {
          if ( v106 != -4096 && v106 != 0 && v106 != -8192 )
            sub_BD60C0(v101);
          v101[2] = m;
          if ( m != 0 && m != -4096 && m != -8192 )
            sub_BD73F0((__int64)v101);
        }
        v101 += 4;
        if ( v101 == v104 )
          break;
      }
    }
    *(_QWORD *)(v4 + 48) = 0;
    sub_D68D70(&v201);
    goto LABEL_147;
  }
  v195 = 0;
  v196 = 0;
  v197 = -4096;
  v201 = 0;
  v202 = 0;
  v203 = -8192;
  do
  {
    v139 = v101[2];
    if ( v139 != 0 && v139 != -4096 && v139 != -8192 )
      sub_BD60C0(v101);
    v101 += 4;
  }
  while ( v101 != v104 );
  if ( v203 != -4096 && v203 != 0 && v203 != -8192 )
    sub_BD60C0(&v201);
  if ( v197 != 0 && v197 != -4096 && v197 != -8192 )
    sub_BD60C0(&v195);
  v140 = *(_DWORD *)(v4 + 56);
  if ( !v99 )
  {
    if ( !v140 )
      goto LABEL_248;
    sub_C7D6A0(*(_QWORD *)(v4 + 40), v103, 8);
    *(_DWORD *)(v4 + 56) = 0;
LABEL_253:
    *(_QWORD *)(v4 + 40) = 0;
    *(_QWORD *)(v4 + 48) = 0;
    goto LABEL_147;
  }
  v141 = 64;
  if ( v99 != 1 )
  {
    _BitScanReverse(&v142, v99 - 1);
    v141 = 1 << (33 - (v142 ^ 0x1F));
    if ( v141 < 64 )
      v141 = 64;
  }
  if ( v141 == v140 )
    goto LABEL_248;
  sub_C7D6A0(*(_QWORD *)(v4 + 40), v103, 8);
  v143 = sub_AF1560(4 * v141 / 3u + 1);
  *(_DWORD *)(v4 + 56) = v143;
  if ( !v143 )
    goto LABEL_253;
  *(_QWORD *)(v4 + 40) = sub_C7D670(32LL * v143, 8);
LABEL_248:
  sub_28EEFB0(v4 + 32);
LABEL_147:
  v162 = v4;
  v107 = v4 + 176;
  v108 = v4 + 752;
  do
  {
    while ( 1 )
    {
LABEL_148:
      v109 = *(_DWORD *)(v107 + 16);
      ++*(_QWORD *)v107;
      if ( !v109 && !*(_DWORD *)(v107 + 20) )
        goto LABEL_187;
      v110 = *(_QWORD **)(v107 + 8);
      v111 = 4 * v109;
      v112 = 72LL * *(unsigned int *)(v107 + 24);
      if ( (unsigned int)(4 * v109) < 0x40 )
        v111 = 64;
      v113 = &v110[(unsigned __int64)v112 / 8];
      if ( *(_DWORD *)(v107 + 24) > v111 )
        break;
      while ( v113 != v110 )
      {
        if ( *v110 != -4096 )
        {
          if ( *v110 != -8192 || v110[1] != -8192 )
          {
LABEL_155:
            v114 = v110[7];
            if ( v114 != 0 && v114 != -4096 && v114 != -8192 )
              sub_BD60C0(v110 + 5);
            v115 = v110[4];
            if ( v115 != 0 && v115 != -4096 && v115 != -8192 )
              sub_BD60C0(v110 + 2);
          }
          *v110 = -4096;
          v110[1] = -4096;
          goto LABEL_162;
        }
        if ( v110[1] != -4096 )
          goto LABEL_155;
LABEL_162:
        v110 += 9;
      }
LABEL_198:
      *(_DWORD *)(v107 + 16) = 0;
      v107 += 32;
      *(_DWORD *)(v107 - 12) = 0;
      if ( v108 == v107 )
        goto LABEL_188;
    }
    do
    {
      while ( *v110 != -4096 )
      {
        if ( *v110 != -8192 || v110[1] != -8192 )
        {
LABEL_168:
          v116 = v110[7];
          if ( v116 != 0 && v116 != -4096 && v116 != -8192 )
          {
            v164 = v109;
            sub_BD60C0(v110 + 5);
            v109 = v164;
          }
          v117 = v110[4];
          if ( v117 != 0 && v117 != -4096 && v117 != -8192 )
          {
            v165 = v109;
            sub_BD60C0(v110 + 2);
            v109 = v165;
          }
        }
        v110 += 9;
        if ( v113 == v110 )
          goto LABEL_178;
      }
      if ( v110[1] != -4096 )
        goto LABEL_168;
      v110 += 9;
    }
    while ( v113 != v110 );
LABEL_178:
    v118 = *(unsigned int *)(v107 + 24);
    if ( !v109 )
    {
      if ( (_DWORD)v118 )
      {
        sub_C7D6A0(*(_QWORD *)(v107 + 8), v112, 8);
        *(_DWORD *)(v107 + 24) = 0;
        *(_QWORD *)(v107 + 8) = 0;
      }
      goto LABEL_198;
    }
    v119 = 64;
    v120 = v109 - 1;
    if ( v120 )
    {
      _BitScanReverse(&v121, v120);
      v119 = 1 << (33 - (v121 ^ 0x1F));
      if ( v119 < 64 )
        v119 = 64;
    }
    v122 = *(_QWORD **)(v107 + 8);
    if ( (_DWORD)v118 == v119 )
    {
      *(_DWORD *)(v107 + 16) = 0;
      *(_DWORD *)(v107 + 20) = 0;
      v138 = &v122[9 * v118];
      do
      {
        if ( v122 )
        {
          *v122 = -4096;
          v122[1] = -4096;
        }
        v122 += 9;
      }
      while ( v138 != v122 );
      v107 += 32;
      if ( v108 == v107 )
        break;
      goto LABEL_148;
    }
    sub_C7D6A0((__int64)v122, v112, 8);
    v123 = (((((((4 * v119 / 3u + 1) | ((unsigned __int64)(4 * v119 / 3u + 1) >> 1)) >> 2)
             | (4 * v119 / 3u + 1)
             | ((unsigned __int64)(4 * v119 / 3u + 1) >> 1)) >> 4)
           | (((4 * v119 / 3u + 1) | ((unsigned __int64)(4 * v119 / 3u + 1) >> 1)) >> 2)
           | (4 * v119 / 3u + 1)
           | ((unsigned __int64)(4 * v119 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v119 / 3u + 1) | ((unsigned __int64)(4 * v119 / 3u + 1) >> 1)) >> 2)
           | (4 * v119 / 3u + 1)
           | ((unsigned __int64)(4 * v119 / 3u + 1) >> 1)) >> 4)
         | (((4 * v119 / 3u + 1) | ((unsigned __int64)(4 * v119 / 3u + 1) >> 1)) >> 2)
         | (4 * v119 / 3u + 1)
         | ((unsigned __int64)(4 * v119 / 3u + 1) >> 1);
    v124 = ((v123 >> 16) | v123) + 1;
    *(_DWORD *)(v107 + 24) = v124;
    v125 = (_QWORD *)sub_C7D670(72 * v124, 8);
    v126 = *(unsigned int *)(v107 + 24);
    *(_DWORD *)(v107 + 16) = 0;
    *(_QWORD *)(v107 + 8) = v125;
    *(_DWORD *)(v107 + 20) = 0;
    for ( n = &v125[9 * v126]; n != v125; v125 += 9 )
    {
      if ( v125 )
      {
        *v125 = -4096;
        v125[1] = -4096;
      }
    }
LABEL_187:
    v107 += 32;
  }
  while ( v108 != v107 );
LABEL_188:
  v128 = (void *)(a1 + 32);
  v129 = (void *)(a1 + 80);
  if ( *(_BYTE *)(v162 + 752) )
  {
    v201 = 0;
    v202 = v205;
    v203 = 2;
    LODWORD(v204) = 0;
    BYTE4(v204) = 1;
    v206 = 0;
    v207 = &v210;
    v208 = 2;
    LODWORD(v209) = 0;
    BYTE4(v209) = 1;
    sub_AE6EC0((__int64)&v201, (__int64)&unk_4F82408);
    sub_C8CF70(a1, v128, 2, (__int64)v205, (__int64)&v201);
    sub_C8CF70(a1 + 48, v129, 2, (__int64)&v210, (__int64)&v206);
    if ( BYTE4(v209) )
    {
      if ( BYTE4(v204) )
        goto LABEL_191;
LABEL_226:
      _libc_free((unsigned __int64)v202);
    }
    else
    {
      _libc_free((unsigned __int64)v207);
      if ( !BYTE4(v204) )
        goto LABEL_226;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v128;
    *(_QWORD *)(a1 + 56) = v129;
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 2;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    sub_AE6EC0(a1, (__int64)&qword_4F82400);
  }
LABEL_191:
  if ( v166 != v168 )
    _libc_free((unsigned __int64)v166);
  return a1;
}
