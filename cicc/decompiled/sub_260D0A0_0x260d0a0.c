// Function: sub_260D0A0
// Address: 0x260d0a0
//
__int64 __fastcall sub_260D0A0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 *v4; // r12
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  _DWORD *v9; // rbx
  _DWORD *v10; // rdx
  __int64 **v11; // r14
  __int64 v12; // r15
  unsigned int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // rbx
  unsigned int v16; // ecx
  char *v17; // rdi
  unsigned int v18; // edx
  int v19; // r12d
  unsigned int v20; // eax
  _DWORD *v21; // rax
  _DWORD *i; // rdx
  __int64 **v23; // rax
  __int64 **v24; // r12
  int v25; // r13d
  __int64 v26; // rdx
  int v27; // r11d
  unsigned __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 *v30; // rdi
  __int64 **v31; // rsi
  __int64 **v32; // rdx
  __int64 **v33; // rcx
  int v34; // eax
  bool v35; // sf
  bool v36; // of
  __int64 v37; // rax
  _BYTE *v38; // rsi
  char *v39; // rbx
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 **v42; // rax
  __int64 **v43; // r15
  __int64 **v44; // rbx
  __int64 **v45; // rsi
  unsigned __int64 v46; // rdi
  int v47; // eax
  bool v48; // cc
  unsigned __int64 v49; // rdi
  _QWORD **v50; // r12
  _QWORD **v51; // rbx
  _QWORD *v52; // rdi
  unsigned int v53; // r13d
  unsigned __int64 m; // r12
  __int64 v55; // rax
  __int64 v56; // rbx
  __int64 v57; // r15
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // rdi
  __int64 **v61; // rax
  __int64 **v62; // r15
  __int64 **v63; // rbx
  __int64 **v64; // rsi
  unsigned __int64 v65; // rdi
  __int64 **v66; // rbx
  __int64 **v67; // rax
  __int64 *v68; // rax
  int v69; // r14d
  int v70; // r12d
  __int64 v71; // rdx
  unsigned __int64 v72; // rax
  __int64 *v73; // rsi
  unsigned int v74; // eax
  unsigned int v75; // r12d
  unsigned int v76; // r12d
  __int64 v77; // r8
  unsigned int v78; // edx
  unsigned int *v79; // rdi
  unsigned int v80; // ecx
  unsigned int v81; // esi
  unsigned int *v82; // r10
  int v83; // edx
  __int64 **v84; // rsi
  int v85; // r11d
  int v86; // eax
  unsigned __int64 v87; // rdi
  __int64 v88; // rax
  __int64 *v89; // rax
  __int64 v90; // r14
  __int64 v91; // rax
  unsigned __int64 v92; // rax
  unsigned __int64 v93; // rbx
  unsigned __int64 v94; // rdx
  unsigned __int64 v95; // rsi
  unsigned __int64 v96; // rcx
  int v97; // eax
  __int64 *v98; // rsi
  unsigned int v99; // ecx
  unsigned __int64 v100; // r10
  __int64 v101; // rax
  unsigned __int64 v102; // rdx
  unsigned __int64 v103; // r10
  __int64 v104; // rax
  __int64 v105; // rbx
  __int64 v106; // rbx
  __int64 v107; // rdx
  unsigned int v108; // eax
  __int64 v109; // rcx
  __int64 v110; // rcx
  __int64 v111; // rax
  __int64 v112; // rbx
  __int64 *v113; // rsi
  __int64 k; // rbx
  __int64 *v115; // rsi
  unsigned __int64 *v116; // rbx
  unsigned __int64 *v117; // r14
  unsigned __int64 v118; // rdi
  __int64 *v119; // rdi
  __int64 v120; // r15
  char *v121; // rax
  unsigned __int64 v122; // r12
  unsigned __int64 *v123; // r12
  unsigned __int64 *v124; // r14
  unsigned __int64 v125; // rbx
  unsigned __int64 v126; // r15
  __int64 v127; // rsi
  __int64 v128; // rdi
  __int64 *v129; // rax
  __int64 v130; // r12
  __int64 v131; // rax
  unsigned __int64 v132; // rax
  unsigned __int64 v133; // r13
  unsigned __int64 v134; // rdx
  unsigned __int64 v135; // rsi
  unsigned __int64 v136; // rcx
  int v137; // eax
  __int64 *v138; // rsi
  unsigned int v139; // ecx
  __int64 v140; // rdx
  unsigned __int64 v141; // rax
  char v142; // r10
  __int64 v143; // r9
  __int64 *v144; // r13
  __int64 v145; // rdx
  unsigned int v146; // eax
  __int64 v147; // rcx
  __int64 v148; // rcx
  __int64 v149; // rax
  __int64 **v150; // rax
  __int64 **v151; // rbx
  __int64 *v152; // rsi
  __int64 **j; // rbx
  __int64 *v154; // rsi
  unsigned __int64 *v155; // rbx
  unsigned __int64 *v156; // r12
  unsigned __int64 v157; // rdi
  __int64 v158; // rax
  __int64 v159; // rax
  char *v160; // r8
  __int64 v161; // rax
  __int64 v162; // rax
  unsigned __int64 v163; // [rsp+0h] [rbp-3B0h]
  __int64 v164; // [rsp+10h] [rbp-3A0h]
  unsigned __int64 v165; // [rsp+18h] [rbp-398h]
  __int64 v166; // [rsp+18h] [rbp-398h]
  _DWORD *v167; // [rsp+20h] [rbp-390h]
  __int64 v168; // [rsp+28h] [rbp-388h]
  char *v169; // [rsp+30h] [rbp-380h]
  __int64 v170; // [rsp+30h] [rbp-380h]
  __int64 v171; // [rsp+30h] [rbp-380h]
  unsigned __int64 v172; // [rsp+38h] [rbp-378h]
  __int64 *v174; // [rsp+48h] [rbp-368h]
  unsigned __int64 *v175; // [rsp+58h] [rbp-358h]
  __int64 **v176; // [rsp+58h] [rbp-358h]
  __int64 *v177; // [rsp+58h] [rbp-358h]
  int v178; // [rsp+60h] [rbp-350h]
  __int64 v179; // [rsp+60h] [rbp-350h]
  __int64 *v180; // [rsp+60h] [rbp-350h]
  __int64 v181; // [rsp+60h] [rbp-350h]
  __int64 **v182; // [rsp+60h] [rbp-350h]
  __int64 v183; // [rsp+68h] [rbp-348h]
  unsigned int v184; // [rsp+70h] [rbp-340h]
  __int64 *v185; // [rsp+70h] [rbp-340h]
  __int64 v186; // [rsp+78h] [rbp-338h]
  __int64 **v187; // [rsp+78h] [rbp-338h]
  unsigned int v188; // [rsp+88h] [rbp-328h] BYREF
  unsigned int v189; // [rsp+8Ch] [rbp-324h] BYREF
  void *src; // [rsp+90h] [rbp-320h] BYREF
  _BYTE *v191; // [rsp+98h] [rbp-318h]
  _BYTE *v192; // [rsp+A0h] [rbp-310h]
  __int64 **v193; // [rsp+B0h] [rbp-300h] BYREF
  __int64 **v194; // [rsp+B8h] [rbp-2F8h]
  __int64 **v195; // [rsp+C0h] [rbp-2F0h]
  _QWORD **v196; // [rsp+D0h] [rbp-2E0h] BYREF
  _QWORD **v197; // [rsp+D8h] [rbp-2D8h]
  __int64 v198; // [rsp+E0h] [rbp-2D0h]
  __int64 v199; // [rsp+F0h] [rbp-2C0h] BYREF
  void *s; // [rsp+F8h] [rbp-2B8h]
  __int64 v201; // [rsp+100h] [rbp-2B0h]
  __int64 v202; // [rsp+108h] [rbp-2A8h]
  __int64 *v203; // [rsp+110h] [rbp-2A0h] BYREF
  __int64 v204; // [rsp+118h] [rbp-298h]
  _QWORD v205[2]; // [rsp+120h] [rbp-290h] BYREF
  __int64 *v206; // [rsp+130h] [rbp-280h] BYREF
  __int64 v207; // [rsp+138h] [rbp-278h]
  __int64 v208; // [rsp+140h] [rbp-270h] BYREF
  __int64 v209; // [rsp+148h] [rbp-268h]
  _QWORD *v210; // [rsp+150h] [rbp-260h]
  _QWORD v211[4]; // [rsp+160h] [rbp-250h] BYREF
  __int64 v212[2]; // [rsp+180h] [rbp-230h] BYREF
  _QWORD v213[2]; // [rsp+190h] [rbp-220h] BYREF
  __int64 v214[2]; // [rsp+1A0h] [rbp-210h] BYREF
  _QWORD v215[2]; // [rsp+1B0h] [rbp-200h] BYREF
  __int64 v216; // [rsp+1C0h] [rbp-1F0h]
  __int64 v217; // [rsp+1C8h] [rbp-1E8h]
  unsigned int **v218; // [rsp+1D0h] [rbp-1E0h] BYREF
  __int64 v219; // [rsp+1D8h] [rbp-1D8h]
  unsigned __int64 *v220[8]; // [rsp+1E0h] [rbp-1D0h] BYREF
  unsigned __int64 *v221; // [rsp+220h] [rbp-190h]
  unsigned int v222; // [rsp+228h] [rbp-188h]
  _BYTE v223[384]; // [rsp+230h] [rbp-180h] BYREF

  v2 = a1;
  *(_BYTE *)(a1 + 408) = LOBYTE(qword_4FDBC48[8]) ^ 1;
  *(_BYTE *)(a1 + 409) = LOBYTE(qword_4FDBB68[8]) ^ 1;
  *(_BYTE *)(a1 + 410) = unk_4FDB9E8 ^ 1;
  v3 = (*(__int64 (__fastcall **)(_QWORD))(a1 + 88))(*(_QWORD *)(a1 + 96));
  v188 = 0;
  v4 = *(__int64 **)(v3 + 312);
  v186 = v3;
  v175 = *(unsigned __int64 **)(v3 + 320);
  v5 = (char *)v175 - (char *)v4;
  if ( (unsigned __int64)((char *)v175 - (char *)v4) > 0x18 )
  {
    sub_25FE7A0((__int64 *)&v218, v4, 0xAAAAAAAAAAAAAAABLL * (v5 >> 3));
    if ( v220[0] )
      sub_260CFB0((unsigned __int64 *)v4, v175, v220[0], v219);
    else
      sub_25FA8B0((unsigned __int64 *)v4, v175);
    v123 = v220[0];
    v124 = &v220[0][3 * v219];
    if ( v220[0] != v124 )
    {
      do
      {
        v125 = v123[1];
        v126 = *v123;
        if ( v125 != *v123 )
        {
          do
          {
            v127 = *(unsigned int *)(v126 + 144);
            v128 = *(_QWORD *)(v126 + 128);
            v126 += 152LL;
            sub_C7D6A0(v128, 8 * v127, 4);
            sub_C7D6A0(*(_QWORD *)(v126 - 56), 8LL * *(unsigned int *)(v126 - 40), 4);
            sub_C7D6A0(*(_QWORD *)(v126 - 88), 16LL * *(unsigned int *)(v126 - 72), 8);
            sub_C7D6A0(*(_QWORD *)(v126 - 120), 16LL * *(unsigned int *)(v126 - 104), 8);
          }
          while ( v125 != v126 );
          v126 = *v123;
        }
        if ( v126 )
          j_j___libc_free_0(v126);
        v123 += 3;
      }
      while ( v124 != v123 );
      v124 = v220[0];
    }
    j_j___libc_free_0((unsigned __int64)v124);
    v4 = *(__int64 **)(v186 + 312);
    v175 = *(unsigned __int64 **)(v186 + 320);
    v6 = 0xAAAAAAAAAAAAAAABLL * (((char *)v175 - (char *)v4) >> 3);
    if ( (unsigned __int64)((char *)v175 - (char *)v4) > 0x999999999999990LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  }
  else
  {
    v6 = 0xAAAAAAAAAAAAAAABLL * (v5 >> 3);
  }
  v7 = 320 * v6;
  if ( v6 )
  {
    v8 = sub_22077B0(320 * v6);
    v9 = (_DWORD *)(v8 + v7);
    v172 = v8;
    v10 = (_DWORD *)v8;
    v167 = v9;
    do
    {
      if ( v10 )
      {
        memset(v10, 0, 0x140u);
        v10[53] = -3;
      }
      v10 += 80;
    }
    while ( v10 != v9 );
    v4 = *(__int64 **)(v186 + 312);
    v175 = *(unsigned __int64 **)(v186 + 320);
  }
  else
  {
    v172 = 0;
    v167 = 0;
  }
  v11 = (__int64 **)v4;
  v199 = 0;
  v12 = v2;
  s = 0;
  v201 = 0;
  v202 = 0;
  src = 0;
  v191 = 0;
  v192 = 0;
  v193 = 0;
  v194 = 0;
  v195 = 0;
  v168 = v2 + 120;
  v184 = 0;
  if ( v175 != (unsigned __int64 *)v4 )
  {
    while ( 1 )
    {
      v14 = 5LL * v184++;
      v15 = v172 + (v14 << 6);
      sub_2608380(v12, v11, v15);
      if ( *(_QWORD *)(v15 + 8) - *(_QWORD *)v15 <= 8u )
        goto LABEL_17;
      ++v199;
      if ( (_DWORD)v201 )
      {
        v16 = 4 * v201;
        v13 = v202;
        if ( (unsigned int)(4 * v201) < 0x40 )
          v16 = 64;
        if ( (unsigned int)v202 <= v16 )
          goto LABEL_13;
        v17 = (char *)s;
        if ( (_DWORD)v201 == 1 )
        {
          v19 = 64;
        }
        else
        {
          _BitScanReverse(&v18, v201 - 1);
          v19 = 1 << (33 - (v18 ^ 0x1F));
          if ( v19 < 64 )
            v19 = 64;
          if ( (_DWORD)v202 == v19 )
          {
            v201 = 0;
            v160 = (char *)s + 4 * (unsigned int)v202;
            do
            {
              if ( v17 )
                *(_DWORD *)v17 = -1;
              v17 += 4;
            }
            while ( v160 != v17 );
            goto LABEL_16;
          }
        }
        sub_C7D6A0((__int64)s, 4LL * (unsigned int)v202, 4);
        v20 = sub_25F87F0(v19);
        LODWORD(v202) = v20;
        if ( v20 )
        {
          v21 = (_DWORD *)sub_C7D670(4LL * v20, 4);
          v201 = 0;
          s = v21;
          for ( i = &v21[(unsigned int)v202]; i != v21; ++v21 )
          {
            if ( v21 )
              *v21 = -1;
          }
          goto LABEL_16;
        }
      }
      else
      {
        if ( !HIDWORD(v201) )
          goto LABEL_16;
        v13 = v202;
        if ( (unsigned int)v202 <= 0x40 )
        {
LABEL_13:
          if ( 4LL * v13 )
            memset(s, 255, 4LL * v13);
          goto LABEL_15;
        }
        sub_C7D6A0((__int64)s, 4LL * (unsigned int)v202, 4);
        LODWORD(v202) = 0;
      }
      s = 0;
LABEL_15:
      v201 = 0;
LABEL_16:
      sub_2602050((__int64 ***)v15, (__int64)&v199);
      if ( *(_BYTE *)(v15 + 64) )
        goto LABEL_17;
      v23 = v193;
      if ( v194 != v193 )
        v194 = v193;
      v24 = *(__int64 ***)v15;
      v187 = *(__int64 ***)(v15 + 8);
      if ( *(__int64 ***)v15 != v187 )
      {
        do
        {
          v203 = *v24;
          sub_2600E40(v203);
          if ( *((_BYTE *)v203 + 256) )
          {
            v206 = 0;
            v219 = 0x600000000LL;
            v218 = (unsigned int **)v220;
            v207 = 0;
            v208 = 0;
            v209 = 0;
            sub_26015A0(*(_QWORD *)(*v203 + 8), *(_QWORD *)(*v203 + 16), (__int64)&v206, (__int64)&v218);
            v25 = v219;
            v178 = (int)v218;
            v212[0] = (__int64)v213;
            sub_25F5F00(v212, "outlined", (__int64)"");
            v26 = *(_QWORD *)(v12 + 120);
            *(_QWORD *)(v12 + 200) += 256LL;
            v27 = v178;
            v28 = (v26 + 7) & 0xFFFFFFFFFFFFFFF8LL;
            if ( *(_QWORD *)(v12 + 128) >= v28 + 256 && v26 )
            {
              *(_QWORD *)(v12 + 120) = v28 + 256;
            }
            else
            {
              v28 = sub_9D1E70(v168, 256, 256, 3);
              v27 = v178;
            }
            v183 = v28;
            sub_29AFB10(v28, v27, v25, 0, 0, 0, 0, 0, 0, 0, 0, (__int64)v212, 0);
            v29 = v203;
            v203[29] = v183;
            if ( (_QWORD *)v212[0] != v213 )
            {
              j_j___libc_free_0(v212[0]);
              v29 = v203;
            }
            sub_2609820(v12, (__int64)a2, v29, (__int64)&v199);
            v30 = v203;
            if ( !*((_BYTE *)v203 + 257) )
            {
              v31 = v194;
              if ( v194 == v195 )
              {
                sub_25FD6B0((__int64)&v193, v194, &v203);
                v30 = v203;
              }
              else
              {
                if ( v194 )
                {
                  *v194 = v203;
                  v31 = v194;
                }
                v194 = v31 + 1;
              }
            }
            sub_2600BA0(v30);
            sub_C7D6A0(v207, 8LL * (unsigned int)v209, 8);
            if ( v218 != (unsigned int **)v220 )
              _libc_free((unsigned __int64)v218);
          }
          ++v24;
        }
        while ( v187 != v24 );
        v187 = *(__int64 ***)v15;
        v23 = v193;
      }
      v32 = v194;
      v33 = v195;
      *(_QWORD *)v15 = v23;
      v193 = 0;
      *(_QWORD *)(v15 + 8) = v32;
      *(_QWORD *)(v15 + 16) = v33;
      v194 = 0;
      v195 = 0;
      if ( v187 )
      {
        j_j___libc_free_0((unsigned __int64)v187);
        v32 = *(__int64 ***)(v15 + 8);
        v23 = *(__int64 ***)v15;
      }
      if ( v23 == v32 )
        goto LABEL_17;
      sub_2608BD0((__int64 **)v15, (_QWORD **)a2);
      if ( *(_BYTE *)(v12 + 1) )
        sub_2600970(v12, (_QWORD **)a2, v15);
      v34 = *(_DWORD *)(v15 + 288);
      v36 = __OFSUB__(*(_DWORD *)(v15 + 304), v34);
      v35 = *(_DWORD *)(v15 + 304) - v34 < 0;
      if ( *(_DWORD *)(v15 + 304) == v34 )
      {
        v37 = *(_QWORD *)(v15 + 280);
        v36 = __OFSUB__(*(_QWORD *)(v15 + 296), v37);
        v35 = *(_QWORD *)(v15 + 296) - v37 < 0;
      }
      if ( v35 == v36 && *(_BYTE *)(v12 + 1) )
      {
        v129 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD))(v12 + 104))(
                            *(_QWORD *)(v12 + 112),
                            *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(***(_QWORD ***)v15 + 8LL) + 16LL) + 40LL)
                                      + 72LL));
        v130 = *v129;
        v174 = v129;
        v131 = sub_B2BE50(*v129);
        if ( sub_B6EA50(v131)
          || (v161 = sub_B2BE50(v130),
              v162 = sub_B6F970(v161),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v162 + 48LL))(v162)) )
        {
          sub_B176B0(
            (__int64)&v218,
            (__int64)"iroutliner",
            (__int64)"WouldNotDecreaseSize",
            20,
            *(_QWORD *)(*(_QWORD *)(***(_QWORD ***)v15 + 8LL) + 16LL));
          sub_B18290((__int64)&v218, "did not outline ", 0x10u);
          v132 = *(_QWORD *)(v15 + 8) - *(_QWORD *)v15;
          v133 = (__int64)v132 >> 3;
          if ( v132 <= 0x48 )
          {
            v203 = v205;
            sub_2240A50((__int64 *)&v203, 1u, 0);
            v138 = v203;
            goto LABEL_261;
          }
          if ( v132 <= 0x318 )
          {
            v203 = v205;
            sub_2240A50((__int64 *)&v203, 2u, 0);
            v138 = v203;
          }
          else
          {
            if ( v132 <= 0x1F38 )
            {
              v135 = 3;
            }
            else if ( v132 <= 0x13878 )
            {
              v135 = 4;
            }
            else
            {
              v134 = (__int64)(*(_QWORD *)(v15 + 8) - *(_QWORD *)v15) >> 3;
              LODWORD(v135) = 1;
              while ( 1 )
              {
                v136 = v134;
                v137 = v135;
                v135 = (unsigned int)(v135 + 4);
                v134 /= 0x2710u;
                if ( v136 <= 0x1869F )
                  break;
                if ( v136 <= 0xF423F )
                {
                  v135 = (unsigned int)(v137 + 5);
                  v203 = v205;
                  goto LABEL_258;
                }
                if ( v136 <= (unsigned __int64)&loc_98967F )
                {
                  v135 = (unsigned int)(v137 + 6);
                  break;
                }
                if ( v136 <= 0x5F5E0FF )
                {
                  v135 = (unsigned int)(v137 + 7);
                  break;
                }
              }
            }
            v203 = v205;
LABEL_258:
            sub_2240A50((__int64 *)&v203, v135, 0);
            v138 = v203;
            v139 = v204 - 1;
            do
            {
              v140 = v133
                   - 20
                   * ((((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v133 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL)
                    + v133 / 0x64);
              v141 = v133;
              v133 /= 0x64u;
              v142 = a00010203040506_0[2 * v140 + 1];
              LOBYTE(v140) = a00010203040506_0[2 * v140];
              *((_BYTE *)v138 + v139) = v142;
              v143 = v139 - 1;
              v139 -= 2;
              *((_BYTE *)v138 + v143) = v140;
            }
            while ( v141 > 0x270F );
            if ( v141 <= 0x3E7 )
            {
LABEL_261:
              *(_BYTE *)v138 = v133 + 48;
LABEL_262:
              v144 = v203;
              v170 = v204;
              v212[0] = (__int64)v213;
              sub_25F5F00(v212, "String", (__int64)"");
              v214[0] = (__int64)v215;
              sub_25F5F00(v214, v144, (__int64)v144 + v170);
              v216 = 0;
              v217 = 0;
              v171 = sub_2445430((__int64)&v218, (__int64)v212);
              sub_B18290(v171, " regions due to estimated increase of ", 0x26u);
              v145 = *(_QWORD *)(v15 + 280);
              v146 = 1;
              v147 = *(_QWORD *)(v15 + 296);
              if ( *(_DWORD *)(v15 + 288) != 1 )
                v146 = *(_DWORD *)(v15 + 304);
              v36 = __OFSUB__(v147, v145);
              v148 = v147 - v145;
              if ( v36 )
              {
                v148 = 0x8000000000000000LL;
                if ( v145 <= 0 )
                  v148 = 0x7FFFFFFFFFFFFFFFLL;
              }
              v165 = v146 | v165 & 0xFFFFFFFF00000000LL;
              sub_B16D50((__int64)&v206, "InstructionIncrease", 19, v148, v165);
              v149 = sub_2445430(v171, (__int64)&v206);
              sub_B18290(v149, " instructions at locations ", 0x1Bu);
              if ( v210 != v211 )
                j_j___libc_free_0((unsigned __int64)v210);
              if ( v206 != &v208 )
                j_j___libc_free_0((unsigned __int64)v206);
              if ( (_QWORD *)v214[0] != v215 )
                j_j___libc_free_0(v214[0]);
              if ( (_QWORD *)v212[0] != v213 )
                j_j___libc_free_0(v212[0]);
              if ( v203 != v205 )
                j_j___libc_free_0((unsigned __int64)v203);
              v150 = *(__int64 ***)(v15 + 8);
              v151 = *(__int64 ***)v15;
              v182 = v150;
              if ( v150 != v151 )
              {
                v152 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(**v151 + 8) + 16LL) + 48LL);
                v206 = v152;
                if ( v152 )
                  sub_B96E90((__int64)&v206, (__int64)v152, 1);
                sub_B16E20((__int64)v212, "DebugLoc", 8, &v206);
                sub_2445430((__int64)&v218, (__int64)v212);
                if ( (_QWORD *)v214[0] != v215 )
                  j_j___libc_free_0(v214[0]);
                if ( (_QWORD *)v212[0] != v213 )
                  j_j___libc_free_0(v212[0]);
                if ( v206 )
                  sub_B91220((__int64)&v206, (__int64)v206);
                for ( j = v151 + 1; v182 != j; ++j )
                {
                  sub_B18290((__int64)&v218, " ", 1u);
                  v154 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(**j + 8) + 16LL) + 48LL);
                  v206 = v154;
                  if ( v154 )
                    sub_B96E90((__int64)&v206, (__int64)v154, 1);
                  sub_B16E20((__int64)v212, "DebugLoc", 8, &v206);
                  sub_2445430((__int64)&v218, (__int64)v212);
                  if ( (_QWORD *)v214[0] != v215 )
                    j_j___libc_free_0(v214[0]);
                  if ( (_QWORD *)v212[0] != v213 )
                    j_j___libc_free_0(v212[0]);
                  if ( v206 )
                    sub_B91220((__int64)&v206, (__int64)v206);
                }
              }
              sub_1049740(v174, (__int64)&v218);
              v155 = v221;
              v218 = (unsigned int **)&unk_49D9D40;
              v156 = &v221[10 * v222];
              if ( v221 != v156 )
              {
                do
                {
                  v156 -= 10;
                  v157 = v156[4];
                  if ( (unsigned __int64 *)v157 != v156 + 6 )
                    j_j___libc_free_0(v157);
                  if ( (unsigned __int64 *)*v156 != v156 + 2 )
                    j_j___libc_free_0(*v156);
                }
                while ( v155 != v156 );
                v156 = v221;
              }
              if ( v156 != (unsigned __int64 *)v223 )
                _libc_free((unsigned __int64)v156);
              goto LABEL_17;
            }
          }
          *((_BYTE *)v138 + 1) = a00010203040506_0[2 * v133 + 1];
          *(_BYTE *)v138 = a00010203040506_0[2 * v133];
          goto LABEL_262;
        }
LABEL_17:
        v11 += 3;
        if ( v175 == (unsigned __int64 *)v11 )
          goto LABEL_65;
      }
      else
      {
        v218 = (unsigned int **)v15;
        v38 = v191;
        if ( v191 == v192 )
        {
          sub_25FE5D0((__int64)&src, v191, &v218);
          goto LABEL_17;
        }
        if ( v191 )
        {
          *(_QWORD *)v191 = v15;
          v38 = v191;
        }
        v11 += 3;
        v191 = v38 + 8;
        if ( v175 == (unsigned __int64 *)v11 )
        {
LABEL_65:
          v2 = v12;
          break;
        }
      }
    }
  }
  sub_25FCE60(v168);
  v39 = (char *)src;
  v169 = v191;
  v40 = v191 - (_BYTE *)src;
  if ( (unsigned __int64)(v191 - (_BYTE *)src) > 8 )
  {
    v120 = v40 >> 3;
    if ( v40 <= 0 )
    {
LABEL_320:
      v122 = 0;
      sub_25F8FC0(v39, v169);
    }
    else
    {
      while ( 1 )
      {
        v121 = (char *)sub_2207800(8 * v120);
        v122 = (unsigned __int64)v121;
        if ( v121 )
          break;
        v120 >>= 1;
        if ( !v120 )
          goto LABEL_320;
      }
      sub_2607EC0(v39, v169, v121, v120);
    }
    j_j___libc_free_0(v122);
    v39 = (char *)src;
    v169 = v191;
  }
  v196 = 0;
  v197 = 0;
  v198 = 0;
  v185 = (__int64 *)v39;
  if ( v169 == v39 )
  {
    v53 = v188;
    goto LABEL_92;
  }
  do
  {
    v41 = *v185;
    v42 = v193;
    if ( v193 != v194 )
      v194 = v193;
    v43 = *(__int64 ***)v41;
    v44 = *(__int64 ***)(v41 + 8);
    if ( v44 != *(__int64 ***)v41 )
    {
      do
      {
        v218 = (unsigned int **)*v43;
        if ( sub_25FFC80(v2, v218) )
        {
          v45 = v194;
          if ( v194 == v195 )
          {
            sub_25FD6B0((__int64)&v193, v194, &v218);
          }
          else
          {
            if ( v194 )
            {
              *v194 = (__int64 *)v218;
              v45 = v194;
            }
            v194 = v45 + 1;
          }
        }
        ++v43;
      }
      while ( v44 != v43 );
      v42 = v193;
    }
    if ( (unsigned __int64)((char *)v194 - (char *)v42) > 8 )
    {
      v46 = *(_QWORD *)v41;
      *(_QWORD *)v41 = v42;
      v193 = 0;
      *(_QWORD *)(v41 + 8) = v194;
      v194 = 0;
      *(_QWORD *)(v41 + 16) = v195;
      v195 = 0;
      if ( v46 )
        j_j___libc_free_0(v46);
      if ( !*(_BYTE *)(v2 + 1) )
        goto LABEL_113;
      *(_QWORD *)(v41 + 280) = 0;
      *(_DWORD *)(v41 + 288) = 0;
      *(_QWORD *)(v41 + 296) = 0;
      *(_DWORD *)(v41 + 304) = 0;
      sub_2600970(v2, (_QWORD **)a2, v41);
      v47 = *(_DWORD *)(v41 + 288);
      v48 = *(_DWORD *)(v41 + 304) < v47;
      if ( *(_DWORD *)(v41 + 304) == v47 )
        v48 = *(_QWORD *)(v41 + 296) < *(_QWORD *)(v41 + 280);
      if ( v48 )
      {
LABEL_113:
        v61 = v193;
        if ( v193 != v194 )
          v194 = v193;
        v62 = *(__int64 ***)v41;
        v63 = *(__int64 ***)(v41 + 8);
        if ( v63 != *(__int64 ***)v41 )
        {
          do
          {
            v218 = (unsigned int **)*v62;
            sub_2600E40((__int64 *)v218);
            if ( *((_BYTE *)v218 + 256) )
            {
              v64 = v194;
              if ( v194 == v195 )
              {
                sub_25FD6B0((__int64)&v193, v194, &v218);
              }
              else
              {
                if ( v194 )
                {
                  *v194 = (__int64 *)v218;
                  v64 = v194;
                }
                v194 = v64 + 1;
              }
            }
            ++v62;
          }
          while ( v63 != v62 );
          v61 = v193;
        }
        v65 = *(_QWORD *)v41;
        *(_QWORD *)v41 = v61;
        v193 = 0;
        *(_QWORD *)(v41 + 8) = v194;
        v194 = 0;
        *(_QWORD *)(v41 + 16) = v195;
        v195 = 0;
        if ( v65 )
          j_j___libc_free_0(v65);
        v66 = *(__int64 ***)v41;
        v176 = *(__int64 ***)(v41 + 8);
        if ( (unsigned __int64)v176 - *(_QWORD *)v41 <= 8 )
        {
          while ( v176 != v66 )
          {
            v119 = *v66++;
            sub_2600BA0(v119);
          }
          goto LABEL_85;
        }
        v67 = v193;
        if ( v193 != v194 )
        {
          v194 = v193;
          v66 = *(__int64 ***)v41;
          v176 = *(__int64 ***)(v41 + 8);
        }
        if ( v66 == v176 )
        {
LABEL_165:
          v87 = *(_QWORD *)v41;
          *(_QWORD *)v41 = v67;
          v193 = 0;
          *(_QWORD *)(v41 + 8) = v194;
          v194 = 0;
          *(_QWORD *)(v41 + 16) = v195;
          v195 = 0;
          if ( v87 )
            j_j___libc_free_0(v87);
          if ( *(_QWORD *)(v41 + 8) == *(_QWORD *)v41 )
            goto LABEL_85;
          v88 = sub_B43CB0(*(_QWORD *)(**(_QWORD **)v41 + 240LL));
          v89 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(v2 + 104))(*(_QWORD *)(v2 + 112), v88);
          v90 = *v89;
          v177 = v89;
          v91 = sub_B2BE50(*v89);
          if ( !sub_B6EA50(v91) )
          {
            v158 = sub_B2BE50(v90);
            v159 = sub_B6F970(v158);
            if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v159 + 48LL))(v159) )
            {
LABEL_224:
              sub_2606BC0(v2, a2, (__int64 *)v41, (__int64)&v196, &v188);
              goto LABEL_85;
            }
          }
          sub_B174A0(
            (__int64)&v218,
            (__int64)"iroutliner",
            (__int64)"Outlined",
            8,
            *(_QWORD *)(*(_QWORD *)(***(_QWORD ***)v41 + 8LL) + 16LL));
          sub_B18290((__int64)&v218, "outlined ", 9u);
          v92 = *(_QWORD *)(v41 + 8) - *(_QWORD *)v41;
          v93 = (__int64)v92 >> 3;
          if ( v92 > 0x48 )
          {
            if ( v92 <= 0x318 )
            {
              v203 = v205;
              sub_2240A50((__int64 *)&v203, 2u, 0);
              v98 = v203;
            }
            else
            {
              if ( v92 <= 0x1F38 )
              {
                v95 = 3;
              }
              else if ( v92 <= 0x13878 )
              {
                v95 = 4;
              }
              else
              {
                v94 = (__int64)(*(_QWORD *)(v41 + 8) - *(_QWORD *)v41) >> 3;
                LODWORD(v95) = 1;
                while ( 1 )
                {
                  v96 = v94;
                  v97 = v95;
                  v95 = (unsigned int)(v95 + 4);
                  v94 /= 0x2710u;
                  if ( v96 <= 0x1869F )
                    break;
                  if ( v96 <= 0xF423F )
                  {
                    v203 = v205;
                    v95 = (unsigned int)(v97 + 5);
                    goto LABEL_179;
                  }
                  if ( v96 <= (unsigned __int64)&loc_98967F )
                  {
                    v95 = (unsigned int)(v97 + 6);
                    break;
                  }
                  if ( v96 <= 0x5F5E0FF )
                  {
                    v95 = (unsigned int)(v97 + 7);
                    break;
                  }
                }
              }
              v203 = v205;
LABEL_179:
              sub_2240A50((__int64 *)&v203, v95, 0);
              v98 = v203;
              v99 = v204 - 1;
              do
              {
                v100 = v93;
                v101 = 5
                     * (v93 / 0x64
                      + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v93 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                v102 = v93;
                v93 /= 0x64u;
                v103 = v100 - 4 * v101;
                *((_BYTE *)v98 + v99) = a00010203040506_0[2 * v103 + 1];
                v104 = v99 - 1;
                v99 -= 2;
                *((_BYTE *)v98 + v104) = a00010203040506_0[2 * v103];
              }
              while ( v102 > 0x270F );
              if ( v102 <= 0x3E7 )
                goto LABEL_182;
            }
            *((_BYTE *)v98 + 1) = a00010203040506_0[2 * v93 + 1];
            *(_BYTE *)v98 = a00010203040506_0[2 * v93];
            goto LABEL_183;
          }
          v203 = v205;
          sub_2240A50((__int64 *)&v203, 1u, 0);
          v98 = v203;
LABEL_182:
          *(_BYTE *)v98 = v93 + 48;
LABEL_183:
          v105 = v204;
          v180 = v203;
          v212[0] = (__int64)v213;
          sub_25F5F00(v212, "String", (__int64)"");
          v214[0] = (__int64)v215;
          sub_25F5F00(v214, v180, (__int64)v180 + v105);
          v216 = 0;
          v217 = 0;
          v106 = sub_23FD640((__int64)&v218, (__int64)v212);
          sub_B18290(v106, " regions with decrease of ", 0x1Au);
          v107 = *(_QWORD *)(v41 + 296);
          v108 = 1;
          v109 = *(_QWORD *)(v41 + 280);
          if ( *(_DWORD *)(v41 + 304) != 1 )
            v108 = *(_DWORD *)(v41 + 288);
          v36 = __OFSUB__(v109, v107);
          v110 = v109 - v107;
          if ( v36 )
          {
            v110 = 0x8000000000000000LL;
            if ( v107 <= 0 )
              v110 = 0x7FFFFFFFFFFFFFFFLL;
          }
          v163 = v108 | v163 & 0xFFFFFFFF00000000LL;
          sub_B16D50((__int64)&v206, "Benefit", 7, v110, v163);
          v111 = sub_23FD640(v106, (__int64)&v206);
          sub_B18290(v111, " instructions at locations ", 0x1Bu);
          if ( v210 != v211 )
            j_j___libc_free_0((unsigned __int64)v210);
          if ( v206 != &v208 )
            j_j___libc_free_0((unsigned __int64)v206);
          if ( (_QWORD *)v214[0] != v215 )
            j_j___libc_free_0(v214[0]);
          if ( (_QWORD *)v212[0] != v213 )
            j_j___libc_free_0(v212[0]);
          if ( v203 != v205 )
            j_j___libc_free_0((unsigned __int64)v203);
          v112 = *(_QWORD *)v41;
          v181 = *(_QWORD *)(v41 + 8);
          if ( v181 != *(_QWORD *)v41 )
          {
            v113 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(**(_QWORD **)v112 + 8LL) + 16LL) + 48LL);
            v206 = v113;
            if ( v113 )
              sub_B96E90((__int64)&v206, (__int64)v113, 1);
            sub_B16E20((__int64)v212, "DebugLoc", 8, &v206);
            sub_23FD640((__int64)&v218, (__int64)v212);
            if ( (_QWORD *)v214[0] != v215 )
              j_j___libc_free_0(v214[0]);
            if ( (_QWORD *)v212[0] != v213 )
              j_j___libc_free_0(v212[0]);
            if ( v206 )
              sub_B91220((__int64)&v206, (__int64)v206);
            for ( k = v112 + 8; v181 != k; k += 8 )
            {
              sub_B18290((__int64)&v218, " ", 1u);
              v115 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(**(_QWORD **)k + 8LL) + 16LL) + 48LL);
              v206 = v115;
              if ( v115 )
                sub_B96E90((__int64)&v206, (__int64)v115, 1);
              sub_B16E20((__int64)v212, "DebugLoc", 8, &v206);
              sub_23FD640((__int64)&v218, (__int64)v212);
              if ( (_QWORD *)v214[0] != v215 )
                j_j___libc_free_0(v214[0]);
              if ( (_QWORD *)v212[0] != v213 )
                j_j___libc_free_0(v212[0]);
              if ( v206 )
                sub_B91220((__int64)&v206, (__int64)v206);
            }
          }
          sub_1049740(v177, (__int64)&v218);
          v116 = v221;
          v218 = (unsigned int **)&unk_49D9D40;
          v117 = &v221[10 * v222];
          if ( v221 != v117 )
          {
            do
            {
              v117 -= 10;
              v118 = v117[4];
              if ( (unsigned __int64 *)v118 != v117 + 6 )
                j_j___libc_free_0(v118);
              if ( (unsigned __int64 *)*v117 != v117 + 2 )
                j_j___libc_free_0(*v117);
            }
            while ( v116 != v117 );
            v117 = v221;
          }
          if ( v117 != (unsigned __int64 *)v223 )
            _libc_free((unsigned __int64)v117);
          goto LABEL_224;
        }
        v164 = v41;
        v166 = v2 + 8;
        while ( 1 )
        {
          v68 = *v66;
          v206 = 0;
          v207 = 0;
          v218 = (unsigned int **)v220;
          v219 = 0x600000000LL;
          v203 = v68;
          v208 = 0;
          v209 = 0;
          sub_26015A0(*(_QWORD *)(*v68 + 8), *(_QWORD *)(*v68 + 16), (__int64)&v206, (__int64)&v218);
          v69 = (int)v218;
          v70 = v219;
          v212[0] = (__int64)v213;
          sub_25F5F00(v212, "outlined", (__int64)"");
          v71 = *(_QWORD *)(v2 + 120);
          *(_QWORD *)(v2 + 200) += 256LL;
          v72 = (v71 + 7) & 0xFFFFFFFFFFFFFFF8LL;
          if ( *(_QWORD *)(v2 + 128) >= v72 + 256 && v71 )
            *(_QWORD *)(v2 + 120) = v72 + 256;
          else
            v72 = sub_9D1E70(v168, 256, 256, 3);
          v179 = v72;
          sub_29AFB10(v72, v69, v70, 0, 0, 0, 0, 0, 0, 0, 0, (__int64)v212, 0);
          v73 = v203;
          v203[29] = v179;
          if ( (_QWORD *)v212[0] != v213 )
          {
            j_j___libc_free_0(v212[0]);
            v73 = v203;
          }
          if ( !(unsigned __int8)sub_2604F70((_QWORD *)v2, v73) )
            goto LABEL_130;
          v74 = *(_DWORD *)*v203;
          v75 = v74 + *(_DWORD *)(*v203 + 4);
          v189 = v74;
          v76 = v75 - 1;
          if ( v76 >= v74 )
            break;
LABEL_151:
          v84 = v194;
          if ( v194 == v195 )
          {
            sub_25FD6B0((__int64)&v193, v194, &v203);
          }
          else
          {
            if ( v194 )
            {
              *v194 = v203;
              v84 = v194;
            }
            v194 = v84 + 1;
          }
LABEL_130:
          sub_C7D6A0(v207, 8LL * (unsigned int)v209, 8);
          if ( v218 != (unsigned int **)v220 )
            _libc_free((unsigned __int64)v218);
          if ( v176 == ++v66 )
          {
            v41 = v164;
            v67 = v193;
            goto LABEL_165;
          }
        }
        while ( 1 )
        {
          v81 = *(_DWORD *)(v2 + 32);
          if ( !v81 )
            break;
          v77 = *(_QWORD *)(v2 + 16);
          v78 = (v81 - 1) & (37 * v74);
          v79 = (unsigned int *)(v77 + 4LL * v78);
          v80 = *v79;
          if ( *v79 != v74 )
          {
            v85 = 1;
            v82 = 0;
            while ( v80 != -1 )
            {
              if ( v80 != -2 || v82 )
                v79 = v82;
              v78 = (v81 - 1) & (v85 + v78);
              v80 = *(_DWORD *)(v77 + 4LL * v78);
              if ( v80 == v74 )
                goto LABEL_142;
              ++v85;
              v82 = v79;
              v79 = (unsigned int *)(v77 + 4LL * v78);
            }
            v86 = *(_DWORD *)(v2 + 24);
            if ( !v82 )
              v82 = v79;
            ++*(_QWORD *)(v2 + 8);
            v83 = v86 + 1;
            v212[0] = (__int64)v82;
            if ( 4 * (v86 + 1) < 3 * v81 )
            {
              if ( v81 - *(_DWORD *)(v2 + 28) - v83 <= v81 >> 3 )
              {
LABEL_146:
                sub_A08C50(v166, v81);
                sub_22B31A0(v166, (int *)&v189, v212);
                v82 = (unsigned int *)v212[0];
                v83 = *(_DWORD *)(v2 + 24) + 1;
              }
              *(_DWORD *)(v2 + 24) = v83;
              if ( *v82 != -1 )
                --*(_DWORD *)(v2 + 28);
              *v82 = v189;
              goto LABEL_142;
            }
LABEL_145:
            v81 *= 2;
            goto LABEL_146;
          }
LABEL_142:
          v74 = v189 + 1;
          v189 = v74;
          if ( v74 > v76 )
            goto LABEL_151;
        }
        ++*(_QWORD *)(v2 + 8);
        v212[0] = 0;
        goto LABEL_145;
      }
    }
LABEL_85:
    ++v185;
  }
  while ( v169 != (char *)v185 );
  v49 = (unsigned __int64)v196;
  v50 = v197;
  if ( v197 != v196 )
  {
    v51 = v196;
    do
    {
      v52 = *v51++;
      sub_B2E860(v52);
    }
    while ( v50 != v51 );
    v49 = (unsigned __int64)v196;
  }
  v53 = v188;
  if ( v49 )
    j_j___libc_free_0(v49);
LABEL_92:
  if ( v193 )
    j_j___libc_free_0((unsigned __int64)v193);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  sub_C7D6A0((__int64)s, 4LL * (unsigned int)v202, 4);
  for ( m = v172; (_DWORD *)m != v167; m += 320LL )
  {
    sub_C7D6A0(*(_QWORD *)(m + 256), 16LL * *(unsigned int *)(m + 272), 8);
    v55 = *(unsigned int *)(m + 240);
    if ( (_DWORD)v55 )
    {
      v56 = *(_QWORD *)(m + 224);
      v57 = v56 + 40 * v55;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v56 <= 0xFFFFFFFD )
          {
            v58 = *(_QWORD *)(v56 + 16);
            if ( v58 != v56 + 32 )
              break;
          }
          v56 += 40;
          if ( v57 == v56 )
            goto LABEL_103;
        }
        _libc_free(v58);
        v56 += 40;
      }
      while ( v57 != v56 );
LABEL_103:
      v55 = *(unsigned int *)(m + 240);
    }
    sub_C7D6A0(*(_QWORD *)(m + 224), 40 * v55, 8);
    sub_C7D6A0(*(_QWORD *)(m + 184), 8LL * *(unsigned int *)(m + 200), 4);
    sub_C7D6A0(*(_QWORD *)(m + 144), 16LL * *(unsigned int *)(m + 160), 8);
    sub_C7D6A0(*(_QWORD *)(m + 112), 16LL * *(unsigned int *)(m + 128), 8);
    sub_C7D6A0(*(_QWORD *)(m + 80), 16LL * *(unsigned int *)(m + 96), 8);
    v59 = *(_QWORD *)(m + 24);
    if ( v59 )
      j_j___libc_free_0(v59);
    if ( *(_QWORD *)m )
      j_j___libc_free_0(*(_QWORD *)m);
  }
  if ( v172 )
    j_j___libc_free_0(v172);
  return v53;
}
