// Function: sub_3581C30
// Address: 0x3581c30
//
__int64 __fastcall sub_3581C30(_DWORD *a1, __int64 *a2)
{
  int v2; // eax
  int v3; // eax
  int v4; // edx
  int v5; // eax
  int v6; // eax
  __int64 v7; // r12
  __int16 v8; // cx
  unsigned __int8 v9; // al
  bool v10; // dl
  __int64 v11; // rdi
  unsigned __int64 v12; // rbx
  _QWORD *v13; // rdx
  _BYTE *v14; // rax
  unsigned __int8 v15; // dl
  unsigned __int8 v16; // dl
  const char *v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rbx
  unsigned int v20; // esi
  __int64 v21; // r9
  __int64 v22; // r14
  int v23; // r8d
  __int64 *v24; // r11
  unsigned int v25; // ecx
  __int64 *v26; // rax
  __int64 v27; // rdx
  bool v28; // zf
  __int64 v29; // rax
  int v30; // esi
  int v31; // edx
  int v32; // edx
  __m128i v33; // xmm2
  int v34; // edx
  int v35; // esi
  _QWORD *v36; // rsi
  __int64 v37; // rsi
  unsigned __int8 *v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // r12
  __int64 v44; // rax
  int v45; // esi
  __int64 v46; // rax
  int v47; // edx
  int v48; // edx
  __m128i v49; // xmm0
  int v50; // edx
  unsigned __int64 v51; // rdx
  int v52; // ebx
  int v53; // ebx
  __int64 v54; // r10
  unsigned int v55; // edx
  __int64 v56; // rsi
  int v57; // eax
  __int64 v58; // rax
  int v59; // esi
  int v60; // edx
  int v61; // edx
  __m128i v62; // xmm1
  int v63; // edx
  __int64 v64; // r13
  __int64 v65; // r12
  unsigned __int64 v66; // rbx
  unsigned __int8 v67; // al
  unsigned __int8 **v68; // rdx
  unsigned __int8 *v69; // rax
  unsigned __int64 v70; // r8
  char *v71; // rdi
  unsigned __int8 *v72; // rsi
  unsigned __int8 v73; // al
  __int64 v74; // rdi
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  unsigned __int8 v77; // al
  __int64 v78; // r12
  unsigned __int64 v79; // rdx
  unsigned int v80; // ebx
  unsigned int v81; // r13d
  unsigned __int64 v82; // rdx
  unsigned int v83; // ecx
  unsigned __int64 v84; // rsi
  unsigned int v85; // edi
  unsigned __int64 v86; // rsi
  _BYTE *v87; // rsi
  unsigned int v88; // edx
  int v89; // ecx
  unsigned int v90; // eax
  unsigned int v91; // r9d
  __int64 v92; // rax
  __int64 v93; // r8
  __int64 v94; // rdx
  size_t v95; // rbx
  int *v96; // r13
  int *v97; // rbx
  size_t v98; // rdx
  size_t v99; // r13
  unsigned __int8 v100; // al
  unsigned __int8 **v101; // rdx
  unsigned __int8 *v102; // rax
  unsigned __int8 v103; // dl
  __int64 v104; // rax
  __int64 v105; // rdi
  int *v106; // rbx
  size_t v107; // rdx
  size_t v108; // r13
  unsigned __int8 v109; // al
  __int64 v110; // rbx
  __int64 v111; // rdx
  __int64 v112; // rbx
  unsigned int v113; // r12d
  __int64 v114; // r13
  unsigned __int64 v115; // rdx
  unsigned __int64 v116; // rsi
  unsigned __int64 v117; // rcx
  int v118; // edi
  _BYTE *v119; // rsi
  __int64 v120; // rdx
  int v121; // ecx
  unsigned __int64 v122; // rdx
  int v123; // eax
  unsigned int v124; // r8d
  __int64 v125; // rax
  __int64 v126; // r9
  __int64 v127; // rdx
  size_t v128; // r12
  int *v129; // r13
  unsigned __int8 v130; // al
  __int64 v131; // r13
  unsigned __int8 **v132; // rdx
  unsigned __int8 *v133; // rax
  unsigned __int8 v134; // dl
  __int64 v135; // rax
  __int64 v136; // rdi
  size_t v137; // rdx
  unsigned __int8 v138; // al
  __int64 v139; // r12
  __int64 v140; // r13
  _QWORD *v141; // rax
  __int64 v142; // rsi
  unsigned __int8 *v143; // rsi
  int v144; // eax
  int v145; // eax
  int v146; // r8d
  __int64 v147; // r10
  __int64 *v148; // rcx
  int v149; // r9d
  unsigned int v150; // edx
  __int64 v151; // rsi
  int v153; // r9d
  __int64 *v155; // [rsp+10h] [rbp-1E0h]
  unsigned __int8 v156; // [rsp+1Fh] [rbp-1D1h]
  int v157; // [rsp+20h] [rbp-1D0h]
  unsigned int v158; // [rsp+24h] [rbp-1CCh]
  __int64 v159; // [rsp+28h] [rbp-1C8h]
  __int64 v161; // [rsp+38h] [rbp-1B8h]
  __int64 v162; // [rsp+40h] [rbp-1B0h]
  int *v163; // [rsp+40h] [rbp-1B0h]
  __int64 v164; // [rsp+48h] [rbp-1A8h]
  size_t v165; // [rsp+48h] [rbp-1A8h]
  __int64 *v166; // [rsp+50h] [rbp-1A0h]
  __int64 v167; // [rsp+58h] [rbp-198h]
  unsigned __int8 *v168; // [rsp+60h] [rbp-190h]
  int v169; // [rsp+60h] [rbp-190h]
  __int64 v170; // [rsp+68h] [rbp-188h]
  int v171; // [rsp+70h] [rbp-180h]
  int v172; // [rsp+74h] [rbp-17Ch]
  int v173; // [rsp+80h] [rbp-170h]
  __int64 v174; // [rsp+88h] [rbp-168h]
  _DWORD v175[4]; // [rsp+90h] [rbp-160h] BYREF
  __int64 v176; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v177; // [rsp+A8h] [rbp-148h]
  __int64 v178; // [rsp+B0h] [rbp-140h]
  unsigned int v179; // [rsp+B8h] [rbp-138h]
  __int64 v180; // [rsp+C0h] [rbp-130h] BYREF
  __int64 v181; // [rsp+C8h] [rbp-128h]
  __int64 v182; // [rsp+D0h] [rbp-120h]
  unsigned int v183; // [rsp+D8h] [rbp-118h]
  unsigned __int64 v184; // [rsp+E0h] [rbp-110h] BYREF
  int v185; // [rsp+E8h] [rbp-108h]
  int v186; // [rsp+ECh] [rbp-104h]
  __m128i v187; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v188; // [rsp+100h] [rbp-F0h] BYREF
  size_t v189; // [rsp+108h] [rbp-E8h]
  _QWORD v190[2]; // [rsp+110h] [rbp-E0h] BYREF
  __int64 v191[26]; // [rsp+120h] [rbp-D0h] BYREF

  v167 = sub_BA8DC0(*(_QWORD *)(*a2 + 40), (__int64)"llvm.pseudo_probe_desc", 22);
  if ( !v167 )
  {
    v156 = sub_B921D0(*a2);
    if ( !v156 )
      return v156;
  }
  v176 = 0;
  v177 = 0;
  v2 = a1[53];
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v3 = (LOBYTE(qword_503F1C8[8]) == 0) + v2 - 1;
  v181 = 0;
  v182 = 0;
  v183 = 0;
  if ( v3 == 31 )
    v4 = -1;
  else
    v4 = (1 << (v3 + 1)) - 1;
  v5 = a1[54];
  if ( v5 == 31 )
    v6 = -1;
  else
    v6 = (1 << (v5 + 1)) - 1;
  v157 = v6 ^ v4;
  v155 = a2 + 40;
  v161 = a2[41];
  if ( (__int64 *)v161 == a2 + 40 )
  {
    v156 = 0;
    v39 = 0;
    v40 = 0;
    goto LABEL_58;
  }
  v156 = 0;
  do
  {
    v7 = *(_QWORD *)(v161 + 56);
    v170 = v161 + 48;
    if ( v7 != v161 + 48 )
    {
      v158 = ((unsigned int)v161 >> 9) ^ ((unsigned int)v161 >> 4);
      while ( 1 )
      {
        if ( v167 )
        {
          if ( *(_WORD *)(v7 + 68) != 24 )
            goto LABEL_52;
        }
        else if ( LOBYTE(qword_503F1C8[8]) && (*(_BYTE *)(*(_QWORD *)(v7 + 16) + 24LL) & 0x10) != 0 )
        {
          goto LABEL_52;
        }
        v166 = (__int64 *)(v7 + 56);
        v174 = sub_B10CD0(v7 + 56);
        if ( !v174 )
        {
          if ( !v7 )
            BUG();
          goto LABEL_52;
        }
        v8 = *(_WORD *)(v7 + 68);
        v173 = v8 == 24 ? *(_DWORD *)(*(_QWORD *)(v7 + 32) + 64LL) : *(_DWORD *)(v174 + 4);
        if ( v173 )
          break;
LABEL_52:
        if ( (*(_BYTE *)v7 & 4) != 0 )
        {
          v7 = *(_QWORD *)(v7 + 8);
          if ( v170 == v7 )
            goto LABEL_54;
        }
        else
        {
          while ( (*(_BYTE *)(v7 + 44) & 8) != 0 )
            v7 = *(_QWORD *)(v7 + 8);
          v7 = *(_QWORD *)(v7 + 8);
          if ( v170 == v7 )
            goto LABEL_54;
        }
      }
      v164 = v174 - 16;
      v9 = *(_BYTE *)(v174 - 16);
      v10 = (v9 & 2) != 0;
      if ( (v9 & 2) != 0 )
        v11 = *(_QWORD *)(v174 - 32);
      else
        v11 = v164 - 8LL * ((v9 >> 2) & 0xF);
      v172 = 0;
      if ( **(_BYTE **)v11 == 20 )
        v172 = *(_DWORD *)(*(_QWORD *)v11 + 4LL);
      if ( a1[52] == 1 && v8 == 24 )
      {
        v141 = sub_26BDBC0(v174, 0);
        sub_B10CB0(v191, (__int64)v141);
        if ( v166 == v191 )
        {
          if ( v191[0] )
            sub_B91220((__int64)v166, v191[0]);
        }
        else
        {
          v142 = *(_QWORD *)(v7 + 56);
          if ( v142 )
            sub_B91220((__int64)v166, v142);
          v143 = (unsigned __int8 *)v191[0];
          *(_QWORD *)(v7 + 56) = v191[0];
          if ( v143 )
            sub_B976B0((__int64)v191, v143, (__int64)v166);
        }
        v172 = 0;
        v9 = *(_BYTE *)(v174 - 16);
        v10 = (v9 & 2) != 0;
      }
      v12 = 0;
      if ( !LOBYTE(qword_503F1C8[8]) )
      {
LABEL_24:
        if ( v10 )
          v13 = *(_QWORD **)(v174 - 32);
        else
          v13 = (_QWORD *)(v164 - 8LL * ((v9 >> 2) & 0xF));
LABEL_26:
        v14 = (_BYTE *)*v13;
        if ( *(_BYTE *)*v13 != 16 )
        {
          v15 = *(v14 - 16);
          if ( (v15 & 2) != 0 )
          {
            v14 = (_BYTE *)**((_QWORD **)v14 - 4);
            if ( v14 )
              goto LABEL_29;
          }
          else
          {
            v14 = *(_BYTE **)&v14[-8 * ((v15 >> 2) & 0xF) - 16];
            if ( v14 )
              goto LABEL_29;
          }
          v18 = 0;
          v17 = byte_3F871B3;
          goto LABEL_32;
        }
LABEL_29:
        v16 = *(v14 - 16);
        if ( (v16 & 2) != 0 )
        {
          v17 = (const char *)**((_QWORD **)v14 - 4);
          if ( v17 )
          {
LABEL_31:
            v17 = (const char *)sub_B91420((__int64)v17);
            goto LABEL_32;
          }
        }
        else
        {
          v17 = *(const char **)&v14[-8 * ((v16 >> 2) & 0xF) - 16];
          if ( v17 )
            goto LABEL_31;
        }
        v18 = 0;
LABEL_32:
        v184 = v12;
        v187.m128i_i64[0] = (__int64)v17;
        v185 = v172;
        v187.m128i_i64[1] = v18;
        v186 = v173;
        if ( (unsigned __int8)sub_3581350((__int64)&v176, (__int64)&v184, &v188) )
        {
          v19 = v188;
          v20 = *(_DWORD *)(v188 + 56);
          v21 = *(_QWORD *)(v188 + 40);
          v22 = v188 + 32;
          if ( !v20 )
            goto LABEL_83;
          v23 = 1;
          v24 = 0;
          v25 = (v20 - 1) & v158;
          v26 = (__int64 *)(v21 + 8LL * v25);
          v27 = *v26;
          if ( *v26 != v161 )
          {
            while ( v27 != -4096 )
            {
              if ( v24 || v27 != -8192 )
                v26 = v24;
              v25 = (v20 - 1) & (v23 + v25);
              v27 = *(_QWORD *)(v21 + 8LL * v25);
              if ( v161 == v27 )
                goto LABEL_35;
              ++v23;
              v24 = v26;
              v26 = (__int64 *)(v21 + 8LL * v25);
            }
            if ( !v24 )
              v24 = v26;
            v144 = *(_DWORD *)(v188 + 48);
            ++*(_QWORD *)(v188 + 32);
            v57 = v144 + 1;
            if ( 4 * v57 < 3 * v20 )
            {
              if ( v20 - *(_DWORD *)(v19 + 52) - v57 <= v20 >> 3 )
              {
                sub_2E61F50(v22, v20);
                v145 = *(_DWORD *)(v19 + 56);
                if ( !v145 )
                {
LABEL_285:
                  ++*(_DWORD *)(v22 + 16);
                  BUG();
                }
                v146 = v145 - 1;
                v147 = *(_QWORD *)(v19 + 40);
                v148 = 0;
                v149 = 1;
                v150 = (v145 - 1) & v158;
                v24 = (__int64 *)(v147 + 8LL * v150);
                v151 = *v24;
                v57 = *(_DWORD *)(v19 + 48) + 1;
                if ( v161 != *v24 )
                {
                  while ( v151 != -4096 )
                  {
                    if ( v151 == -8192 && !v148 )
                      v148 = v24;
                    v150 = v146 & (v149 + v150);
                    v24 = (__int64 *)(v147 + 8LL * v150);
                    v151 = *v24;
                    if ( v161 == *v24 )
                      goto LABEL_86;
                    ++v149;
                  }
                  goto LABEL_276;
                }
              }
              goto LABEL_86;
            }
LABEL_84:
            sub_2E61F50(v22, 2 * v20);
            v52 = *(_DWORD *)(v22 + 24);
            if ( !v52 )
              goto LABEL_285;
            v53 = v52 - 1;
            v54 = *(_QWORD *)(v22 + 8);
            v55 = v53 & v158;
            v24 = (__int64 *)(v54 + 8LL * (v53 & v158));
            v56 = *v24;
            v57 = *(_DWORD *)(v22 + 16) + 1;
            if ( v161 != *v24 )
            {
              v153 = 1;
              v148 = 0;
              while ( v56 != -4096 )
              {
                if ( v56 == -8192 && !v148 )
                  v148 = v24;
                v55 = v53 & (v153 + v55);
                v24 = (__int64 *)(v54 + 8LL * v55);
                v56 = *v24;
                if ( v161 == *v24 )
                  goto LABEL_86;
                ++v153;
              }
LABEL_276:
              if ( v148 )
                v24 = v148;
            }
LABEL_86:
            *(_DWORD *)(v22 + 16) = v57;
            if ( *v24 != -4096 )
              --*(_DWORD *)(v22 + 20);
            *v24 = v161;
            if ( *(_DWORD *)(v22 + 16) == 1 )
              goto LABEL_52;
            v28 = (unsigned __int8)sub_3581570((__int64)&v180, (__int64)&v184, &v188) == 0;
            v58 = v188;
            if ( v28 )
            {
              v59 = v183;
              v191[0] = v188;
              ++v180;
              v60 = v182 + 1;
              if ( 4 * ((int)v182 + 1) >= 3 * v183 )
              {
                v59 = 2 * v183;
              }
              else if ( v183 - HIDWORD(v182) - v60 > v183 >> 3 )
              {
LABEL_92:
                LODWORD(v182) = v60;
                if ( *(_QWORD *)(v58 + 16) != -1
                  || *(_DWORD *)(v58 + 12) != -1
                  || *(_DWORD *)(v58 + 8) != -1
                  || *(_QWORD *)v58 != -1 )
                {
                  --HIDWORD(v182);
                }
                v61 = v186;
                v62 = _mm_loadu_si128(&v187);
                *(_DWORD *)(v58 + 32) = 0;
                *(_DWORD *)(v58 + 12) = v61;
                v63 = v185;
                *(__m128i *)(v58 + 16) = v62;
                *(_DWORD *)(v58 + 8) = v63;
                *(_QWORD *)v58 = v184;
                goto LABEL_96;
              }
              sub_3581A20((__int64)&v180, v59);
              sub_3581570((__int64)&v180, (__int64)&v184, v191);
              v60 = v182 + 1;
              v58 = v191[0];
              goto LABEL_92;
            }
LABEL_96:
            v35 = *(_DWORD *)(v58 + 32) + 1;
            *(_DWORD *)(v58 + 32) = v35;
LABEL_44:
            v171 = v35 << a1[53];
            if ( LOBYTE(qword_503F1C8[8]) )
            {
LABEL_45:
              v36 = sub_26BDBC0(v174, v172 | v157 & (unsigned int)v171);
              if ( v36 )
              {
                sub_B10CB0(v191, (__int64)v36);
                if ( v166 == v191 )
                {
                  if ( v191[0] )
                    sub_B91220((__int64)v166, v191[0]);
                }
                else
                {
                  v37 = *(_QWORD *)(v7 + 56);
                  if ( v37 )
                    sub_B91220((__int64)v166, v37);
                  v38 = (unsigned __int8 *)v191[0];
                  *(_QWORD *)(v7 + 56) = v191[0];
                  if ( v38 )
                    sub_B976B0((__int64)v191, v38, (__int64)v166);
                }
                v156 = 1;
              }
              goto LABEL_52;
            }
            v80 = *(_DWORD *)(v174 + 4);
            if ( v80 <= 9 )
            {
              v188 = (__int64)v190;
              sub_2240A50(&v188, 1u, 0);
              v87 = (_BYTE *)v188;
              goto LABEL_143;
            }
            if ( v80 <= 0x63 )
            {
              v188 = (__int64)v190;
              sub_2240A50(&v188, 2u, 0);
              v87 = (_BYTE *)v188;
            }
            else
            {
              if ( v80 <= 0x3E7 )
              {
                v86 = 3;
                v81 = *(_DWORD *)(v174 + 4);
              }
              else
              {
                v81 = *(_DWORD *)(v174 + 4);
                v82 = v80;
                if ( v80 <= 0x270F )
                {
                  v86 = 4;
                }
                else
                {
                  v83 = 1;
                  do
                  {
                    v84 = v82;
                    v85 = v83;
                    v83 += 4;
                    v82 /= 0x2710u;
                    if ( v84 <= 0x1869F )
                    {
                      v86 = v83;
                      goto LABEL_138;
                    }
                    if ( (unsigned int)v82 <= 0x63 )
                    {
                      v86 = v85 + 5;
                      v188 = (__int64)v190;
                      goto LABEL_139;
                    }
                    if ( (unsigned int)v82 <= 0x3E7 )
                    {
                      v86 = v85 + 6;
                      goto LABEL_138;
                    }
                  }
                  while ( (unsigned int)v82 > 0x270F );
                  v86 = v85 + 7;
                }
              }
LABEL_138:
              v188 = (__int64)v190;
LABEL_139:
              sub_2240A50(&v188, v86, 0);
              v87 = (_BYTE *)v188;
              v88 = v81;
              v89 = v189 - 1;
              while ( 1 )
              {
                v90 = v80;
                v91 = v80;
                v80 = v88 / 0x64;
                v92 = 2 * (v90 - 100 * (v88 / 0x64));
                v93 = (unsigned int)(v92 + 1);
                LOBYTE(v92) = a00010203040506[v92];
                v87[v89] = a00010203040506[v93];
                v94 = (unsigned int)(v89 - 1);
                v89 -= 2;
                v87[v94] = v92;
                if ( v91 <= 0x270F )
                  break;
                v88 = v80;
              }
              if ( v91 <= 0x3E7 )
              {
LABEL_143:
                *v87 = v80 + 48;
                goto LABEL_144;
              }
            }
            v110 = 2 * v80;
            v87[1] = a00010203040506[(unsigned int)(v110 + 1)];
            *v87 = a00010203040506[v110];
LABEL_144:
            v95 = v189;
            v96 = (int *)v188;
            v169 = v189;
            if ( v189 )
            {
              sub_C7D030(v191);
              sub_C7D280((int *)v191, v96, v95);
              sub_C7D290(v191, v175);
              v96 = (int *)v188;
              v169 = v175[0];
            }
            if ( v96 != (int *)v190 )
              j_j___libc_free_0((unsigned __int64)v96);
            v97 = (int *)sub_2E31BC0(v161);
            v99 = v98;
            if ( v98 )
            {
              sub_C7D030(v191);
              sub_C7D280((int *)v191, v97, v99);
              sub_C7D290(v191, &v188);
              v169 ^= v188;
            }
            v100 = *(_BYTE *)(v174 - 16);
            if ( (v100 & 2) != 0 )
              v101 = *(unsigned __int8 ***)(v174 - 32);
            else
              v101 = (unsigned __int8 **)(v164 - 8LL * ((v100 >> 2) & 0xF));
            v102 = sub_AF34D0(*v101);
            v103 = *(v102 - 16);
            if ( (v103 & 2) != 0 )
              v104 = *((_QWORD *)v102 - 4);
            else
              v104 = (__int64)&v102[-8 * ((v103 >> 2) & 0xF) - 16];
            v105 = *(_QWORD *)(v104 + 24);
            if ( v105 )
            {
              v106 = (int *)sub_B91420(v105);
              v108 = v107;
              if ( v107 )
              {
                sub_C7D030(v191);
                sub_C7D280((int *)v191, v106, v108);
                sub_C7D290(v191, &v188);
                v169 ^= v188;
              }
            }
            v109 = *(_BYTE *)(v174 - 16);
            if ( (v109 & 2) != 0 )
            {
              if ( *(_DWORD *)(v174 - 24) != 2 )
                goto LABEL_159;
              v111 = *(_QWORD *)(v174 - 32);
            }
            else
            {
              if ( ((*(_WORD *)(v174 - 16) >> 6) & 0xF) != 2 )
                goto LABEL_159;
              v111 = v164 - 8LL * ((v109 >> 2) & 0xF);
            }
            v112 = *(_QWORD *)(v111 + 8);
            if ( !v112 )
            {
LABEL_159:
              v171 += v169;
              goto LABEL_45;
            }
            v159 = v7;
            v113 = *(_DWORD *)(v112 + 4);
            if ( v113 <= 9 )
              goto LABEL_211;
LABEL_175:
            if ( v113 <= 0x63 )
            {
              v188 = (__int64)v190;
              sub_2240A50(&v188, 2u, 0);
              v119 = (_BYTE *)v188;
            }
            else
            {
              if ( v113 <= 0x3E7 )
              {
                v116 = 3;
                v114 = v113;
              }
              else
              {
                v114 = v113;
                v115 = v113;
                if ( v113 <= 0x270F )
                {
                  v116 = 4;
                }
                else
                {
                  LODWORD(v116) = 1;
                  while ( 1 )
                  {
                    v117 = v115;
                    v118 = v116;
                    v116 = (unsigned int)(v116 + 4);
                    v115 /= 0x2710u;
                    if ( v117 <= 0x1869F )
                      break;
                    if ( (unsigned int)v115 <= 0x63 )
                    {
                      v116 = (unsigned int)(v118 + 5);
                      v188 = (__int64)v190;
                      goto LABEL_184;
                    }
                    if ( (unsigned int)v115 <= 0x3E7 )
                    {
                      v116 = (unsigned int)(v118 + 6);
                      break;
                    }
                    if ( (unsigned int)v115 <= 0x270F )
                    {
                      v116 = (unsigned int)(v118 + 7);
                      break;
                    }
                  }
                }
              }
              v188 = (__int64)v190;
LABEL_184:
              sub_2240A50(&v188, v116, 0);
              v119 = (_BYTE *)v188;
              v120 = v114;
              v121 = v189 - 1;
              while ( 1 )
              {
                v122 = (unsigned __int64)(1374389535 * v120) >> 37;
                v123 = v113 - 100 * v122;
                v124 = v113;
                v113 = v122;
                v125 = (unsigned int)(2 * v123);
                v126 = (unsigned int)(v125 + 1);
                LOBYTE(v125) = a00010203040506[v125];
                v119[v121] = a00010203040506[v126];
                v127 = (unsigned int)(v121 - 1);
                v121 -= 2;
                v119[v127] = v125;
                if ( v124 <= 0x270F )
                  break;
                v120 = v113;
              }
              if ( v124 <= 0x3E7 )
              {
                while ( 1 )
                {
                  *v119 = v113 + 48;
LABEL_189:
                  v128 = v189;
                  v129 = (int *)v188;
                  if ( v189 )
                  {
                    sub_C7D030(v191);
                    sub_C7D280((int *)v191, v129, v128);
                    sub_C7D290(v191, v175);
                    LODWORD(v128) = v175[0];
                    v129 = (int *)v188;
                  }
                  if ( v129 != (int *)v190 )
                    j_j___libc_free_0((unsigned __int64)v129);
                  v130 = *(_BYTE *)(v112 - 16);
                  v131 = v112 - 16;
                  if ( (v130 & 2) != 0 )
                    v132 = *(unsigned __int8 ***)(v112 - 32);
                  else
                    v132 = (unsigned __int8 **)(v131 - 8LL * ((v130 >> 2) & 0xF));
                  v133 = sub_AF34D0(*v132);
                  v134 = *(v133 - 16);
                  if ( (v134 & 2) != 0 )
                    v135 = *((_QWORD *)v133 - 4);
                  else
                    v135 = (__int64)&v133[-8 * ((v134 >> 2) & 0xF) - 16];
                  v136 = *(_QWORD *)(v135 + 24);
                  if ( v136 )
                  {
                    v163 = (int *)sub_B91420(v136);
                    v165 = v137;
                    if ( v137 )
                    {
                      sub_C7D030(v191);
                      sub_C7D280((int *)v191, v163, v165);
                      sub_C7D290(v191, &v188);
                      LODWORD(v128) = v188 ^ v128;
                    }
                  }
                  v138 = *(_BYTE *)(v112 - 16);
                  v169 ^= v128;
                  if ( (v138 & 2) != 0 )
                  {
                    if ( *(_DWORD *)(v112 - 24) != 2 )
                      goto LABEL_202;
                    v140 = *(_QWORD *)(v112 - 32);
                  }
                  else
                  {
                    if ( ((*(_WORD *)(v112 - 16) >> 6) & 0xF) != 2 )
                      goto LABEL_202;
                    v140 = v131 - 8LL * ((v138 >> 2) & 0xF);
                  }
                  v112 = *(_QWORD *)(v140 + 8);
                  if ( !v112 )
                  {
LABEL_202:
                    v7 = v159;
                    goto LABEL_159;
                  }
                  v113 = *(_DWORD *)(v112 + 4);
                  if ( v113 > 9 )
                    goto LABEL_175;
LABEL_211:
                  v188 = (__int64)v190;
                  sub_2240A50(&v188, 1u, 0);
                  v119 = (_BYTE *)v188;
                }
              }
            }
            v139 = 2 * v113;
            v119[1] = a00010203040506[(unsigned int)(v139 + 1)];
            *v119 = a00010203040506[v139];
            goto LABEL_189;
          }
LABEL_35:
          if ( *(_DWORD *)(v188 + 48) != 1 )
          {
            v28 = (unsigned __int8)sub_3581570((__int64)&v180, (__int64)&v184, &v188) == 0;
            v29 = v188;
            if ( v28 )
            {
              v30 = v183;
              v191[0] = v188;
              ++v180;
              v31 = v182 + 1;
              if ( 4 * ((int)v182 + 1) >= 3 * v183 )
              {
                v30 = 2 * v183;
              }
              else if ( v183 - HIDWORD(v182) - v31 > v183 >> 3 )
              {
LABEL_39:
                LODWORD(v182) = v31;
                if ( *(_QWORD *)(v29 + 16) != -1
                  || *(_DWORD *)(v29 + 12) != -1
                  || *(_DWORD *)(v29 + 8) != -1
                  || *(_QWORD *)v29 != -1 )
                {
                  --HIDWORD(v182);
                }
                v32 = v186;
                v33 = _mm_loadu_si128(&v187);
                *(_DWORD *)(v29 + 32) = 0;
                *(_DWORD *)(v29 + 12) = v32;
                v34 = v185;
                *(__m128i *)(v29 + 16) = v33;
                *(_DWORD *)(v29 + 8) = v34;
                *(_QWORD *)v29 = v184;
                goto LABEL_43;
              }
              sub_3581A20((__int64)&v180, v30);
              sub_3581570((__int64)&v180, (__int64)&v184, v191);
              v31 = v182 + 1;
              v29 = v191[0];
              goto LABEL_39;
            }
LABEL_43:
            v35 = *(_DWORD *)(v29 + 32);
            goto LABEL_44;
          }
          goto LABEL_52;
        }
        v45 = v179;
        v46 = v188;
        ++v176;
        v47 = v178 + 1;
        v191[0] = v188;
        if ( 4 * ((int)v178 + 1) >= 3 * v179 )
        {
          v45 = 2 * v179;
        }
        else if ( v179 - HIDWORD(v178) - v47 > v179 >> 3 )
        {
LABEL_79:
          LODWORD(v178) = v47;
          if ( *(_QWORD *)(v46 + 16) != -1
            || *(_DWORD *)(v46 + 12) != -1
            || *(_DWORD *)(v46 + 8) != -1
            || *(_QWORD *)v46 != -1 )
          {
            --HIDWORD(v178);
          }
          v48 = v186;
          v49 = _mm_loadu_si128(&v187);
          *(_QWORD *)(v46 + 32) = 0;
          v22 = v46 + 32;
          *(_QWORD *)(v46 + 40) = 0;
          *(_DWORD *)(v46 + 12) = v48;
          v50 = v185;
          *(_QWORD *)(v46 + 48) = 0;
          *(_DWORD *)(v46 + 8) = v50;
          v51 = v184;
          *(_DWORD *)(v46 + 56) = 0;
          *(_QWORD *)v46 = v51;
          *(__m128i *)(v46 + 16) = v49;
LABEL_83:
          ++*(_QWORD *)v22;
          v20 = 0;
          goto LABEL_84;
        }
        sub_3581780((__int64)&v176, v45);
        sub_3581350((__int64)&v176, (__int64)&v184, v191);
        v47 = v178 + 1;
        v46 = v191[0];
        goto LABEL_79;
      }
      if ( v10 )
      {
        v13 = *(_QWORD **)(v174 - 32);
        if ( *(_DWORD *)(v174 - 24) != 2 )
          goto LABEL_26;
      }
      else
      {
        v13 = (_QWORD *)(v164 - 8LL * ((v9 >> 2) & 0xF));
        if ( ((*(_WORD *)(v174 - 16) >> 6) & 0xF) != 2 )
          goto LABEL_26;
      }
      v64 = v13[1];
      if ( !v64 )
        goto LABEL_26;
      v162 = v7;
      v12 = 0;
      while ( 1 )
      {
        v65 = v64 - 16;
        LOBYTE(v191[0]) = *(_DWORD *)(v64 + 4);
        v66 = ((v12 >> 2) + sub_CBF760(v191, 1u) + 2654435769LL + (v12 << 6)) ^ v12;
        v67 = *(_BYTE *)(v64 - 16);
        if ( (v67 & 2) != 0 )
          v68 = *(unsigned __int8 ***)(v64 - 32);
        else
          v68 = (unsigned __int8 **)(v65 - 8LL * ((v67 >> 2) & 0xF));
        v69 = sub_AF34D0(*v68);
        v70 = 0;
        v71 = (char *)byte_3F871B3;
        v72 = v69;
        if ( !v69 )
          goto LABEL_108;
        v168 = v69 - 16;
        v73 = *(v69 - 16);
        if ( (v73 & 2) != 0 )
        {
          v74 = *(_QWORD *)(*((_QWORD *)v72 - 4) + 24LL);
          if ( !v74 )
            goto LABEL_161;
        }
        else
        {
          v74 = *(_QWORD *)&v168[-8 * ((v73 >> 2) & 0xF) + 24];
          if ( !v74 )
            goto LABEL_118;
        }
        v75 = sub_B91420(v74);
        v70 = v76;
        v71 = (char *)v75;
        if ( !v76 )
          break;
LABEL_108:
        v12 = (sub_CBF760(v71, v70) + (v66 >> 2) + (v66 << 6) + 2654435769u) ^ v66;
        v77 = *(_BYTE *)(v64 - 16);
        if ( (v77 & 2) != 0 )
        {
          if ( *(_DWORD *)(v64 - 24) != 2 )
            goto LABEL_110;
          v78 = *(_QWORD *)(v64 - 32);
        }
        else
        {
          if ( ((*(_WORD *)(v64 - 16) >> 6) & 0xF) != 2 )
            goto LABEL_110;
          v78 = v65 - 8LL * ((v77 >> 2) & 0xF);
        }
        v64 = *(_QWORD *)(v78 + 8);
        if ( !v64 )
        {
LABEL_110:
          v7 = v162;
          v9 = *(_BYTE *)(v174 - 16);
          v10 = (v9 & 2) != 0;
          goto LABEL_24;
        }
      }
      v73 = *(v72 - 16);
      if ( (v73 & 2) != 0 )
      {
LABEL_161:
        v71 = *(char **)(*((_QWORD *)v72 - 4) + 16LL);
        if ( !v71 )
        {
LABEL_162:
          v70 = 0;
          goto LABEL_108;
        }
      }
      else
      {
LABEL_118:
        v71 = *(char **)&v168[-8 * ((v73 >> 2) & 0xF) + 16];
        if ( !v71 )
          goto LABEL_162;
      }
      v71 = (char *)sub_B91420((__int64)v71);
      v70 = v79;
      goto LABEL_108;
    }
LABEL_54:
    v161 = *(_QWORD *)(v161 + 8);
  }
  while ( v155 != (__int64 *)v161 );
  if ( v156 )
    sub_2A61200(*(__int64 ***)(*a2 + 40));
  v39 = v181;
  v40 = 40LL * v183;
LABEL_58:
  sub_C7D6A0(v39, v40, 8);
  v41 = v179;
  if ( !v179 )
    goto LABEL_249;
  v42 = v177;
  v43 = v177 + ((unsigned __int64)v179 << 6);
  while ( 2 )
  {
    v44 = *(_QWORD *)(v42 + 16);
    if ( v44 != -1 )
    {
      if ( v44 == -2 && *(_DWORD *)(v42 + 12) == -2 && *(_DWORD *)(v42 + 8) == -2 && *(_QWORD *)v42 == -2 )
        goto LABEL_62;
LABEL_61:
      sub_C7D6A0(*(_QWORD *)(v42 + 40), 8LL * *(unsigned int *)(v42 + 56), 8);
      goto LABEL_62;
    }
    if ( *(_DWORD *)(v42 + 12) != -1 || *(_DWORD *)(v42 + 8) != -1 || *(_QWORD *)v42 != -1 )
      goto LABEL_61;
LABEL_62:
    v42 += 64;
    if ( v43 != v42 )
      continue;
    break;
  }
  v41 = v179;
LABEL_249:
  sub_C7D6A0(v177, v41 << 6, 8);
  return v156;
}
