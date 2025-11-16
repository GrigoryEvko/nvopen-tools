// Function: sub_2B70420
// Address: 0x2b70420
//
__int64 __fastcall sub_2B70420(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        int a4,
        __int64 a5,
        const __m128i *a6,
        _DWORD *a7,
        const void *a8,
        __int64 a9,
        unsigned int *a10,
        __int64 a11)
{
  __int64 *v11; // r14
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  char *v19; // rcx
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // r9
  int v22; // edx
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // r8
  __int64 v27; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rcx
  int v30; // eax
  signed __int64 v31; // r15
  __int64 v32; // r8
  unsigned __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 result; // rax
  __int64 **v36; // r14
  __int64 *v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // rdi
  int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // r8
  __int64 v43; // r13
  __int64 *v44; // r10
  __int64 v45; // rax
  __int64 **v46; // rax
  __int64 *v47; // r13
  __int64 v48; // rax
  __int64 *v49; // r8
  signed __int64 v50; // rsi
  unsigned __int64 v51; // r15
  unsigned int *v52; // r13
  __int64 *v53; // r12
  __int64 v54; // rax
  unsigned __int64 v55; // rax
  char v56; // dl
  __int64 v57; // rax
  __int64 v58; // r10
  __int64 v59; // r8
  __int64 v60; // r9
  unsigned int *v61; // rax
  __int64 v62; // r10
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r13
  __int64 v68; // rax
  __int64 v69; // r8
  unsigned __int8 **v70; // r15
  unsigned int v71; // esi
  int v72; // r11d
  unsigned int v73; // eax
  _QWORD *v74; // r12
  __int64 v75; // rdx
  __int64 v76; // rdi
  __int64 v77; // rsi
  _QWORD *v78; // rax
  __int64 v79; // r13
  __int64 v80; // rdi
  char v81; // di
  __int64 v82; // rsi
  _QWORD *v83; // rax
  int v84; // edx
  int v85; // r11d
  unsigned int v86; // eax
  __int64 *v87; // rsi
  __int64 v88; // rdi
  int v89; // r8d
  __int64 v90; // rax
  __int64 *v91; // r13
  __int64 *v92; // r15
  unsigned int v93; // eax
  __int64 v94; // rdi
  unsigned int v95; // esi
  _QWORD *v96; // r10
  __int64 *v97; // r15
  __int64 v98; // rdx
  __int64 *v99; // r13
  __int64 v100; // rcx
  __int64 v101; // r12
  __int64 *v102; // rax
  __int64 *v103; // rdx
  __int64 **v104; // r14
  __int64 v105; // r13
  __int64 *v106; // rdx
  __int64 v107; // rax
  __int64 v108; // rdx
  _BYTE *v109; // rdx
  __int64 **v110; // r12
  __int64 v111; // rsi
  int v112; // eax
  __int64 v113; // rcx
  __int64 v114; // r15
  __int64 v115; // rax
  __int64 *v116; // rdx
  char v117; // di
  bool v118; // zf
  __int64 *v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 *v122; // rdx
  __int64 v123; // rax
  __int64 v124; // rdx
  __int64 *v125; // rdx
  __int64 v126; // rax
  __int64 v127; // rdx
  unsigned int *v128; // r15
  __int64 v129; // r9
  __int64 v130; // rax
  unsigned __int64 v131; // rcx
  __int64 v132; // rax
  __int64 v133; // r15
  _QWORD *v134; // rax
  __int64 v135; // r13
  __int64 v136; // rax
  int v137; // eax
  int v138; // edi
  char *v139; // rax
  int v140; // r11d
  int v141; // eax
  __int64 *v142; // rdx
  __int64 v143; // rdx
  __int64 v144; // rcx
  __int64 *v145; // rdx
  __int64 v146; // rdx
  __int64 v147; // rcx
  __int64 *v148; // rdx
  __int64 v149; // rdx
  __int64 v150; // rcx
  __int64 *v151; // rax
  int v152; // r13d
  unsigned int v153; // r11d
  _QWORD *v154; // r10
  __int64 v155; // [rsp+8h] [rbp-F8h]
  __int64 v156; // [rsp+8h] [rbp-F8h]
  __int64 v157; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v158; // [rsp+10h] [rbp-F0h]
  __int64 v159; // [rsp+10h] [rbp-F0h]
  __int64 v160; // [rsp+10h] [rbp-F0h]
  unsigned __int8 **v161; // [rsp+10h] [rbp-F0h]
  char *v162; // [rsp+10h] [rbp-F0h]
  void *src; // [rsp+18h] [rbp-E8h]
  __int64 v165; // [rsp+20h] [rbp-E0h]
  __int64 v166; // [rsp+20h] [rbp-E0h]
  size_t nb; // [rsp+30h] [rbp-D0h]
  size_t n; // [rsp+30h] [rbp-D0h]
  size_t nc; // [rsp+30h] [rbp-D0h]
  size_t na; // [rsp+30h] [rbp-D0h]
  char v172; // [rsp+38h] [rbp-C8h]
  __int64 *v173; // [rsp+38h] [rbp-C8h]
  unsigned int *v175; // [rsp+48h] [rbp-B8h]
  __int64 v176; // [rsp+50h] [rbp-B0h] BYREF
  __int64 *v177; // [rsp+58h] [rbp-A8h] BYREF
  unsigned __int8 *v178[2]; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int *v179; // [rsp+70h] [rbp-90h]
  char v180; // [rsp+80h] [rbp-80h]
  _BYTE *v181; // [rsp+90h] [rbp-70h] BYREF
  __int64 **v182; // [rsp+98h] [rbp-68h]
  __int64 v183; // [rsp+A0h] [rbp-60h]
  int v184; // [rsp+A8h] [rbp-58h]
  char v185; // [rsp+ACh] [rbp-54h]
  _BYTE v186[80]; // [rsp+B0h] [rbp-50h] BYREF

  v11 = a2;
  if ( !*(_BYTE *)(a1 + 1256)
    || a4 != 3
    || !a6->m128i_i64[0]
    || !a6->m128i_i64[1]
    || *(_BYTE *)a6->m128i_i64[0] != 61
    || a7[2] != -1
    || (result = *(_QWORD *)a7) != 0 )
  {
    v15 = sub_22077B0(0x1B8u);
    v16 = v15;
    if ( v15 )
    {
      *(_QWORD *)(v15 + 80) = 6;
      *(_QWORD *)v15 = v15 + 16;
      *(_QWORD *)(v15 + 8) = 0x800000000LL;
      *(_QWORD *)(v15 + 112) = v15 + 128;
      *(_QWORD *)(v15 + 120) = 0x400000000LL;
      *(_QWORD *)(v15 + 152) = 0x400000000LL;
      *(_QWORD *)(v15 + 144) = v15 + 160;
      *(_QWORD *)(v15 + 208) = v15 + 224;
      *(_QWORD *)(v15 + 88) = 0;
      *(_QWORD *)(v15 + 96) = 0;
      *(_DWORD *)(v15 + 108) = -1;
      *(_QWORD *)(v15 + 176) = a1;
      *(_QWORD *)(v15 + 184) = 0;
      *(_DWORD *)(v15 + 192) = -1;
      *(_DWORD *)(v15 + 200) = 0;
      *(_QWORD *)(v15 + 216) = 0x200000000LL;
      *(_QWORD *)(v15 + 240) = v15 + 256;
      *(_QWORD *)(v15 + 248) = 0x200000000LL;
      *(_QWORD *)(v15 + 416) = 0;
      *(_QWORD *)(v15 + 424) = 0;
      *(_DWORD *)(v15 + 432) = 0;
    }
    v17 = *(unsigned int *)(a1 + 8);
    v18 = *(unsigned int *)(a1 + 12);
    v19 = (char *)&v181;
    v181 = (_BYTE *)v16;
    v20 = *(_QWORD *)a1;
    v21 = v17 + 1;
    v22 = v17;
    if ( v17 + 1 > v18 )
    {
      if ( v20 > (unsigned __int64)&v181 )
      {
        v157 = v16;
        sub_2B47650(a1, v17 + 1, v17, (__int64)&v181, v16, v21);
        v17 = *(unsigned int *)(a1 + 8);
        v20 = *(_QWORD *)a1;
        v19 = (char *)&v181;
        v16 = v157;
        v22 = *(_DWORD *)(a1 + 8);
      }
      else
      {
        v156 = v16;
        if ( (unsigned __int64)&v181 < v20 + 8 * v17 )
        {
          v162 = (char *)&v181 - v20;
          sub_2B47650(a1, v21, v17, (__int64)&v181, v16, v21);
          v20 = *(_QWORD *)a1;
          v16 = v156;
          v19 = &v162[*(_QWORD *)a1];
          v17 = *(unsigned int *)(a1 + 8);
        }
        else
        {
          sub_2B47650(a1, v21, v17, (__int64)&v181, v16, v21);
          v17 = *(unsigned int *)(a1 + 8);
          v20 = *(_QWORD *)a1;
          v16 = v156;
          v19 = (char *)&v181;
        }
        v22 = *(_DWORD *)(a1 + 8);
      }
    }
    v23 = (_QWORD *)(v20 + 8 * v17);
    if ( v23 )
    {
      *v23 = *(_QWORD *)v19;
      *(_QWORD *)v19 = 0;
      v16 = (__int64)v181;
      v22 = *(_DWORD *)(a1 + 8);
    }
    v24 = (unsigned int)(v22 + 1);
    *(_DWORD *)(a1 + 8) = v24;
    if ( v16 )
    {
      v158 = v16;
      sub_2B2F3A0(v16);
      j_j___libc_free_0(v158);
      v24 = *(unsigned int *)(a1 + 8);
      v22 = v24 - 1;
    }
    v25 = 4 * a9;
    v26 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v24 - 8);
    v27 = (4 * a9) >> 2;
    *(_DWORD *)(v26 + 104) = a4;
    v28 = *(unsigned int *)(v26 + 120);
    v29 = *(unsigned int *)(v26 + 124);
    *(_DWORD *)(v26 + 200) = v22;
    v176 = v26;
    if ( v27 + v28 > v29 )
    {
      v160 = v26;
      sub_C8D5F0(v26 + 112, (const void *)(v26 + 128), v27 + v28, 4u, v26, v25);
      v26 = v160;
      v25 = 4 * a9;
      v28 = *(unsigned int *)(v160 + 120);
    }
    if ( v25 )
    {
      v159 = v26;
      memcpy((void *)(*(_QWORD *)(v26 + 112) + 4 * v28), a8, v25);
      v26 = v159;
      LODWORD(v28) = *(_DWORD *)(v159 + 120);
    }
    v30 = v27 + v28;
    v31 = 8 * a3;
    *(_DWORD *)(v26 + 120) = v30;
    LODWORD(v28) = a11;
    v32 = v176;
    if ( a11 )
    {
      sub_2B39BA0(v176, a3, 0, v29, v176, v25);
      v48 = v176;
      v49 = *(__int64 **)v176;
      v50 = 4 * a11;
      if ( a10 != &a10[a11] )
      {
        v165 = 8 * a3;
        v51 = a3;
        v52 = a10;
        v155 = a5;
        v53 = *(__int64 **)v176;
        do
        {
          v55 = *v52;
          if ( v51 > v55 )
            v54 = a2[v55];
          else
            v54 = sub_ACA8A0(*(__int64 ***)(*a2 + 8));
          ++v52;
          *v53++ = v54;
        }
        while ( &a10[(unsigned __int64)v50 / 4] != v52 );
        v48 = v176;
        v31 = v165;
        a5 = v155;
        v49 = *(__int64 **)v176;
      }
      v65 = sub_2B5F980(v49, *(unsigned int *)(v48 + 8), *(__int64 **)(a1 + 3304));
      v67 = v176;
      if ( v65 && v66 )
      {
        *(_QWORD *)(v176 + 416) = v65;
        *(_QWORD *)(v67 + 424) = v66;
      }
      v68 = *(unsigned int *)(v67 + 152);
      v69 = v50 >> 2;
      v33 = (v50 >> 2) + v68;
      if ( v33 > *(unsigned int *)(v67 + 156) )
      {
        sub_C8D5F0(v67 + 144, (const void *)(v67 + 160), v33, 4u, v69, v25);
        v68 = *(unsigned int *)(v67 + 152);
        v69 = v50 >> 2;
      }
      v29 = 4 * a11;
      if ( v50 )
      {
        nc = v69;
        memcpy((void *)(*(_QWORD *)(v67 + 144) + 4 * v68), a10, v50);
        v68 = *(unsigned int *)(v67 + 152);
        v69 = nc;
      }
      v32 = v68 + v69;
      result = v176;
      *(_DWORD *)(v67 + 152) = v32;
    }
    else
    {
      v33 = *(unsigned int *)(v176 + 12);
      *(_DWORD *)(v176 + 8) = 0;
      v34 = 0;
      if ( v31 >> 3 > v33 )
      {
        na = v32;
        sub_C8D5F0(v32, (const void *)(v32 + 16), v31 >> 3, 8u, v32, v25);
        v32 = na;
        v28 = *(unsigned int *)(na + 8);
        v34 = 8 * v28;
      }
      if ( v31 )
      {
        nb = v32;
        memcpy((void *)(*(_QWORD *)v32 + v34), a2, v31);
        v32 = nb;
        LODWORD(v28) = *(_DWORD *)(nb + 8);
      }
      *(_DWORD *)(v32 + 8) = (v31 >> 3) + v28;
      result = v176;
      if ( a6->m128i_i64[0] && a6->m128i_i64[1] )
        *(__m128i *)(v176 + 416) = _mm_loadu_si128(a6);
    }
    n = (size_t)a2 + v31;
    if ( a4 != 5 )
    {
      if ( *(_DWORD *)(result + 104) != 3 )
      {
        v181 = 0;
        v182 = (__int64 **)v186;
        v183 = 4;
        v184 = 0;
        v185 = 1;
        if ( a2 == (__int64 *)n )
          goto LABEL_45;
        v36 = (__int64 **)a2;
        while ( 1 )
        {
          v37 = *v36;
          v177 = v37;
          if ( *(_BYTE *)v37 == 13 )
            goto LABEL_38;
          v38 = *(_BYTE *)(a1 + 88) & 1;
          if ( (*(_BYTE *)(a1 + 88) & 1) != 0 )
          {
            v39 = a1 + 96;
            v40 = 3;
          }
          else
          {
            v41 = *(unsigned int *)(a1 + 104);
            v39 = *(_QWORD *)(a1 + 96);
            if ( !(_DWORD)v41 )
              goto LABEL_83;
            v40 = v41 - 1;
          }
          v41 = v40 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
          v42 = 9 * v41;
          v43 = v39 + 72 * v41;
          v44 = *(__int64 **)v43;
          if ( v37 != *(__int64 **)v43 )
            break;
LABEL_30:
          v45 = 288;
          if ( !(_BYTE)v38 )
            v45 = 72LL * *(unsigned int *)(a1 + 104);
          if ( v43 == v39 + v45 )
          {
            sub_2B4BD40((__int64)v178, a1 + 80, (__int64 *)&v177);
            v61 = v179;
            v62 = v176;
            v63 = v179[4];
            if ( v63 + 1 > (unsigned __int64)v179[5] )
            {
              v175 = v179;
              sub_C8D5F0((__int64)(v179 + 2), v179 + 6, v63 + 1, 8u, v59, v60);
              v61 = v175;
              v62 = v176;
              v63 = v175[4];
            }
            v64 = *((_QWORD *)v61 + 1);
            ++v36;
            *(_QWORD *)(v64 + 8 * v63) = v62;
            ++v61[4];
            sub_2411830((__int64)v178, (__int64)&v181, v177, v64, v59, v60);
            if ( (__int64 **)n == v36 )
            {
LABEL_39:
              v47 = a2;
              if ( a5 )
              {
                do
                {
                  if ( !(unsigned __int8)sub_2B14730(*v47) && a5 )
                  {
                    *(_QWORD *)(a5 + 8) = v176;
                    a5 = *(_QWORD *)(a5 + 24);
                  }
                  ++v47;
                }
                while ( (__int64 *)n != v47 );
              }
              result = v176;
              if ( v185 )
              {
LABEL_45:
                if ( *(_QWORD *)a7 )
                {
                  *(_QWORD *)(result + 184) = *(_QWORD *)a7;
                  *(_DWORD *)(result + 192) = a7[2];
                }
                return result;
              }
LABEL_131:
              _libc_free((unsigned __int64)v182);
              result = v176;
              goto LABEL_45;
            }
          }
          else
          {
            if ( v185 )
            {
              v46 = v182;
              v38 = HIDWORD(v183);
              v41 = (__int64)&v182[HIDWORD(v183)];
              if ( v182 != (__int64 **)v41 )
              {
                while ( v37 != *v46 )
                {
                  if ( (__int64 **)v41 == ++v46 )
                    goto LABEL_71;
                }
                goto LABEL_38;
              }
LABEL_71:
              if ( HIDWORD(v183) < (unsigned int)v183 )
              {
                ++HIDWORD(v183);
                *(_QWORD *)v41 = v37;
                ++v181;
                goto LABEL_63;
              }
            }
            sub_C8CC70((__int64)&v181, (__int64)v37, v41, v38, v42, v25);
            if ( v56 )
            {
LABEL_63:
              v57 = *(unsigned int *)(v43 + 16);
              v58 = v176;
              if ( v57 + 1 > (unsigned __int64)*(unsigned int *)(v43 + 20) )
              {
                sub_C8D5F0(v43 + 8, (const void *)(v43 + 24), v57 + 1, 8u, v42, v25);
                v57 = *(unsigned int *)(v43 + 16);
                v58 = v176;
              }
              ++v36;
              *(_QWORD *)(*(_QWORD *)(v43 + 8) + 8 * v57) = v58;
              ++*(_DWORD *)(v43 + 16);
              if ( (__int64 **)n == v36 )
                goto LABEL_39;
            }
            else
            {
LABEL_38:
              if ( (__int64 **)n == ++v36 )
                goto LABEL_39;
            }
          }
        }
        v89 = 1;
        while ( v44 != (__int64 *)-4096LL )
        {
          v25 = (unsigned int)(v89 + 1);
          v41 = v40 & (unsigned int)(v89 + v41);
          v42 = 9LL * (unsigned int)v41;
          v43 = v39 + 72LL * (unsigned int)v41;
          v44 = *(__int64 **)v43;
          if ( v37 == *(__int64 **)v43 )
            goto LABEL_30;
          v89 = v25;
        }
        if ( (_BYTE)v38 )
        {
          v42 = 288;
        }
        else
        {
          v41 = *(unsigned int *)(a1 + 104);
LABEL_83:
          v42 = 72 * v41;
        }
        v43 = v39 + v42;
        goto LABEL_30;
      }
      if ( a2 == (__int64 *)n )
        goto LABEL_175;
      v70 = (unsigned __int8 **)a2;
      v172 = 1;
      v166 = a1 + 1168;
      do
      {
        v178[0] = *v70;
        if ( !(unsigned __int8)sub_2B0D8B0(v178[0]) )
        {
          v172 = (unsigned __int8)(*(_BYTE *)v29 - 67) <= 0xCu
              && (*(_BYTE *)(*(_QWORD *)(v29 + 8) + 8LL) == 12) & (unsigned __int8)v172;
          if ( a7[2] != -1 || !*(_QWORD *)a7 || *(_DWORD *)(*(_QWORD *)a7 + 104LL) != 3 )
          {
            v71 = *(_DWORD *)(a1 + 1192);
            if ( !v71 )
            {
              ++*(_QWORD *)(a1 + 1168);
              v181 = 0;
              goto LABEL_235;
            }
            v25 = v71 - 1;
            v32 = *(_QWORD *)(a1 + 1176);
            v72 = 1;
            v73 = v25 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
            v74 = (_QWORD *)(v32 + 88LL * v73);
            v75 = 0;
            v76 = *v74;
            if ( v29 == *v74 )
              goto LABEL_95;
            while ( 1 )
            {
              if ( v76 == -4096 )
              {
                v137 = *(_DWORD *)(a1 + 1184);
                if ( !v75 )
                  v75 = (__int64)v74;
                ++*(_QWORD *)(a1 + 1168);
                v138 = v137 + 1;
                v181 = (_BYTE *)v75;
                if ( 4 * (v137 + 1) < 3 * v71 )
                {
                  v32 = v71 >> 3;
                  if ( v71 - *(_DWORD *)(a1 + 1188) - v138 > (unsigned int)v32 )
                  {
LABEL_229:
                    *(_DWORD *)(a1 + 1184) = v138;
                    if ( *(_QWORD *)v75 != -4096 )
                      --*(_DWORD *)(a1 + 1188);
                    *(_QWORD *)v75 = v29;
                    v29 = v75 + 56;
                    v77 = 0;
                    *(_QWORD *)(v75 + 40) = v75 + 56;
                    v79 = v176;
                    *(_QWORD *)(v75 + 48) = 0x400000000LL;
                    *(_OWORD *)(v75 + 8) = 0;
                    *(_OWORD *)(v75 + 24) = 0;
                    *(_OWORD *)(v75 + 56) = 0;
                    *(_OWORD *)(v75 + 72) = 0;
                    goto LABEL_232;
                  }
LABEL_236:
                  sub_2B4C110(v166, v71);
                  sub_2B3DE10(v166, (__int64 *)v178, &v181);
                  v29 = (unsigned __int64)v178[0];
                  v75 = (__int64)v181;
                  v138 = *(_DWORD *)(a1 + 1184) + 1;
                  goto LABEL_229;
                }
LABEL_235:
                v71 *= 2;
                goto LABEL_236;
              }
              if ( v75 || v76 != -8192 )
                v74 = (_QWORD *)v75;
              v152 = v72 + 1;
              v153 = v73 + v72;
              v73 = v25 & v153;
              v154 = (_QWORD *)(v32 + 88LL * ((unsigned int)v25 & v153));
              v76 = *v154;
              if ( v29 == *v154 )
                break;
              v75 = (__int64)v74;
              v72 = v152;
              v74 = v154;
            }
            v74 = (_QWORD *)(v32 + 88LL * ((unsigned int)v25 & v153));
LABEL_95:
            if ( *((_DWORD *)v74 + 6) )
            {
              v84 = *((_DWORD *)v74 + 8);
              if ( v84 )
              {
                v29 = v176;
                v33 = (unsigned int)(v84 - 1);
                v32 = v74[2];
                v25 = 0;
                v85 = 1;
                v86 = v33 & (((unsigned int)v176 >> 9) ^ ((unsigned int)v176 >> 4));
                v87 = (__int64 *)(v32 + 8LL * v86);
                v88 = *v87;
                if ( v176 == *v87 )
                  goto LABEL_104;
                while ( v88 != -4096 )
                {
                  if ( !v25 && v88 == -8192 )
                    v25 = (__int64)v87;
                  v86 = v33 & (v85 + v86);
                  v87 = (__int64 *)(v32 + 8LL * v86);
                  v88 = *v87;
                  if ( v176 == *v87 )
                    goto LABEL_104;
                  ++v85;
                }
                if ( !v25 )
                  v25 = (__int64)v87;
              }
              else
              {
                v25 = 0;
              }
              v134 = sub_2B5CC20((__int64)(v74 + 1), &v176, (_QWORD *)v25);
              v135 = v176;
              *v134 = v176;
              v136 = *((unsigned int *)v74 + 12);
              v29 = *((unsigned int *)v74 + 13);
              if ( v136 + 1 > v29 )
              {
                sub_C8D5F0((__int64)(v74 + 5), v74 + 7, v136 + 1, 8u, v32, v25);
                v136 = *((unsigned int *)v74 + 12);
              }
              v33 = v74[5];
              *(_QWORD *)(v33 + 8 * v136) = v135;
              ++*((_DWORD *)v74 + 12);
            }
            else
            {
              v77 = *((unsigned int *)v74 + 12);
              v78 = (_QWORD *)v74[5];
              v79 = v176;
              v29 = (unsigned __int64)&v78[v77];
              v80 = (8 * v77) >> 3;
              v33 = (8 * v77) >> 5;
              if ( v33 )
              {
                while ( 1 )
                {
                  if ( v176 == *v78 )
                    goto LABEL_103;
                  if ( v176 == v78[1] )
                    break;
                  if ( v176 == v78[2] )
                  {
                    v78 += 2;
                    goto LABEL_103;
                  }
                  if ( v176 == v78[3] )
                  {
                    v78 += 3;
                    goto LABEL_103;
                  }
                  v78 += 4;
                  if ( !--v33 )
                  {
                    v80 = (__int64)(v29 - (_QWORD)v78) >> 3;
                    goto LABEL_133;
                  }
                }
                ++v78;
                goto LABEL_103;
              }
LABEL_133:
              if ( v80 == 2 )
                goto LABEL_134;
              if ( v80 != 3 )
              {
                if ( v80 == 1 )
                {
LABEL_136:
                  if ( v176 == *v78 )
                    goto LABEL_103;
                  goto LABEL_137;
                }
                v75 = (__int64)v74;
LABEL_232:
                v74 = (_QWORD *)v75;
                goto LABEL_137;
              }
              if ( v176 == *v78 )
                goto LABEL_103;
              ++v78;
LABEL_134:
              if ( v176 != *v78 )
              {
                ++v78;
                goto LABEL_136;
              }
LABEL_103:
              if ( v78 != (_QWORD *)v29 )
                goto LABEL_104;
LABEL_137:
              v33 = v77 + 1;
              if ( v77 + 1 > (unsigned __int64)*((unsigned int *)v74 + 13) )
              {
                sub_C8D5F0((__int64)(v74 + 5), v74 + 7, v33, 8u, v32, v25);
                v33 = *((unsigned int *)v74 + 12);
                v29 = v74[5] + 8 * v33;
              }
              *(_QWORD *)v29 = v79;
              v90 = (unsigned int)(*((_DWORD *)v74 + 12) + 1);
              *((_DWORD *)v74 + 12) = v90;
              if ( (unsigned int)v90 > 4 )
              {
                v91 = (__int64 *)v74[5];
                v161 = v70;
                src = v74 + 1;
                v92 = &v91[v90];
                while ( 1 )
                {
                  v95 = *((_DWORD *)v74 + 8);
                  if ( !v95 )
                    break;
                  v33 = *v91;
                  v25 = v95 - 1;
                  v32 = v74[2];
                  v93 = v25 & (((unsigned int)*v91 >> 9) ^ ((unsigned int)*v91 >> 4));
                  v29 = v32 + 8LL * v93;
                  v94 = *(_QWORD *)v29;
                  if ( *(_QWORD *)v29 != *v91 )
                  {
                    v140 = 1;
                    v96 = 0;
                    while ( v94 != -4096 )
                    {
                      if ( !v96 && v94 == -8192 )
                        v96 = (_QWORD *)v29;
                      v93 = v25 & (v140 + v93);
                      v29 = v32 + 8LL * v93;
                      v94 = *(_QWORD *)v29;
                      if ( v33 == *(_QWORD *)v29 )
                        goto LABEL_142;
                      ++v140;
                    }
                    if ( !v96 )
                      v96 = (_QWORD *)v29;
                    v181 = v96;
                    v141 = *((_DWORD *)v74 + 6);
                    ++v74[1];
                    v33 = (unsigned int)(v141 + 1);
                    if ( 4 * (int)v33 < 3 * v95 )
                    {
                      v29 = v95 >> 3;
                      if ( v95 - *((_DWORD *)v74 + 7) - (unsigned int)v33 <= (unsigned int)v29 )
                      {
LABEL_146:
                        sub_2B5CA50((__int64)src, v95);
                        sub_2B42300((__int64)src, v91, &v181);
                        v96 = v181;
                        v33 = (unsigned int)(*((_DWORD *)v74 + 6) + 1);
                      }
                      *((_DWORD *)v74 + 6) = v33;
                      if ( *v96 != -4096 )
                        --*((_DWORD *)v74 + 7);
                      *v96 = *v91;
                      goto LABEL_142;
                    }
LABEL_145:
                    v95 *= 2;
                    goto LABEL_146;
                  }
LABEL_142:
                  if ( v92 == ++v91 )
                  {
                    v70 = v161;
                    goto LABEL_104;
                  }
                }
                v181 = 0;
                ++v74[1];
                goto LABEL_145;
              }
            }
          }
        }
LABEL_104:
        ++v70;
      }
      while ( (unsigned __int8 **)n != v70 );
      if ( v172 )
      {
LABEL_175:
        v118 = *(_BYTE *)(a1 + 3568) == 0;
        *(_QWORD *)(a1 + 3560) = 0x1FFFFFFFFLL;
        if ( v118 )
          *(_BYTE *)(a1 + 3568) = 1;
        if ( a2 == (__int64 *)n )
        {
LABEL_113:
          result = v176;
          goto LABEL_45;
        }
      }
      v81 = *(_BYTE *)(a1 + 796);
      while ( 1 )
      {
        while ( 1 )
        {
          v82 = *v11;
          if ( v81 )
            break;
LABEL_114:
          ++v11;
          sub_C8CC70(a1 + 768, v82, v33, v29, v32, v25);
          v81 = *(_BYTE *)(a1 + 796);
          if ( (__int64 *)n == v11 )
            goto LABEL_113;
        }
        v83 = *(_QWORD **)(a1 + 776);
        v29 = *(unsigned int *)(a1 + 788);
        v33 = (unsigned __int64)&v83[v29];
        if ( v83 == (_QWORD *)v33 )
        {
LABEL_116:
          if ( (unsigned int)v29 >= *(_DWORD *)(a1 + 784) )
            goto LABEL_114;
          v29 = (unsigned int)(v29 + 1);
          ++v11;
          *(_DWORD *)(a1 + 788) = v29;
          *(_QWORD *)v33 = v82;
          v81 = *(_BYTE *)(a1 + 796);
          ++*(_QWORD *)(a1 + 768);
          if ( (__int64 *)n == v11 )
            goto LABEL_113;
        }
        else
        {
          while ( v82 != *v83 )
          {
            if ( (_QWORD *)v33 == ++v83 )
              goto LABEL_116;
          }
          if ( (__int64 *)n == ++v11 )
            goto LABEL_113;
        }
      }
    }
    v97 = *(__int64 **)result;
    v98 = 8LL * *(unsigned int *)(result + 8);
    v99 = (__int64 *)(*(_QWORD *)result + v98);
    v100 = v98 >> 3;
    v101 = v98 >> 5;
    if ( v98 >> 5 )
    {
      v102 = *(__int64 **)result;
      v103 = &v97[4 * v101];
      do
      {
        v104 = (__int64 **)*v102;
        if ( *(_BYTE *)*v102 > 0x1Cu )
          goto LABEL_157;
        v104 = (__int64 **)v102[1];
        if ( *(_BYTE *)v104 > 0x1Cu )
          goto LABEL_157;
        v104 = (__int64 **)v102[2];
        if ( *(_BYTE *)v104 > 0x1Cu )
          goto LABEL_157;
        v104 = (__int64 **)v102[3];
        if ( *(_BYTE *)v104 > 0x1Cu )
          goto LABEL_157;
        v102 += 4;
      }
      while ( v103 != v102 );
      v139 = (char *)((char *)v99 - (char *)v103);
      if ( (char *)v99 - (char *)v103 == 16 )
      {
        v104 = (__int64 **)*v103;
        if ( *(_BYTE *)*v103 > 0x1Cu )
          goto LABEL_157;
        v104 = (__int64 **)v103[1];
        if ( *(_BYTE *)v104 > 0x1Cu )
          goto LABEL_157;
LABEL_273:
        v104 = (__int64 **)*v99;
        goto LABEL_274;
      }
      if ( v139 != (char *)24 )
      {
        if ( v139 != (char *)8 )
        {
          v173 = v99;
          v104 = (__int64 **)*v99;
          v105 = v101;
          goto LABEL_158;
        }
        v104 = (__int64 **)*v103;
        if ( *(_BYTE *)*v103 <= 0x1Cu )
          goto LABEL_273;
LABEL_157:
        v173 = v99;
        v105 = v101;
LABEL_158:
        while ( 1 )
        {
          if ( *(_BYTE *)*v97 > 0x1Cu )
          {
            v106 = *(__int64 **)(a1 + 3304);
            v181 = (_BYTE *)*v97;
            v182 = v104;
            v107 = sub_2B5F980((__int64 *)&v181, 2u, v106);
            if ( v108 == 0 || v107 == 0 || v107 != v108 )
              break;
          }
          if ( *(_BYTE *)v97[1] > 0x1Cu )
          {
            v119 = *(__int64 **)(a1 + 3304);
            v181 = (_BYTE *)v97[1];
            v182 = v104;
            v120 = sub_2B5F980((__int64 *)&v181, 2u, v119);
            if ( v120 == 0 || v121 == 0 || v120 != v121 )
            {
              result = v176;
              ++v97;
              goto LABEL_162;
            }
          }
          if ( *(_BYTE *)v97[2] > 0x1Cu )
          {
            v122 = *(__int64 **)(a1 + 3304);
            v181 = (_BYTE *)v97[2];
            v182 = v104;
            v123 = sub_2B5F980((__int64 *)&v181, 2u, v122);
            if ( v124 == 0 || v123 == 0 || v123 != v124 )
            {
              result = v176;
              v97 += 2;
              goto LABEL_162;
            }
          }
          if ( *(_BYTE *)v97[3] > 0x1Cu )
          {
            v125 = *(__int64 **)(a1 + 3304);
            v181 = (_BYTE *)v97[3];
            v182 = v104;
            v126 = sub_2B5F980((__int64 *)&v181, 2u, v125);
            if ( v126 == 0 || v127 == 0 || v126 != v127 )
            {
              result = v176;
              v97 += 3;
              goto LABEL_162;
            }
          }
          v97 += 4;
          if ( !--v105 )
          {
            v99 = v173;
            v100 = v173 - v97;
            goto LABEL_193;
          }
        }
        result = v176;
LABEL_162:
        v109 = (_BYTE *)*v97;
        v110 = (__int64 **)a2;
        *(_QWORD *)(result + 416) = v104;
        *(_QWORD *)(result + 424) = v109;
        v181 = 0;
        v182 = (__int64 **)v186;
        v183 = 4;
        v184 = 0;
        v185 = 1;
        if ( a2 == (__int64 *)n )
          goto LABEL_45;
        while ( 1 )
        {
          v116 = *v110;
          v177 = v116;
          if ( *(_BYTE *)v116 > 0x1Cu )
            break;
LABEL_170:
          if ( (__int64 **)n == ++v110 )
          {
            result = v176;
            if ( v185 )
              goto LABEL_45;
            goto LABEL_131;
          }
        }
        v117 = *(_BYTE *)(a1 + 392) & 1;
        if ( v117 )
        {
          v111 = a1 + 400;
          v112 = 3;
        }
        else
        {
          v113 = *(unsigned int *)(a1 + 408);
          v111 = *(_QWORD *)(a1 + 400);
          if ( !(_DWORD)v113 )
            goto LABEL_205;
          v112 = v113 - 1;
        }
        v113 = v112 & (((unsigned int)v116 >> 9) ^ ((unsigned int)v116 >> 4));
        v114 = v111 + 72 * v113;
        v32 = *(_QWORD *)v114;
        if ( v116 == *(__int64 **)v114 )
        {
LABEL_166:
          v115 = 288;
          if ( !v117 )
            v115 = 72LL * *(unsigned int *)(a1 + 408);
          if ( v114 == v111 + v115 )
          {
            sub_2B4BD40((__int64)v178, a1 + 384, (__int64 *)&v177);
            v128 = v179;
            v129 = v176;
            v130 = v179[4];
            v131 = v179[5];
            if ( v130 + 1 > v131 )
            {
              sub_C8D5F0((__int64)(v179 + 2), v179 + 6, v130 + 1, 8u, (__int64)v178, v176);
              v130 = v128[4];
              v129 = v176;
            }
            *(_QWORD *)(*((_QWORD *)v128 + 1) + 8 * v130) = v129;
            ++v128[4];
            sub_2411830((__int64)v178, (__int64)&v181, v177, v131, (__int64)v178, v129);
          }
          else
          {
            sub_2411830((__int64)v178, (__int64)&v181, v116, v113, v32, v25);
            if ( v180 )
            {
              v132 = *(unsigned int *)(v114 + 16);
              v32 = v176;
              if ( v132 + 1 > (unsigned __int64)*(unsigned int *)(v114 + 20) )
              {
                sub_C8D5F0(v114 + 8, (const void *)(v114 + 24), v132 + 1, 8u, v176, v25);
                v132 = *(unsigned int *)(v114 + 16);
                v32 = v176;
              }
              *(_QWORD *)(*(_QWORD *)(v114 + 8) + 8 * v132) = v32;
              ++*(_DWORD *)(v114 + 16);
            }
          }
          goto LABEL_170;
        }
        v25 = 1;
        while ( v32 != -4096 )
        {
          v113 = v112 & (unsigned int)(v25 + v113);
          v114 = v111 + 72LL * (unsigned int)v113;
          v32 = *(_QWORD *)v114;
          if ( v116 == *(__int64 **)v114 )
            goto LABEL_166;
          v25 = (unsigned int)(v25 + 1);
        }
        if ( v117 )
        {
          v133 = 288;
          goto LABEL_206;
        }
        v113 = *(unsigned int *)(a1 + 408);
LABEL_205:
        v133 = 72 * v113;
LABEL_206:
        v114 = v111 + v133;
        goto LABEL_166;
      }
      v104 = (__int64 **)*v103;
      if ( *(_BYTE *)*v103 > 0x1Cu )
        goto LABEL_157;
      v104 = (__int64 **)v103[1];
      v151 = v103 + 1;
      if ( *(_BYTE *)v104 > 0x1Cu )
        goto LABEL_157;
    }
    else
    {
      if ( v98 == 16 )
      {
        v104 = (__int64 **)*v97;
        if ( *(_BYTE *)*v97 > 0x1Cu )
          goto LABEL_264;
        v104 = (__int64 **)v97[1];
        if ( *(_BYTE *)v104 > 0x1Cu )
        {
LABEL_193:
          if ( v100 != 2 )
          {
            if ( v100 != 3 )
            {
              if ( v100 != 1 )
              {
LABEL_196:
                v97 = v99;
LABEL_197:
                result = v176;
                goto LABEL_162;
              }
LABEL_268:
              if ( *(_BYTE *)*v97 > 0x1Cu )
              {
                v148 = *(__int64 **)(a1 + 3304);
                v181 = (_BYTE *)*v97;
                v182 = v104;
                v150 = sub_2B5F980((__int64 *)&v181, 2u, v148);
                if ( v150 == 0 || v149 == 0 || v150 != v149 )
                  goto LABEL_197;
              }
              goto LABEL_196;
            }
            goto LABEL_260;
          }
LABEL_264:
          if ( *(_BYTE *)*v97 > 0x1Cu )
          {
            v145 = *(__int64 **)(a1 + 3304);
            v181 = (_BYTE *)*v97;
            v182 = v104;
            v147 = sub_2B5F980((__int64 *)&v181, 2u, v145);
            result = v176;
            if ( v147 == 0 || v146 == 0 || v147 != v146 )
              goto LABEL_162;
          }
          ++v97;
          goto LABEL_268;
        }
        goto LABEL_273;
      }
      if ( v98 != 24 )
      {
        if ( v98 != 8 )
        {
          v104 = (__int64 **)*v99;
          v97 = (__int64 *)(*(_QWORD *)result + v98);
          goto LABEL_162;
        }
        v104 = (__int64 **)*v97;
        if ( *(_BYTE *)*v97 > 0x1Cu )
          goto LABEL_268;
        goto LABEL_273;
      }
      v104 = (__int64 **)*v97;
      if ( *(_BYTE *)*v97 > 0x1Cu )
      {
LABEL_260:
        if ( *(_BYTE *)*v97 > 0x1Cu )
        {
          v142 = *(__int64 **)(a1 + 3304);
          v181 = (_BYTE *)*v97;
          v182 = v104;
          v144 = sub_2B5F980((__int64 *)&v181, 2u, v142);
          result = v176;
          if ( v144 == 0 || v143 == 0 || v144 != v143 )
            goto LABEL_162;
        }
        ++v97;
        goto LABEL_264;
      }
      v104 = (__int64 **)v97[1];
      v151 = v97 + 1;
      if ( *(_BYTE *)v104 > 0x1Cu )
        goto LABEL_193;
    }
    v104 = (__int64 **)v151[1];
    if ( *(_BYTE *)v104 > 0x1Cu )
    {
LABEL_274:
      if ( !v101 )
        goto LABEL_193;
      goto LABEL_157;
    }
    goto LABEL_273;
  }
  return result;
}
