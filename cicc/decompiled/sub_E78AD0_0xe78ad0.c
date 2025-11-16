// Function: sub_E78AD0
// Address: 0xe78ad0
//
__int64 __fastcall sub_E78AD0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int16 a5,
        unsigned int a6,
        __int128 a7,
        char a8,
        __int128 a9,
        __int64 a10)
{
  __int64 v13; // rbx
  size_t v14; // rdx
  unsigned __int64 v15; // r9
  size_t v16; // rdx
  int v17; // eax
  char v18; // al
  char v19; // al
  int v20; // eax
  __int64 v21; // r8
  __int64 v22; // rsi
  __int64 **v23; // rcx
  size_t v24; // r15
  __int64 v25; // rsi
  __int64 v26; // r14
  const void *v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // r14
  int v31; // r12d
  unsigned __int64 v32; // rsi
  unsigned int v33; // edx
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // r14
  const void *v37; // rdi
  __int64 v38; // rax
  const void *v39; // rdx
  __int64 v40; // rcx
  size_t v41; // r14
  const void *v42; // r15
  int v43; // eax
  unsigned int v44; // r8d
  __int64 *v45; // r11
  __int64 v46; // rax
  _BYTE *v47; // rdi
  char v48; // dl
  unsigned int v49; // eax
  __int64 v50; // rdx
  __int64 v51; // r12
  unsigned int v52; // r14d
  __int64 v53; // rax
  __int64 v54; // rbx
  __int64 v56; // rax
  int v57; // r9d
  unsigned int v58; // r8d
  __int64 *v59; // r11
  __int64 v60; // rcx
  int v61; // eax
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // r15
  __int64 v65; // rdx
  __int64 v66; // r14
  _BYTE *v67; // r8
  size_t v68; // r14
  __int64 *v69; // rax
  __int64 *v70; // rdi
  __int64 v71; // rdx
  char v72; // al
  __m128i v73; // xmm2
  __m128i v74; // xmm3
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 *v77; // rdi
  __int64 v78; // rdx
  char *v79; // rax
  __int64 v80; // rdx
  bool v81; // zf
  _BYTE *v82; // r9
  __m128i *v83; // r8
  _QWORD *v84; // rax
  __int64 v85; // rax
  __int64 v86; // rsi
  int v87; // edx
  __int64 v88; // rsi
  __m128i *v89; // rax
  __int64 v90; // rsi
  _QWORD *v91; // rdi
  _QWORD *v92; // r14
  _QWORD *v93; // rbx
  __int64 v94; // rax
  _QWORD *v95; // rdi
  unsigned __int64 v96; // rax
  const void *v97; // rsi
  int v98; // eax
  int v99; // eax
  int v100; // eax
  int v101; // r9d
  unsigned __int64 v102; // r15
  __int64 v103; // rdi
  __int64 v104; // [rsp+10h] [rbp-210h]
  __int64 v105; // [rsp+18h] [rbp-208h]
  __int64 v106; // [rsp+18h] [rbp-208h]
  __int64 *v107; // [rsp+20h] [rbp-200h]
  __int64 **v108; // [rsp+20h] [rbp-200h]
  __int64 **v109; // [rsp+20h] [rbp-200h]
  __int64 **v110; // [rsp+28h] [rbp-1F8h]
  unsigned int v111; // [rsp+28h] [rbp-1F8h]
  _BYTE *v112; // [rsp+28h] [rbp-1F8h]
  __int64 **v113; // [rsp+28h] [rbp-1F8h]
  __int64 v114; // [rsp+30h] [rbp-1F0h]
  __int64 v115; // [rsp+30h] [rbp-1F0h]
  __int64 **v116; // [rsp+30h] [rbp-1F0h]
  unsigned int src; // [rsp+38h] [rbp-1E8h]
  signed __int64 srca; // [rsp+38h] [rbp-1E8h]
  __int64 *srcb; // [rsp+38h] [rbp-1E8h]
  _BYTE *srcd; // [rsp+38h] [rbp-1E8h]
  __int64 **srce; // [rsp+38h] [rbp-1E8h]
  __int64 **srcf; // [rsp+38h] [rbp-1E8h]
  __int64 **srcc; // [rsp+38h] [rbp-1E8h]
  char v124; // [rsp+42h] [rbp-1DEh]
  char v125; // [rsp+43h] [rbp-1DDh]
  int v127; // [rsp+44h] [rbp-1DCh]
  __int64 **v129; // [rsp+48h] [rbp-1D8h]
  __int64 **v130; // [rsp+48h] [rbp-1D8h]
  __int64 **v131; // [rsp+48h] [rbp-1D8h]
  __int64 **v132; // [rsp+48h] [rbp-1D8h]
  _QWORD v133[2]; // [rsp+70h] [rbp-1B0h] BYREF
  char v134; // [rsp+80h] [rbp-1A0h]
  __int16 v135; // [rsp+90h] [rbp-190h]
  _QWORD v136[2]; // [rsp+A0h] [rbp-180h] BYREF
  _QWORD v137[2]; // [rsp+B0h] [rbp-170h] BYREF
  __int16 v138; // [rsp+C0h] [rbp-160h]
  __m128i s2; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v140; // [rsp+E0h] [rbp-140h] BYREF
  _BYTE v141[312]; // [rsp+E8h] [rbp-138h] BYREF

  v13 = a2;
  v14 = *(_QWORD *)(a3 + 8);
  v124 = a8;
  v125 = a10;
  if ( *(_QWORD *)(a2 + 408) == v14 && (!v14 || !memcmp(*(const void **)a3, *(const void **)(a2 + 400), v14)) )
  {
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)a3 = byte_3F871B3;
  }
  if ( !*(_QWORD *)(a4 + 8) )
  {
    *(_QWORD *)(a4 + 8) = 7;
    *(_QWORD *)a4 = "<stdin>";
    *(_QWORD *)a3 = byte_3F871B3;
    *(_QWORD *)(a3 + 8) = 0;
  }
  v15 = *(unsigned int *)(a2 + 128);
  if ( !(_DWORD)v15 )
  {
    *(_BYTE *)(a2 + 513) &= v124;
    *(_BYTE *)(a2 + 514) |= v124;
    *(_BYTE *)(a2 + 512) |= v125;
  }
  if ( a5 > 4u )
  {
    v16 = *(_QWORD *)(a2 + 440);
    LOBYTE(v140) = a8;
    s2 = _mm_loadu_si128((const __m128i *)&a7);
    if ( v16 )
    {
      if ( v16 == *(_QWORD *)(a4 + 8) )
      {
        src = v15;
        v17 = memcmp(*(const void **)(a2 + 432), *(const void **)a4, v16);
        v15 = src;
        if ( !v17 )
        {
          v18 = *(_BYTE *)(a2 + 484);
          if ( v18 == (_BYTE)v140 )
          {
            if ( !v18 || (v20 = memcmp((const void *)(a2 + 468), &s2, 0x10u), v15 = src, !v20) )
            {
              v19 = *(_BYTE *)(a1 + 8);
              *(_DWORD *)a1 = 0;
              *(_BYTE *)(a1 + 8) = v19 & 0xFC | 2;
              return a1;
            }
          }
        }
      }
    }
  }
  v21 = a6;
  if ( !a6 )
  {
    v37 = *(const void **)a3;
    s2 = (__m128i)(unsigned __int64)v141;
    v38 = *(_QWORD *)(a4 + 8);
    srcb = (__int64 *)(a2 + 376);
    v39 = *(const void **)a4;
    v40 = *(_QWORD *)(a3 + 8);
    v133[0] = v37;
    v138 = 1282;
    v133[1] = v40;
    v135 = 2053;
    v127 = v15;
    v136[0] = v133;
    v137[0] = v39;
    v137[1] = v38;
    v140 = 256;
    v134 = 0;
    sub_CA0EC0((__int64)v136, (__int64)&s2);
    v41 = s2.m128i_u64[1];
    v42 = (const void *)s2.m128i_i64[0];
    v43 = sub_C92610();
    v44 = sub_C92740(a2 + 376, v42, v41, v43);
    v45 = (__int64 *)(*(_QWORD *)(a2 + 376) + 8LL * v44);
    v46 = *v45;
    if ( *v45 )
    {
      if ( v46 != -8 )
      {
        v47 = (_BYTE *)s2.m128i_i64[0];
        v48 = *(_BYTE *)(a1 + 8) & 0xFC;
        *(_DWORD *)a1 = *(_DWORD *)(v46 + 8);
        *(_BYTE *)(a1 + 8) = v48 | 2;
        if ( v47 != v141 )
          _libc_free(v47, v42);
        return a1;
      }
      --*(_DWORD *)(a2 + 392);
    }
    v107 = v45;
    v111 = v44;
    v56 = sub_C7D670(v41 + 17, 8);
    v57 = v127;
    v58 = v111;
    v59 = v107;
    v60 = v56;
    if ( v41 )
    {
      v106 = v56;
      memcpy((void *)(v56 + 16), v42, v41);
      v57 = v127;
      v58 = v111;
      v59 = v107;
      v60 = v106;
    }
    v61 = 1;
    *(_BYTE *)(v60 + v41 + 16) = 0;
    v62 = v58;
    if ( v57 )
      v61 = v57;
    *(_QWORD *)v60 = v41;
    *(_DWORD *)(v60 + 8) = v61;
    *v59 = v60;
    ++*(_DWORD *)(v13 + 388);
    a6 = v61;
    sub_C929D0(srcb, v58);
    if ( (_BYTE *)s2.m128i_i64[0] != v141 )
      _libc_free(s2.m128i_i64[0], v62);
    v15 = *(unsigned int *)(v13 + 128);
  }
  if ( a6 < (unsigned int)v15 || (v32 = a6 + 1, v33 = a6 + 1, v32 == v15) )
  {
    v22 = *(_QWORD *)(v13 + 120);
  }
  else
  {
    v34 = 80 * v32;
    if ( v32 < v15 )
    {
      v22 = *(_QWORD *)(v13 + 120);
      v92 = (_QWORD *)(v22 + v34);
      if ( (_QWORD *)(v22 + 80 * v15) != v92 )
      {
        v115 = v13;
        v93 = (_QWORD *)(v22 + 80 * v15);
        do
        {
          v93 -= 10;
          if ( (_QWORD *)*v93 != v93 + 2 )
            j_j___libc_free_0(*v93, v93[2] + 1LL);
        }
        while ( v92 != v93 );
        v13 = v115;
        v33 = a6 + 1;
        v22 = *(_QWORD *)(v115 + 120);
      }
    }
    else
    {
      if ( v32 > *(unsigned int *)(v13 + 132) )
      {
        sub_E78960(v13 + 120, v32, v32, a4, v21, v15);
        v15 = *(unsigned int *)(v13 + 128);
        v33 = a6 + 1;
      }
      v22 = *(_QWORD *)(v13 + 120);
      v35 = v22 + 80 * v15;
      v36 = v22 + v34;
      if ( v35 != v36 )
      {
        do
        {
          if ( v35 )
          {
            *(_QWORD *)(v35 + 8) = 0;
            *(_QWORD *)v35 = v35 + 16;
            *(_OWORD *)(v35 + 16) = 0;
            *(_OWORD *)(v35 + 32) = 0;
            *(_OWORD *)(v35 + 48) = 0;
            *(_OWORD *)(v35 + 64) = 0;
          }
          v35 += 80;
        }
        while ( v36 != v35 );
        v22 = *(_QWORD *)(v13 + 120);
      }
    }
    *(_DWORD *)(v13 + 128) = v33;
  }
  v23 = (__int64 **)(v22 + 80LL * a6);
  if ( !v23[1] )
  {
    v24 = *(_QWORD *)(a3 + 8);
    if ( v24 )
      goto LABEL_22;
    v63 = sub_C80C60(*(_QWORD *)a4, *(_QWORD *)(a4 + 8), 0);
    v23 = (__int64 **)(v22 + 80LL * a6);
    v64 = v63;
    v66 = v65;
    if ( v65 )
    {
      v79 = sub_C80DA0(*(char **)a4, *(_QWORD *)(a4 + 8), 0);
      v23 = (__int64 **)(v22 + 80LL * a6);
      *(_QWORD *)(a3 + 8) = v80;
      v81 = *(_QWORD *)(a3 + 8) == 0;
      *(_QWORD *)a3 = v79;
      if ( v81 )
        goto LABEL_68;
      *(_QWORD *)a4 = v64;
      *(_QWORD *)(a4 + 8) = v66;
    }
    v24 = *(_QWORD *)(a3 + 8);
    if ( v24 )
    {
LABEL_22:
      v25 = *(_QWORD *)(v13 + 8);
      srca = *(unsigned int *)(v13 + 16);
      v104 = v25;
      v114 = (32 * srca) >> 5;
      if ( (32 * srca) >> 7 )
      {
        v110 = v23;
        v26 = *(_QWORD *)(v13 + 8);
        v105 = v13;
        v27 = *(const void **)a3;
        while ( 1 )
        {
          if ( *(_QWORD *)(v26 + 8) == v24 && !memcmp(*(const void **)v26, v27, v24) )
          {
            v23 = v110;
            v30 = (v26 - v25) >> 5;
            v13 = v105;
            v96 = (unsigned int)v30;
            goto LABEL_120;
          }
          if ( *(_QWORD *)(v26 + 40) == v24 && !memcmp(*(const void **)(v26 + 32), v27, v24) )
          {
            v23 = v110;
            v30 = (v26 + 32 - v25) >> 5;
            v13 = v105;
            v96 = (unsigned int)v30;
            goto LABEL_120;
          }
          if ( *(_QWORD *)(v26 + 72) == v24 && !memcmp(*(const void **)(v26 + 64), v27, v24) )
          {
            v23 = v110;
            v30 = (v26 + 64 - v25) >> 5;
            v13 = v105;
            v96 = (unsigned int)v30;
            goto LABEL_120;
          }
          if ( *(_QWORD *)(v26 + 104) == v24 && !memcmp(*(const void **)(v26 + 96), v27, v24) )
            break;
          v26 += 128;
          if ( v26 == v25 + ((32 * srca) >> 7 << 7) )
          {
            v23 = v110;
            v13 = v105;
            v28 = (v25 + 32 * srca - v26) >> 5;
            goto LABEL_34;
          }
        }
        v23 = v110;
        v30 = (v26 + 96 - v25) >> 5;
        v13 = v105;
        v96 = (unsigned int)v30;
LABEL_120:
        if ( v96 < srca )
          goto LABEL_39;
        v82 = *(_BYTE **)a3;
        v136[0] = v137;
        goto LABEL_98;
      }
      v28 = (32 * srca) >> 5;
      v26 = *(_QWORD *)(v13 + 8);
LABEL_34:
      switch ( v28 )
      {
        case 2LL:
          v97 = *(const void **)a3;
          break;
        case 3LL:
          v97 = *(const void **)a3;
          if ( *(_QWORD *)(v26 + 8) == v24 )
          {
            v108 = v23;
            v98 = memcmp(*(const void **)v26, v97, v24);
            v23 = v108;
            if ( !v98 )
              goto LABEL_133;
          }
          v26 += 32;
          break;
        case 1LL:
          v97 = *(const void **)a3;
          goto LABEL_131;
        default:
LABEL_37:
          v29 = (32 * srca) >> 5;
LABEL_38:
          LODWORD(v30) = v114;
          if ( v29 < srca )
          {
LABEL_39:
            v31 = v30 + 1;
LABEL_69:
            s2.m128i_i64[0] = (__int64)&v140;
            v67 = *(_BYTE **)a4;
            v68 = *(_QWORD *)(a4 + 8);
            if ( !(v68 + *(_QWORD *)a4) || v67 )
            {
              v136[0] = *(_QWORD *)(a4 + 8);
              if ( v68 > 0xF )
              {
                srcd = v67;
                v130 = v23;
                v76 = sub_22409D0(&s2, v136, 0);
                v23 = v130;
                v67 = srcd;
                s2.m128i_i64[0] = v76;
                v77 = (__int64 *)v76;
                v140 = v136[0];
              }
              else
              {
                if ( v68 == 1 )
                {
                  LOBYTE(v140) = *v67;
                  v69 = &v140;
                  goto LABEL_74;
                }
                if ( !v68 )
                {
                  v69 = &v140;
                  goto LABEL_74;
                }
                v77 = &v140;
              }
              v131 = v23;
              memcpy(v77, v67, v68);
              v68 = v136[0];
              v69 = (__int64 *)s2.m128i_i64[0];
              v23 = v131;
LABEL_74:
              s2.m128i_i64[1] = v68;
              *((_BYTE *)v69 + v68) = 0;
              v70 = *v23;
              if ( (__int64 *)s2.m128i_i64[0] == &v140 )
              {
                v78 = s2.m128i_i64[1];
                if ( s2.m128i_i64[1] )
                {
                  if ( s2.m128i_i64[1] == 1 )
                  {
                    *(_BYTE *)v70 = v140;
                    v78 = s2.m128i_i64[1];
                    v70 = *v23;
                  }
                  else
                  {
                    v132 = v23;
                    memcpy(v70, &v140, s2.m128i_u64[1]);
                    v23 = v132;
                    v78 = s2.m128i_i64[1];
                    v70 = *v132;
                  }
                }
                v23[1] = (__int64 *)v78;
                *((_BYTE *)v70 + v78) = 0;
                v70 = (__int64 *)s2.m128i_i64[0];
                goto LABEL_78;
              }
              if ( v70 == (__int64 *)(v23 + 2) )
              {
                *(__m128i *)v23 = s2;
                v23[2] = (__int64 *)v140;
              }
              else
              {
                *v23 = (__int64 *)s2.m128i_i64[0];
                v71 = (__int64)v23[2];
                v23[1] = (__int64 *)s2.m128i_i64[1];
                v23[2] = (__int64 *)v140;
                if ( v70 )
                {
                  s2.m128i_i64[0] = (__int64)v70;
                  v140 = v71;
LABEL_78:
                  s2.m128i_i64[1] = 0;
                  *(_BYTE *)v70 = 0;
                  if ( (__int64 *)s2.m128i_i64[0] != &v140 )
                  {
                    v129 = v23;
                    j_j___libc_free_0(s2.m128i_i64[0], v140 + 1);
                    v23 = v129;
                  }
                  v72 = a8;
                  v73 = _mm_loadu_si128((const __m128i *)&a7);
                  *((_DWORD *)v23 + 8) = v31;
                  v74 = _mm_loadu_si128((const __m128i *)&a9);
                  *((_BYTE *)v23 + 52) = v72;
                  *(__m128i *)((char *)v23 + 36) = v73;
                  *(_BYTE *)(v13 + 513) &= v124;
                  *(_BYTE *)(v13 + 514) |= v124;
                  v75 = a10;
                  *(__m128i *)(v23 + 7) = v74;
                  v23[9] = (__int64 *)v75;
                  if ( v125 )
                    *(_BYTE *)(v13 + 512) = 1;
                  *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
                  *(_DWORD *)a1 = a6;
                  return a1;
                }
              }
              s2.m128i_i64[0] = (__int64)&v140;
              v70 = &v140;
              goto LABEL_78;
            }
LABEL_141:
            sub_426248((__int64)"basic_string::_M_construct null not valid");
          }
          v82 = *(_BYTE **)a3;
          v136[0] = v137;
          if ( !v82 )
            goto LABEL_141;
LABEL_98:
          s2.m128i_i64[0] = v24;
          if ( v24 > 0xF )
          {
            v112 = v82;
            v116 = v23;
            v94 = sub_22409D0(v136, &s2, 0);
            v23 = v116;
            v136[0] = v94;
            v95 = (_QWORD *)v94;
            v82 = v112;
            v137[0] = s2.m128i_i64[0];
          }
          else
          {
            if ( v24 == 1 )
            {
              v83 = (__m128i *)v136;
              LOBYTE(v137[0]) = *v82;
              v84 = v137;
LABEL_101:
              v136[1] = v24;
              *((_BYTE *)v84 + v24) = 0;
              v85 = *(unsigned int *)(v13 + 16);
              v86 = v85 + 1;
              v87 = *(_DWORD *)(v13 + 16);
              if ( v85 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 20) )
              {
                v102 = *(_QWORD *)(v13 + 8);
                v103 = v13 + 8;
                srcc = v23;
                if ( v102 > (unsigned __int64)v136 || (unsigned __int64)v136 >= v102 + 32 * v85 )
                {
                  sub_95D880(v103, v86);
                  v85 = *(unsigned int *)(v13 + 16);
                  v88 = *(_QWORD *)(v13 + 8);
                  v83 = (__m128i *)v136;
                  v23 = srcc;
                  v87 = *(_DWORD *)(v13 + 16);
                }
                else
                {
                  sub_95D880(v103, v86);
                  v88 = *(_QWORD *)(v13 + 8);
                  v85 = *(unsigned int *)(v13 + 16);
                  v23 = srcc;
                  v83 = (__m128i *)((char *)v136 + v88 - v102);
                  v87 = *(_DWORD *)(v13 + 16);
                }
              }
              else
              {
                v88 = *(_QWORD *)(v13 + 8);
              }
              v89 = (__m128i *)(v88 + 32 * v85);
              if ( v89 )
              {
                v89->m128i_i64[0] = (__int64)v89[1].m128i_i64;
                if ( (__m128i *)v83->m128i_i64[0] == &v83[1] )
                {
                  v89[1] = _mm_loadu_si128(v83 + 1);
                }
                else
                {
                  v89->m128i_i64[0] = v83->m128i_i64[0];
                  v89[1].m128i_i64[0] = v83[1].m128i_i64[0];
                }
                v90 = v83->m128i_i64[1];
                v83->m128i_i64[0] = (__int64)v83[1].m128i_i64;
                v83->m128i_i64[1] = 0;
                v89->m128i_i64[1] = v90;
                v83[1].m128i_i8[0] = 0;
                v87 = *(_DWORD *)(v13 + 16);
              }
              v91 = (_QWORD *)v136[0];
              *(_DWORD *)(v13 + 16) = v87 + 1;
              if ( v91 != v137 )
              {
                srce = v23;
                j_j___libc_free_0(v91, v137[0] + 1LL);
                v23 = srce;
              }
              goto LABEL_39;
            }
            v95 = v137;
          }
          srcf = v23;
          memcpy(v95, v82, v24);
          v24 = s2.m128i_i64[0];
          v84 = (_QWORD *)v136[0];
          v83 = (__m128i *)v136;
          v23 = srcf;
          goto LABEL_101;
      }
      if ( *(_QWORD *)(v26 + 8) == v24 )
      {
        v109 = v23;
        v99 = memcmp(*(const void **)v26, v97, v24);
        v23 = v109;
        if ( !v99 )
          goto LABEL_133;
      }
      v26 += 32;
LABEL_131:
      if ( *(_QWORD *)(v26 + 8) != v24 )
        goto LABEL_37;
      v113 = v23;
      v100 = memcmp(*(const void **)v26, v97, v24);
      v23 = v113;
      v101 = v100;
      v29 = (32 * srca) >> 5;
      if ( v101 )
        goto LABEL_38;
LABEL_133:
      v114 = (v26 - v104) >> 5;
      v29 = (unsigned int)v114;
      goto LABEL_38;
    }
LABEL_68:
    v31 = 0;
    goto LABEL_69;
  }
  v49 = sub_C63BB0();
  v141[9] = 1;
  s2.m128i_i64[0] = (__int64)"file number already allocated";
  v51 = v50;
  v52 = v49;
  v141[8] = 3;
  v53 = sub_22077B0(64);
  v54 = v53;
  if ( v53 )
    sub_C63EB0(v53, (__int64)&s2, v52, v51);
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v54 & 0xFFFFFFFFFFFFFFFELL;
  return a1;
}
