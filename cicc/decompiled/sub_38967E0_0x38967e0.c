// Function: sub_38967E0
// Address: 0x38967e0
//
__int64 __fastcall sub_38967E0(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  __int64 *v13; // r15
  __int64 *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rcx
  unsigned int *v18; // r12
  __int64 v19; // rax
  unsigned int v20; // edi
  __int64 v21; // rsi
  __int64 v22; // rcx
  __int64 v23; // rdx
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdi
  int v28; // eax
  unsigned int v29; // r12d
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // edi
  _QWORD *v34; // rsi
  __int64 *v35; // rax
  _QWORD *v36; // rax
  __int64 v37; // rax
  __int64 *v38; // rax
  __int64 v39; // r13
  _QWORD *i; // r12
  __int64 v41; // rdi
  __int64 *v42; // rbx
  __int64 v43; // r13
  __int64 v44; // rdx
  __int64 v45; // r14
  __int64 v46; // rax
  __int64 *v47; // rdi
  unsigned __int64 v48; // rdx
  __int64 v49; // rcx
  unsigned __int64 *v50; // rax
  unsigned __int64 v51; // rdi
  __int64 v52; // rbx
  __int64 v53; // rdx
  int v54; // edx
  size_t **v55; // rax
  size_t **v56; // r13
  size_t *v57; // rcx
  size_t **v58; // r15
  __int64 v59; // rbx
  size_t v60; // r12
  size_t *v61; // rdx
  size_t *v62; // rax
  __int64 k; // r12
  __int32 v64; // edx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 *v67; // rbx
  __int64 *v68; // r13
  __int64 *v69; // rdi
  __int64 *v70; // r13
  __int64 *v71; // rax
  __int64 v72; // r12
  double v73; // xmm4_8
  double v74; // xmm5_8
  __m128i *v75; // rax
  __int64 v76; // rax
  unsigned __int64 v77; // rdi
  __m128i *v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  const char *v81; // rbx
  _QWORD *v82; // r13
  void *v83; // rdi
  __int64 v84; // rax
  _QWORD *v85; // [rsp+18h] [rbp-198h]
  unsigned int v86; // [rsp+20h] [rbp-190h]
  __int64 *v87; // [rsp+28h] [rbp-188h]
  size_t **v88; // [rsp+28h] [rbp-188h]
  __int64 *src; // [rsp+30h] [rbp-180h]
  unsigned __int8 *srca; // [rsp+30h] [rbp-180h]
  __int64 v91; // [rsp+38h] [rbp-178h]
  size_t v92; // [rsp+38h] [rbp-178h]
  _QWORD *v93; // [rsp+40h] [rbp-170h]
  __int64 *j; // [rsp+40h] [rbp-170h]
  __int64 *v95; // [rsp+40h] [rbp-170h]
  unsigned int *v96; // [rsp+48h] [rbp-168h]
  _QWORD *v97; // [rsp+48h] [rbp-168h]
  __int64 v98; // [rsp+58h] [rbp-158h] BYREF
  __m128i *v99; // [rsp+60h] [rbp-150h] BYREF
  __int64 v100; // [rsp+68h] [rbp-148h]
  __int64 v101; // [rsp+70h] [rbp-140h] BYREF
  _QWORD *v102; // [rsp+78h] [rbp-138h]
  __int64 *v103; // [rsp+80h] [rbp-130h]
  __int64 *v104; // [rsp+88h] [rbp-128h]
  __int64 v105; // [rsp+90h] [rbp-120h]
  __int64 v106; // [rsp+98h] [rbp-118h]
  __int64 v107; // [rsp+A0h] [rbp-110h]
  __int64 v108; // [rsp+A8h] [rbp-108h]
  __int64 v109; // [rsp+B0h] [rbp-100h]
  __int64 v110; // [rsp+B8h] [rbp-F8h]
  __m128i v111; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v112; // [rsp+D0h] [rbp-E0h] BYREF
  _QWORD *v113; // [rsp+D8h] [rbp-D8h]
  unsigned int v114; // [rsp+F8h] [rbp-B8h]
  __m128i v115; // [rsp+120h] [rbp-90h] BYREF
  __m128i v116; // [rsp+130h] [rbp-80h] BYREF

  if ( !a1[22] )
    return 0;
  v13 = a1;
  v14 = a1 + 142;
  v15 = (__int64)(a1 + 147);
  src = (__int64 *)a1[144];
  v16 = (__int64)(a1 + 148);
  v87 = a1 + 142;
  v93 = a1 + 147;
  if ( src != a1 + 142 )
  {
    while ( 1 )
    {
      v17 = src[4];
      v99 = 0;
      LODWORD(v101) = 0;
      v91 = v17;
      v102 = 0;
      v103 = &v101;
      v104 = &v101;
      v105 = 0;
      v106 = 0;
      v107 = 0;
      v108 = 0;
      v109 = 0;
      v110 = 0;
      v18 = (unsigned int *)src[5];
      v96 = (unsigned int *)src[6];
      if ( v18 != v96 )
        break;
LABEL_13:
      v24 = *(_BYTE *)(v91 + 16);
      if ( v24 )
      {
        if ( v24 == 78 || v24 == 29 )
        {
          v98 = *(_QWORD *)(v91 + 56);
          v31 = sub_1560250(&v98);
          sub_1563030(&v111, v31);
          v98 = sub_1560530(&v98, (__int64 *)*v13, -1);
          sub_15625F0(&v111, &v99);
          v32 = sub_1560BF0((__int64 *)*v13, &v111);
          sub_1563030(&v115, v32);
          a2 = *v13;
          v98 = sub_15637E0(&v98, (__int64 *)*v13, -1, &v115);
          sub_3887AD0((_QWORD *)v116.m128i_i64[1]);
          *(_QWORD *)(v91 + 56) = v98;
          sub_3887AD0(v113);
        }
        else
        {
          sub_1563030(&v115, *(_QWORD *)(v91 + 72));
          sub_15625F0(&v115, &v99);
          a2 = (__int64)&v115;
          *(_QWORD *)(v91 + 72) = sub_1560BF0((__int64 *)*v13, &v115);
          sub_3887AD0((_QWORD *)v116.m128i_i64[1]);
        }
      }
      else
      {
        v98 = *(_QWORD *)(v91 + 112);
        v25 = sub_1560250(&v98);
        sub_1563030(&v111, v25);
        v98 = sub_1560530(&v98, (__int64 *)*v13, -1);
        sub_15625F0(&v111, &v99);
        if ( sub_1560E20((__int64)&v111) )
        {
          sub_15E4CC0(v91, v114);
          sub_1560700(&v111, 1);
        }
        v26 = sub_1560BF0((__int64 *)*v13, &v111);
        sub_1563030(&v115, v26);
        a2 = *v13;
        v98 = sub_15637E0(&v98, (__int64 *)*v13, -1, &v115);
        sub_3887AD0((_QWORD *)v116.m128i_i64[1]);
        *(_QWORD *)(v91 + 112) = v98;
        sub_3887AD0(v113);
      }
      sub_3887AD0(v102);
      src = (__int64 *)sub_220EEE0((__int64)src);
      if ( v87 == src )
        goto LABEL_18;
    }
    while ( 1 )
    {
      v19 = v13[149];
      if ( v19 )
      {
        v20 = *v18;
        v21 = v16;
        do
        {
          while ( 1 )
          {
            v22 = *(_QWORD *)(v19 + 16);
            v23 = *(_QWORD *)(v19 + 24);
            if ( *(_DWORD *)(v19 + 32) >= v20 )
              break;
            v19 = *(_QWORD *)(v19 + 24);
            if ( !v23 )
              goto LABEL_9;
          }
          v21 = v19;
          v19 = *(_QWORD *)(v19 + 16);
        }
        while ( v22 );
LABEL_9:
        if ( v21 != v16 && v20 >= *(_DWORD *)(v21 + 32) )
          goto LABEL_12;
      }
      else
      {
        v21 = v16;
      }
      v115.m128i_i64[0] = (__int64)v18;
      v21 = sub_3896710(v93, v21, (unsigned int **)&v115);
LABEL_12:
      ++v18;
      sub_15625F0(&v99, (_QWORD *)(v21 + 40));
      if ( v96 == v18 )
        goto LABEL_13;
    }
  }
LABEL_18:
  if ( v13[139] )
  {
    v115.m128i_i64[0] = (__int64)"expected function name in blockaddress";
    v46 = v13[137];
    v116.m128i_i16[0] = 259;
    return (unsigned int)sub_38814C0((__int64)(v13 + 1), *(_QWORD *)(v46 + 40), (__int64)&v115);
  }
  v27 = v13[98];
  v97 = v13 + 96;
  if ( v13 + 96 == (__int64 *)v27 )
  {
LABEL_30:
    v33 = *((_DWORD *)v13 + 184);
    if ( v33 )
    {
      v34 = (_QWORD *)v13[91];
      if ( *v34 && *v34 != -8 )
      {
        v14 = (__int64 *)v13[91];
      }
      else
      {
        v35 = v34 + 1;
        do
        {
          do
          {
            v15 = *v35;
            v14 = v35++;
          }
          while ( v15 == -8 );
        }
        while ( !v15 );
      }
      a2 = (__int64)&v34[v33];
LABEL_37:
      if ( v14 != (__int64 *)a2 )
      {
        while ( 1 )
        {
          v36 = (_QWORD *)*v14;
          if ( *(_QWORD *)(*v14 + 16) )
            break;
          v37 = v14[1];
          v15 = (__int64)(v14 + 1);
          if ( v37 && v37 != -8 )
          {
            ++v14;
            goto LABEL_37;
          }
          v38 = v14 + 2;
          do
          {
            do
            {
              v15 = *v38;
              v14 = v38++;
            }
            while ( v15 == -8 );
          }
          while ( !v15 );
          if ( v14 == (__int64 *)a2 )
            goto LABEL_45;
        }
        v66 = *v36;
        v99 = (__m128i *)(v36 + 3);
        v111.m128i_i64[0] = (__int64)"use of undefined type named '";
        v111.m128i_i64[1] = (__int64)&v99;
        v115.m128i_i64[0] = (__int64)&v111;
        v100 = v66;
        LOWORD(v112) = 1283;
        v115.m128i_i64[1] = (__int64)"'";
        v116.m128i_i16[0] = 770;
        return (unsigned int)sub_38814C0((__int64)(v13 + 1), *(_QWORD *)(*v14 + 16), (__int64)&v115);
      }
    }
LABEL_45:
    if ( v13[133] )
    {
      sub_8FD6D0((__int64)&v99, "use of undefined comdat '$", (_QWORD *)(v13[131] + 32));
      if ( v100 != 0x3FFFFFFFFFFFFFFFLL )
      {
        v78 = (__m128i *)sub_2241490((unsigned __int64 *)&v99, "'", 1u);
        v115.m128i_i64[0] = (__int64)&v116;
        if ( (__m128i *)v78->m128i_i64[0] == &v78[1] )
        {
          v116 = _mm_loadu_si128(v78 + 1);
        }
        else
        {
          v115.m128i_i64[0] = v78->m128i_i64[0];
          v116.m128i_i64[0] = v78[1].m128i_i64[0];
        }
        v115.m128i_i64[1] = v78->m128i_i64[1];
        v78->m128i_i64[0] = (__int64)v78[1].m128i_i64;
        v78->m128i_i64[1] = 0;
        v78[1].m128i_i8[0] = 0;
        v111.m128i_i64[0] = (__int64)&v115;
        v79 = v13[131];
        LOWORD(v112) = 260;
        v29 = sub_38814C0((__int64)(v13 + 1), *(_QWORD *)(v79 + 64), (__int64)&v111);
        if ( (__m128i *)v115.m128i_i64[0] != &v116 )
          j_j___libc_free_0(v115.m128i_u64[0]);
        v77 = (unsigned __int64)v99;
        if ( v99 == (__m128i *)&v101 )
          return v29;
LABEL_106:
        j_j___libc_free_0(v77);
        return v29;
      }
    }
    else
    {
      if ( !v13[118] )
      {
        if ( v13[124] )
        {
          v80 = v13[122];
          v81 = "use of undefined value '@";
          LODWORD(v99) = *(_DWORD *)(v80 + 32);
        }
        else
        {
          if ( !v13[112] )
          {
            v39 = v13[104];
            for ( i = v13 + 102; i != (_QWORD *)v39; v39 = sub_220EEE0(v39) )
            {
              v41 = *(_QWORD *)(v39 + 40);
              if ( v41 && (*(_BYTE *)(v41 + 1) == 2 || *(_DWORD *)(v41 + 12)) )
                sub_161F200(v41, a2, v15, (__int64)v14, a13);
            }
            v42 = (__int64 *)v13[25];
            for ( j = &v42[*((unsigned int *)v13 + 52)]; j != v42; ++v42 )
            {
              v45 = *v42;
              v43 = *(_QWORD *)(*v42 + 48);
              if ( v43 || *(__int16 *)(v45 + 18) < 0 )
              {
                a2 = 1;
                v43 = sub_1625790(*v42, 1);
              }
              v44 = sub_1568EC0(v43, a2);
              if ( v44 != v43 )
              {
                a2 = 1;
                sub_1625C10(v45, 1, v44);
              }
            }
            v47 = (__int64 *)v13[22];
            v67 = (__int64 *)v47[4];
            v68 = v47 + 3;
            if ( v47 + 3 != v67 )
            {
              do
              {
                v69 = v67;
                v67 = (__int64 *)v67[1];
                sub_157E300((__int64)(v69 - 7));
              }
              while ( v68 != v67 );
              v47 = (__int64 *)v13[22];
              v70 = (__int64 *)v47[4];
              if ( v47 + 3 != v70 )
              {
                do
                {
                  v71 = v70;
                  v70 = (__int64 *)v70[1];
                  v72 = (__int64)(v71 - 7);
                  a2 = (__int64)(v71 - 7);
                  sub_15E33D0((__int64)&v115, (__int64)(v71 - 7));
                  if ( v115.m128i_i8[8] )
                  {
                    a2 = v115.m128i_i64[0];
                    sub_164D160(v72, v115.m128i_i64[0], a3, a4, a5, a6, v73, v74, a9, a10);
                    sub_15E3D00(v72);
                  }
                }
                while ( v47 + 3 != v70 );
                v47 = (__int64 *)v13[22];
                i = v13 + 102;
              }
            }
            if ( *((_BYTE *)v13 + 1440) )
            {
              sub_157E370(v47);
              v47 = (__int64 *)v13[22];
            }
            sub_1569750(v47, a2);
            sub_1569DC0(v13[22], a2, v48, v49);
            v50 = (unsigned __int64 *)v13[24];
            if ( v50 )
            {
              v51 = *v50;
              *v50 = v13[125];
              v50[1] = v13[126];
              v50[2] = v13[127];
              v13[125] = 0;
              v13[126] = 0;
              v13[127] = 0;
              if ( v51 )
                j_j___libc_free_0(v51);
              v52 = v13[24];
              sub_3888040(*(_QWORD **)(v52 + 40));
              *(_QWORD *)(v52 + 40) = 0;
              *(_QWORD *)(v52 + 48) = v52 + 32;
              *(_QWORD *)(v52 + 56) = v52 + 32;
              *(_QWORD *)(v52 + 64) = 0;
              if ( v13[103] )
              {
                *(_DWORD *)(v52 + 32) = *((_DWORD *)v13 + 204);
                v53 = v13[103];
                *(_QWORD *)(v52 + 40) = v53;
                *(_QWORD *)(v52 + 48) = v13[104];
                *(_QWORD *)(v52 + 56) = v13[105];
                *(_QWORD *)(v53 + 8) = v52 + 32;
                *(_QWORD *)(v52 + 64) = v13[106];
                v13[103] = 0;
                v13[104] = (__int64)i;
                v13[105] = (__int64)i;
                v13[106] = 0;
              }
              v54 = *((_DWORD *)v13 + 184);
              if ( v54 )
              {
                v55 = (size_t **)v13[91];
                v56 = v55;
                if ( !*v55 || *v55 == (size_t *)-8LL )
                {
                  do
                  {
                    do
                    {
                      v57 = v56[1];
                      ++v56;
                    }
                    while ( v57 == (size_t *)-8LL );
                  }
                  while ( !v57 );
                }
                v88 = &v55[v54];
                if ( v56 != v88 )
                {
                  v95 = v13;
                  v58 = v56;
                  while ( 1 )
                  {
                    v59 = v95[24];
                    v60 = **v58;
                    srca = (unsigned __int8 *)(*v58 + 3);
                    v92 = (*v58)[1];
                    v86 = sub_16D19C0(v59 + 72, srca, v60);
                    v85 = (_QWORD *)(*(_QWORD *)(v59 + 72) + 8LL * v86);
                    if ( *v85 )
                    {
                      if ( *v85 != -8 )
                        goto LABEL_82;
                      --*(_DWORD *)(v59 + 88);
                    }
                    v82 = (_QWORD *)malloc(v60 + 17);
                    if ( !v82 )
                    {
                      if ( v60 == -17 )
                      {
                        v84 = malloc(1u);
                        if ( v84 )
                        {
                          v83 = (void *)(v84 + 16);
                          v82 = (_QWORD *)v84;
LABEL_126:
                          v83 = memcpy(v83, srca, v60);
                          goto LABEL_123;
                        }
                      }
                      sub_16BD1C0("Allocation failed", 1u);
                    }
                    v83 = v82 + 2;
                    if ( v60 + 1 > 1 )
                      goto LABEL_126;
LABEL_123:
                    *((_BYTE *)v83 + v60) = 0;
                    *v82 = v60;
                    v82[1] = v92;
                    *v85 = v82;
                    ++*(_DWORD *)(v59 + 84);
                    sub_16D1CD0(v59 + 72, v86);
LABEL_82:
                    v61 = v58[1];
                    ++v58;
                    if ( !v61 || v61 == (size_t *)-8LL )
                    {
                      do
                      {
                        do
                        {
                          v62 = v58[1];
                          ++v58;
                        }
                        while ( v62 == (size_t *)-8LL );
                      }
                      while ( !v62 );
                    }
                    if ( v58 == v88 )
                    {
                      v13 = v95;
                      break;
                    }
                  }
                }
              }
              for ( k = v13[98]; v97 != (_QWORD *)k; k = sub_220EEE0(k) )
              {
                v64 = *(_DWORD *)(k + 32);
                v115.m128i_i64[1] = *(_QWORD *)(k + 40);
                v65 = v13[24];
                v115.m128i_i32[0] = v64;
                sub_388FB70((_QWORD *)(v65 + 104), (unsigned int *)&v115);
              }
            }
            return 0;
          }
          v80 = v13[110];
          v81 = "use of undefined metadata '!";
          LODWORD(v99) = *(_DWORD *)(v80 + 32);
        }
        v111.m128i_i64[0] = (__int64)v81;
        v116.m128i_i16[0] = 770;
        v111.m128i_i64[1] = (__int64)v99;
        LOWORD(v112) = 2307;
        v115.m128i_i64[0] = (__int64)&v111;
        v115.m128i_i64[1] = (__int64)"'";
        return (unsigned int)sub_38814C0((__int64)(v13 + 1), *(_QWORD *)(v80 + 48), (__int64)&v115);
      }
      sub_8FD6D0((__int64)&v111, "use of undefined value '@", (_QWORD *)(v13[116] + 32));
      if ( v111.m128i_i64[1] != 0x3FFFFFFFFFFFFFFFLL )
      {
        v75 = (__m128i *)sub_2241490((unsigned __int64 *)&v111, "'", 1u);
        v115.m128i_i64[0] = (__int64)&v116;
        if ( (__m128i *)v75->m128i_i64[0] == &v75[1] )
        {
          v116 = _mm_loadu_si128(v75 + 1);
        }
        else
        {
          v115.m128i_i64[0] = v75->m128i_i64[0];
          v116.m128i_i64[0] = v75[1].m128i_i64[0];
        }
        v115.m128i_i64[1] = v75->m128i_i64[1];
        v75->m128i_i64[0] = (__int64)v75[1].m128i_i64;
        v75->m128i_i64[1] = 0;
        v75[1].m128i_i8[0] = 0;
        v99 = &v115;
        v76 = v13[116];
        LOWORD(v101) = 260;
        v29 = sub_38814C0((__int64)(v13 + 1), *(_QWORD *)(v76 + 72), (__int64)&v99);
        if ( (__m128i *)v115.m128i_i64[0] != &v116 )
          j_j___libc_free_0(v115.m128i_u64[0]);
        v77 = v111.m128i_i64[0];
        if ( (__int64 *)v111.m128i_i64[0] == &v112 )
          return v29;
        goto LABEL_106;
      }
    }
    sub_4262D8((__int64)"basic_string::append");
  }
  while ( !*(_QWORD *)(v27 + 48) )
  {
    v27 = sub_220EEE0(v27);
    if ( v13 + 96 == (__int64 *)v27 )
      goto LABEL_30;
  }
  v28 = *(_DWORD *)(v27 + 32);
  LOWORD(v112) = 2307;
  LODWORD(v99) = v28;
  v111.m128i_i64[0] = (__int64)"use of undefined type '%";
  v116.m128i_i16[0] = 770;
  v111.m128i_i64[1] = (__int64)v99;
  v115.m128i_i64[0] = (__int64)&v111;
  v115.m128i_i64[1] = (__int64)"'";
  return (unsigned int)sub_38814C0((__int64)(v13 + 1), *(_QWORD *)(v27 + 48), (__int64)&v115);
}
