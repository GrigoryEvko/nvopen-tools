// Function: sub_23F9D10
// Address: 0x23f9d10
//
__int64 __fastcall sub_23F9D10(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  char *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  char *v10; // r9
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 *v14; // r15
  __int64 *v15; // r12
  __int64 v16; // r14
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 v30; // rbx
  __int64 v31; // r13
  unsigned __int64 v32; // r14
  __int64 v33; // r13
  __int64 *v34; // rbx
  unsigned __int64 v35; // r12
  unsigned __int64 v36; // r15
  __m128i *v37; // r14
  __m128i *v38; // r13
  unsigned __int64 v39; // rsi
  const __m128i *v40; // r11
  __int8 *v41; // rax
  unsigned __int64 v42; // rdx
  __m128i v43; // xmm0
  __m128i *v44; // rsi
  char *v45; // r14
  char *v46; // r13
  unsigned __int64 v47; // rsi
  char *v48; // r11
  char *i; // rdi
  unsigned __int64 v50; // rsi
  unsigned __int64 v51; // rdx
  char *v52; // rax
  __int64 v53; // rdx
  char *v54; // r13
  unsigned __int64 v55; // rsi
  char *v56; // r11
  char *j; // rdi
  unsigned __int64 v58; // rsi
  unsigned __int64 v59; // rdx
  char *v60; // rax
  __int64 v61; // rdx
  char *v62; // rsi
  char *v63; // rax
  __int64 v64; // rdi
  char *v65; // rdx
  __int64 *v66; // rax
  char *v67; // rdx
  char *v68; // rax
  __int64 v69; // rdi
  __int64 v70; // r9
  char *v71; // rdi
  __int64 *v72; // rax
  __int64 v73; // rcx
  char *v74; // rdx
  char *v75; // rdi
  signed __int64 v76; // rax
  _BYTE *v77; // r15
  __int64 v78; // [rsp+8h] [rbp-308h]
  __int64 v79; // [rsp+10h] [rbp-300h]
  __int64 v80; // [rsp+18h] [rbp-2F8h]
  __int64 v81; // [rsp+28h] [rbp-2E8h]
  __int64 v84; // [rsp+40h] [rbp-2D0h]
  __int64 v85; // [rsp+48h] [rbp-2C8h]
  __int64 v86; // [rsp+48h] [rbp-2C8h]
  char *v87; // [rsp+48h] [rbp-2C8h]
  __int64 v88; // [rsp+50h] [rbp-2C0h]
  __int64 v89; // [rsp+58h] [rbp-2B8h]
  __int64 *v90; // [rsp+68h] [rbp-2A8h]
  __int64 *v91; // [rsp+78h] [rbp-298h]
  __int64 v92; // [rsp+80h] [rbp-290h]
  __int64 *v93; // [rsp+80h] [rbp-290h]
  __int64 *v94; // [rsp+88h] [rbp-288h]
  __int64 *v95; // [rsp+98h] [rbp-278h] BYREF
  _QWORD v96[2]; // [rsp+A0h] [rbp-270h] BYREF
  __m128i v97; // [rsp+B0h] [rbp-260h] BYREF
  __int64 v98; // [rsp+C0h] [rbp-250h]
  __int64 v99; // [rsp+D0h] [rbp-240h] BYREF
  __int64 v100; // [rsp+D8h] [rbp-238h]
  __int64 v101; // [rsp+E0h] [rbp-230h]
  unsigned int v102; // [rsp+E8h] [rbp-228h]
  __int64 *v103; // [rsp+F0h] [rbp-220h]
  __int64 v104; // [rsp+F8h] [rbp-218h]
  unsigned __int64 v105; // [rsp+100h] [rbp-210h] BYREF
  _QWORD *v106; // [rsp+108h] [rbp-208h]
  __int64 v107; // [rsp+110h] [rbp-200h] BYREF
  _QWORD v108[9]; // [rsp+150h] [rbp-1C0h] BYREF
  __m128i *v109; // [rsp+198h] [rbp-178h]
  __m128i *v110; // [rsp+1A0h] [rbp-170h]
  __int64 v111; // [rsp+1A8h] [rbp-168h]
  char *v112; // [rsp+1B0h] [rbp-160h]
  char *v113; // [rsp+1B8h] [rbp-158h]
  __int64 v114; // [rsp+1C0h] [rbp-150h]
  __int64 v115; // [rsp+1C8h] [rbp-148h]
  __int64 v116; // [rsp+1D0h] [rbp-140h]
  __int64 v117; // [rsp+1D8h] [rbp-138h]
  __int64 v118; // [rsp+1E0h] [rbp-130h]
  char *v119; // [rsp+1E8h] [rbp-128h]
  char *v120; // [rsp+1F0h] [rbp-120h]
  __int64 v121; // [rsp+1F8h] [rbp-118h]
  _QWORD v122[5]; // [rsp+200h] [rbp-110h] BYREF
  char v123; // [rsp+228h] [rbp-E8h] BYREF
  _QWORD *v124; // [rsp+248h] [rbp-C8h]
  __int64 v125; // [rsp+250h] [rbp-C0h]
  _QWORD v126[16]; // [rsp+258h] [rbp-B8h] BYREF
  char v127; // [rsp+2D8h] [rbp-38h]

  v108[5] = 0x800000000LL;
  v108[8] = 0x800000000LL;
  v6 = (__int64)v108;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = (__int64 *)&v105;
  v104 = 0;
  memset(v108, 0, 40);
  v108[6] = 0;
  v108[7] = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v122[3] = &v123;
  v122[4] = 0x400000000LL;
  v124 = v126;
  v126[15] = v122;
  memset(&v126[2], 0, 104);
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  memset(v122, 0, 24);
  v125 = 0;
  v126[0] = 0;
  v126[1] = 1;
  v127 = 0;
  sub_ED5CA0(&v105, (__int64)v108, a1, a3, 1, a6);
  if ( (v105 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v105 = v105 & 0xFFFFFFFFFFFFFFFELL | 1;
    sub_C63C30(&v105, (__int64)v108);
  }
  v92 = a1 + 24;
  if ( *(_QWORD *)(a1 + 32) != a1 + 24 )
  {
    v11 = *(_QWORD *)(a1 + 32);
    do
    {
      v12 = 0;
      if ( v11 )
        v12 = v11 - 56;
      v13 = v12;
      if ( !sub_B2FC80(v12) )
      {
        v6 = v13;
        sub_B2EE70((__int64)&v105, v13, 0);
        if ( (_BYTE)v107 )
        {
          v6 = (__int64)&unk_4F8D9A8;
          v90 = (__int64 *)(sub_BC1CD0(a2, &unk_4F8D9A8, v13) + 8);
          if ( sub_FDC4B0((__int64)v90) )
          {
            v6 = (__int64)&unk_4F89C30;
            v25 = sub_BC1CD0(a2, &unk_4F89C30, v13);
            v26 = *(_QWORD *)(v13 + 80);
            v91 = (__int64 *)(v25 + 8);
            v89 = v13 + 72;
            if ( v26 != v13 + 72 )
            {
              v88 = v13;
              v78 = v11;
              do
              {
                while ( 1 )
                {
                  v27 = v26 - 24;
                  if ( !v26 )
                    v27 = 0;
                  v6 = v27;
                  v28 = sub_FDD2C0(v90, v27, 0);
                  v96[1] = v7;
                  v96[0] = v28;
                  if ( (_BYTE)v7 )
                  {
                    v29 = *(_QWORD *)(v27 + 56);
                    v30 = v27 + 48;
                    if ( v29 != v30 )
                      break;
                  }
                  v26 = *(_QWORD *)(v26 + 8);
                  if ( v89 == v26 )
                    goto LABEL_41;
                }
                v81 = v26;
                do
                {
                  if ( !v29 )
                    BUG();
                  if ( (unsigned __int8)(*(_BYTE *)(v29 - 24) - 34) <= 0x33u )
                  {
                    v6 = 0x8000000000041LL;
                    if ( _bittest64(&v6, (unsigned int)*(unsigned __int8 *)(v29 - 24) - 34) )
                    {
                      if ( sub_B491E0(v29 - 24) )
                      {
                        sub_ED2710((__int64)&v105, v29 - 24, 0, 8u, &v95, 0);
                        v9 = v105;
                        v33 = 16LL * (unsigned int)v106;
                        v94 = (__int64 *)(v105 + v33);
                        if ( v105 != v105 + v33 )
                        {
                          v79 = v29;
                          v80 = v30;
                          v34 = (__int64 *)v105;
                          do
                          {
                            v35 = v34[1];
                            v36 = *v34;
                            if ( !v127 )
                            {
                              v37 = v110;
                              v38 = v109;
                              if ( v110 != v109 )
                              {
                                v85 = (char *)v110 - (char *)v109;
                                _BitScanReverse64(&v39, 0xAAAAAAAAAAAAAAABLL * (((char *)v110 - (char *)v109) >> 3));
                                sub_ED4580((__int64)v109, v110, 2LL * (int)(63 - (v39 ^ 0x3F)), v8, v9, (__int64)v10);
                                if ( v85 <= 384 )
                                {
                                  sub_23F9360((__int64)v38, (unsigned __int64 *)v37);
                                }
                                else
                                {
                                  sub_23F9360((__int64)v38, (unsigned __int64 *)&v38[24]);
                                  for ( ;
                                        v37 != v40;
                                        *(__m128i *)((char *)v44 + 8) = _mm_loadu_si128((const __m128i *)&v97.m128i_u64[1]) )
                                  {
                                    v97 = _mm_loadu_si128(v40);
                                    v98 = v40[1].m128i_i64[0];
                                    v8 = v40->m128i_i64[0];
                                    v41 = &v40[-2].m128i_i8[8];
                                    v42 = v40[-2].m128i_u64[1];
                                    if ( v40->m128i_i64[0] >= v42 )
                                    {
                                      v44 = (__m128i *)v40;
                                    }
                                    else
                                    {
                                      do
                                      {
                                        v43 = _mm_loadu_si128((const __m128i *)(v41 + 8));
                                        *((_QWORD *)v41 + 3) = v42;
                                        v44 = (__m128i *)v41;
                                        v41 -= 24;
                                        *(__m128i *)(v41 + 56) = v43;
                                        v42 = *(_QWORD *)v41;
                                      }
                                      while ( (unsigned __int64)v8 < *(_QWORD *)v41 );
                                    }
                                    v44->m128i_i64[0] = v8;
                                    v40 = (const __m128i *)((char *)v40 + 24);
                                  }
                                }
                              }
                              v45 = v113;
                              v46 = v112;
                              if ( v113 != v112 )
                              {
                                v86 = v113 - v112;
                                _BitScanReverse64(&v47, (v113 - v112) >> 4);
                                sub_ED48E0(v112, v113, 2LL * (int)(63 - (v47 ^ 0x3F)));
                                if ( v86 <= 256 )
                                {
                                  sub_23F92B0(v46, v45);
                                }
                                else
                                {
                                  sub_23F92B0(v46, v46 + 256);
                                  for ( i = v48; v45 != i; *(_QWORD *)(v8 + 8) = v9 )
                                  {
                                    v50 = *(_QWORD *)i;
                                    v51 = *((_QWORD *)i - 2);
                                    v8 = (__int64)i;
                                    v52 = i - 16;
                                    v9 = *((_QWORD *)i + 1);
                                    if ( *(_QWORD *)i < v51 )
                                    {
                                      do
                                      {
                                        *((_QWORD *)v52 + 2) = v51;
                                        v53 = *((_QWORD *)v52 + 1);
                                        v8 = (__int64)v52;
                                        v52 -= 16;
                                        *((_QWORD *)v52 + 5) = v53;
                                        v51 = *(_QWORD *)v52;
                                      }
                                      while ( v50 < *(_QWORD *)v52 );
                                    }
                                    i += 16;
                                    *(_QWORD *)v8 = v50;
                                  }
                                }
                              }
                              v54 = v120;
                              if ( v120 != v119 )
                              {
                                v87 = v119;
                                v84 = v120 - v119;
                                _BitScanReverse64(&v55, (v120 - v119) >> 4);
                                sub_ED4B00(v119, v120, 2LL * (int)(63 - (v55 ^ 0x3F)));
                                if ( v84 <= 256 )
                                {
                                  sub_23F9200(v87, v54);
                                }
                                else
                                {
                                  sub_23F9200(v87, v87 + 256);
                                  for ( j = v56; v54 != j; *(_QWORD *)(v8 + 8) = v9 )
                                  {
                                    v58 = *(_QWORD *)j;
                                    v59 = *((_QWORD *)j - 2);
                                    v8 = (__int64)j;
                                    v60 = j - 16;
                                    v9 = *((_QWORD *)j + 1);
                                    if ( *(_QWORD *)j < v59 )
                                    {
                                      do
                                      {
                                        *((_QWORD *)v60 + 2) = v59;
                                        v61 = *((_QWORD *)v60 + 1);
                                        v8 = (__int64)v60;
                                        v60 -= 16;
                                        *((_QWORD *)v60 + 5) = v61;
                                        v59 = *(_QWORD *)v60;
                                      }
                                      while ( v58 < *(_QWORD *)v60 );
                                    }
                                    j += 16;
                                    *(_QWORD *)v8 = v58;
                                  }
                                }
                                v62 = v120;
                                v63 = v119;
                                if ( v120 != v119 )
                                {
                                  while ( 1 )
                                  {
                                    v63 += 16;
                                    if ( v120 == v63 )
                                      goto LABEL_86;
                                    v64 = *((_QWORD *)v63 - 2);
                                    v65 = v63 - 16;
                                    if ( v64 == *(_QWORD *)v63 )
                                    {
                                      v8 = *((_QWORD *)v63 + 1);
                                      if ( *((_QWORD *)v63 - 1) == v8 )
                                        break;
                                    }
                                  }
                                  if ( v120 != v65 )
                                  {
                                    v66 = (__int64 *)(v63 + 16);
                                    if ( v120 != (char *)v66 )
                                    {
                                      while ( 1 )
                                      {
                                        v8 = *v66;
                                        if ( *v66 == v64 && *((_QWORD *)v65 + 1) == v66[1] )
                                        {
                                          v66 += 2;
                                          if ( v62 == (char *)v66 )
                                            break;
                                        }
                                        else
                                        {
                                          *((_QWORD *)v65 + 2) = v8;
                                          v8 = v66[1];
                                          v66 += 2;
                                          v65 += 16;
                                          *((_QWORD *)v65 + 1) = v8;
                                          if ( v62 == (char *)v66 )
                                            break;
                                        }
                                        v64 = *(_QWORD *)v65;
                                      }
                                    }
                                    v67 = v65 + 16;
                                    if ( v62 != v67 )
                                    {
                                      v68 = v120;
                                      v69 = v120 - v62;
                                      if ( v62 == v120 )
                                      {
                                        v74 = &v67[v69];
LABEL_85:
                                        v120 = v74;
                                        goto LABEL_86;
                                      }
                                      v70 = v69 >> 4;
                                      if ( v69 > 0 )
                                      {
                                        v71 = v67;
                                        v72 = (__int64 *)v62;
                                        do
                                        {
                                          v73 = *v72;
                                          v71 += 16;
                                          v72 += 2;
                                          *((_QWORD *)v71 - 2) = v73;
                                          v8 = *(v72 - 1);
                                          *((_QWORD *)v71 - 1) = v8;
                                          --v70;
                                        }
                                        while ( v70 );
                                        v68 = v120;
                                        v69 = v120 - v62;
                                      }
                                      v74 = &v67[v69];
                                      if ( v68 != v74 )
                                        goto LABEL_85;
                                    }
                                  }
                                }
                              }
LABEL_86:
                              v127 = 1;
                            }
                            v10 = v113;
                            v75 = v112;
                            v7 = (char *)(v113 - v112);
                            v76 = (v113 - v112) >> 4;
                            if ( v113 - v112 > 0 )
                            {
                              do
                              {
                                while ( 1 )
                                {
                                  v8 = v76 >> 1;
                                  v7 = &v75[16 * (v76 >> 1)];
                                  if ( v36 <= *(_QWORD *)v7 )
                                    break;
                                  v75 = v7 + 16;
                                  v76 = v76 - v8 - 1;
                                  if ( v76 <= 0 )
                                    goto LABEL_92;
                                }
                                v76 >>= 1;
                              }
                              while ( v8 > 0 );
                            }
LABEL_92:
                            if ( v75 != v113 && v36 == *(_QWORD *)v75 )
                            {
                              v77 = (_BYTE *)*((_QWORD *)v75 + 1);
                              if ( v35 )
                              {
                                if ( v77 && (unsigned __int8)sub_DF9C30(v91, v77) && (v77[33] & 3) != 1 )
                                  sub_23F9970((__int64)&v99, v88, (__int64)v77, v35);
                              }
                            }
                            v34 += 2;
                          }
                          while ( v94 != v34 );
                          v30 = v80;
                          v29 = v79;
                          v94 = (__int64 *)v105;
                        }
                        v6 = (__int64)v94;
                        if ( v94 != &v107 )
                          _libc_free((unsigned __int64)v94);
                      }
                      else
                      {
                        v31 = *(_QWORD *)(v29 - 56);
                        if ( v31 )
                        {
                          if ( !*(_BYTE *)v31 )
                          {
                            v32 = v96[0];
                            if ( *(_QWORD *)(v31 + 24) == *(_QWORD *)(v29 + 56) )
                            {
                              if ( v96[0] )
                              {
                                v6 = *(_QWORD *)(v29 - 56);
                                if ( (unsigned __int8)sub_DF9C30(v91, (_BYTE *)v6) )
                                {
                                  if ( (*(_BYTE *)(v31 + 33) & 3) != 1 )
                                  {
                                    v6 = v88;
                                    sub_23F9970((__int64)&v99, v88, v31, v32);
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                  v29 = *(_QWORD *)(v29 + 8);
                }
                while ( v30 != v29 );
                v26 = *(_QWORD *)(v81 + 8);
              }
              while ( v89 != v26 );
LABEL_41:
              v11 = v78;
            }
          }
        }
      }
      v11 = *(_QWORD *)(v11 + 8);
    }
    while ( v92 != v11 );
  }
  if ( (_DWORD)v104 )
  {
    v14 = v103;
    v97 = 0u;
    v15 = *(__int64 **)a1;
    v98 = 0;
    v93 = &v103[3 * (unsigned int)v104];
    v95 = v15;
    do
    {
      while ( 1 )
      {
        v16 = v14[1];
        v17 = v14[2];
        v105 = (unsigned __int64)sub_B98A20(*v14, v6);
        v106 = sub_B98A20(v16, v6);
        v18 = sub_BCB2E0(v15);
        v19 = sub_ACD640(v18, v17, 0);
        v107 = sub_B8C140((__int64)&v95, v19, v20, v21);
        v22 = sub_B9C770(v15, (__int64 *)&v105, (__int64 *)3, 0, 1);
        v6 = v97.m128i_i64[1];
        v96[0] = v22;
        if ( v97.m128i_i64[1] != v98 )
          break;
        v14 += 3;
        sub_914280((__int64)&v97, (_BYTE *)v97.m128i_i64[1], v96);
        if ( v93 == v14 )
          goto LABEL_18;
      }
      if ( v97.m128i_i64[1] )
      {
        *(_QWORD *)v97.m128i_i64[1] = v22;
        v6 = v97.m128i_i64[1];
      }
      v6 += 8;
      v14 += 3;
      v97.m128i_i64[1] = v6;
    }
    while ( v93 != v14 );
LABEL_18:
    v23 = sub_B9C770(v15, (__int64 *)v97.m128i_i64[0], (__int64 *)((v97.m128i_i64[1] - v97.m128i_i64[0]) >> 3), 1, 1);
    v6 = 5;
    sub_BA92F0((__int64 **)a1, 5u, "CG Profile", 0xAu, v23);
    if ( v97.m128i_i64[0] )
    {
      v6 = v98 - v97.m128i_i64[0];
      j_j___libc_free_0(v97.m128i_u64[0]);
    }
  }
  sub_23F9450((__int64)v108, v6, (__int64)v7, v8, v9, (unsigned __int64)v10);
  if ( v103 != (__int64 *)&v105 )
    _libc_free((unsigned __int64)v103);
  return sub_C7D6A0(v100, 24LL * v102, 8);
}
