// Function: sub_1A44940
// Address: 0x1a44940
//
__int64 __fastcall sub_1A44940(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 v10; // r12
  __int64 v11; // r15
  __int64 v12; // r14
  unsigned int v13; // r13d
  int v15; // r9d
  int v16; // ebx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // eax
  _BYTE *v22; // rdi
  unsigned __int64 v23; // rbx
  _BYTE **v24; // r8
  _BYTE *v25; // rax
  __int64 v26; // rdx
  _BYTE *v27; // r12
  _BYTE *v28; // rbx
  unsigned __int64 v29; // rcx
  __int64 v30; // rax
  _BYTE *v31; // rdx
  _BYTE *i; // rsi
  unsigned __int64 v33; // r13
  __int64 v34; // rbx
  __int64 v35; // r12
  unsigned __int64 v36; // rcx
  _QWORD *v37; // rdx
  _QWORD *v38; // r14
  __int64 v39; // rcx
  int v40; // r8d
  int v41; // r9d
  _BYTE *v42; // rdi
  unsigned __int64 *v43; // rdi
  __int64 *v44; // rdi
  __int64 v45; // rax
  double v46; // xmm4_8
  double v47; // xmm5_8
  __int64 v48; // rsi
  __int64 v49; // rax
  __int64 v50; // r14
  __int64 v51; // r15
  _QWORD *v52; // r8
  __int64 v53; // rax
  int v54; // r9d
  _BYTE *v55; // r8
  int v56; // r9d
  __int64 v57; // rax
  __int64 v58; // rdx
  char v59; // al
  _QWORD *v60; // rdx
  __int64 *v61; // r12
  _BYTE *v62; // rbx
  unsigned __int64 v63; // r12
  unsigned __int64 v64; // rdi
  _BYTE *v65; // rsi
  _BYTE *v66; // rbx
  unsigned __int64 v67; // rdi
  _BYTE *v68; // [rsp+30h] [rbp-680h]
  _QWORD *v69; // [rsp+30h] [rbp-680h]
  unsigned __int64 v70; // [rsp+38h] [rbp-678h]
  unsigned __int64 v71; // [rsp+40h] [rbp-670h]
  unsigned int v72; // [rsp+40h] [rbp-670h]
  __int64 v73; // [rsp+48h] [rbp-668h]
  __int64 v74; // [rsp+48h] [rbp-668h]
  unsigned __int64 v75; // [rsp+48h] [rbp-668h]
  __int64 v76; // [rsp+58h] [rbp-658h]
  __int64 v77; // [rsp+58h] [rbp-658h]
  __int64 v78; // [rsp+68h] [rbp-648h]
  __int64 n; // [rsp+70h] [rbp-640h]
  __int64 v80; // [rsp+78h] [rbp-638h]
  int v81; // [rsp+78h] [rbp-638h]
  int v82; // [rsp+78h] [rbp-638h]
  unsigned __int8 v84; // [rsp+8Fh] [rbp-621h]
  _QWORD v85[2]; // [rsp+90h] [rbp-620h] BYREF
  __m128i v86; // [rsp+A0h] [rbp-610h] BYREF
  __int64 v87; // [rsp+B0h] [rbp-600h]
  _QWORD *v88; // [rsp+C0h] [rbp-5F0h] BYREF
  __int16 v89; // [rsp+D0h] [rbp-5E0h]
  __m128 v90; // [rsp+E0h] [rbp-5D0h] BYREF
  __int64 v91; // [rsp+F0h] [rbp-5C0h]
  __int64 v92[5]; // [rsp+100h] [rbp-5B0h] BYREF
  int v93; // [rsp+128h] [rbp-588h]
  __int64 v94; // [rsp+130h] [rbp-580h]
  __int64 v95; // [rsp+138h] [rbp-578h]
  _BYTE *v96; // [rsp+150h] [rbp-560h] BYREF
  __int64 v97; // [rsp+158h] [rbp-558h]
  _BYTE s[64]; // [rsp+160h] [rbp-550h] BYREF
  _BYTE *v99; // [rsp+1A0h] [rbp-510h] BYREF
  __int64 v100; // [rsp+1A8h] [rbp-508h]
  _BYTE v101[64]; // [rsp+1B0h] [rbp-500h] BYREF
  unsigned __int64 v102[16]; // [rsp+1F0h] [rbp-4C0h] BYREF
  _BYTE *v103; // [rsp+270h] [rbp-440h] BYREF
  __int64 v104; // [rsp+278h] [rbp-438h]
  _BYTE v105[1072]; // [rsp+280h] [rbp-430h] BYREF

  v10 = a2;
  if ( *(_DWORD *)(a1 + 496) && !(unsigned __int8)sub_1A3F5B0(a1, a2) )
    return 0;
  v11 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
    return 0;
  v12 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v12 + 16) )
    return 0;
  v13 = *(_DWORD *)(v12 + 36);
  if ( !v13 )
    return 0;
  v84 = sub_14C3AC0(v13);
  if ( !v84 )
    return 0;
  v16 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v76 = *(_QWORD *)(v11 + 32);
  if ( *(char *)(a2 + 23) >= 0 )
    goto LABEL_17;
  v17 = sub_1648A40(a2);
  v80 = v18 + v17;
  if ( *(char *)(a2 + 23) >= 0 )
  {
    if ( (unsigned int)(v80 >> 4) )
LABEL_107:
      BUG();
LABEL_17:
    v21 = 0;
    goto LABEL_18;
  }
  if ( !(unsigned int)((v80 - sub_1648A40(a2)) >> 4) )
    goto LABEL_17;
  if ( *(char *)(a2 + 23) >= 0 )
    goto LABEL_107;
  v81 = *(_DWORD *)(sub_1648A40(a2) + 8);
  if ( *(char *)(a2 + 23) >= 0 )
    BUG();
  v19 = sub_1648A40(a2);
  v21 = *(_DWORD *)(v19 + v20 - 4) - v81;
LABEL_18:
  v22 = s;
  v23 = (unsigned int)(v16 - 1 - v21);
  v24 = &v96;
  v82 = v23;
  v96 = s;
  v97 = 0x800000000LL;
  if ( (unsigned int)v23 > 8 )
  {
    sub_16CD150((__int64)&v96, s, v23, 8, (int)&v96, v15);
    v22 = v96;
  }
  LODWORD(v97) = v23;
  n = 8 * v23;
  if ( 8 * v23 )
    memset(v22, 0, n);
  memset(v102, 0, sizeof(v102));
  v102[5] = (unsigned __int64)&v102[7];
  v102[6] = 0x800000000LL;
  v104 = 0x800000000LL;
  v25 = v105;
  v103 = v105;
  if ( v23 > 8 )
  {
    sub_1A3EFF0((__int64)&v103, v23);
    v25 = v103;
  }
  LODWORD(v104) = v23;
  v26 = (__int64)&v25[128 * v23];
  v70 = v23 << 7;
  if ( (_BYTE *)v26 != v25 )
  {
    v27 = v25;
    v71 = v23;
    v28 = &v25[128 * v23];
    do
    {
      if ( v27 )
      {
        *(_QWORD *)v27 = v102[0];
        *((_QWORD *)v27 + 1) = v102[1];
        *((_QWORD *)v27 + 2) = v102[2];
        *((_QWORD *)v27 + 3) = v102[3];
        v29 = v102[4];
        *((_DWORD *)v27 + 12) = 0;
        *((_QWORD *)v27 + 4) = v29;
        *((_QWORD *)v27 + 5) = v27 + 56;
        *((_DWORD *)v27 + 13) = 8;
        if ( LODWORD(v102[6]) )
          sub_1A3EC80((__int64)(v27 + 40), (__int64)&v102[5], v26, (__int64)(v27 + 56), (int)v24, v15);
        *((_DWORD *)v27 + 30) = v102[15];
      }
      v27 += 128;
    }
    while ( v28 != v27 );
    v10 = a2;
    v23 = v71;
  }
  if ( (unsigned __int64 *)v102[5] != &v102[7] )
    _libc_free(v102[5]);
  v30 = (unsigned int)v104;
  if ( v23 < (unsigned int)v104 )
  {
    v65 = &v103[v70];
    if ( &v103[128 * (unsigned __int64)(unsigned int)v104] != &v103[v70] )
    {
      v75 = v23;
      v66 = &v103[128 * (unsigned __int64)(unsigned int)v104];
      do
      {
        v66 -= 128;
        v67 = *((_QWORD *)v66 + 5);
        if ( (_BYTE *)v67 != v66 + 56 )
          _libc_free(v67);
      }
      while ( v65 != v66 );
      v23 = v75;
    }
    goto LABEL_42;
  }
  if ( v23 > (unsigned int)v104 )
  {
    if ( v23 > HIDWORD(v104) )
    {
      sub_1A3EFF0((__int64)&v103, v23);
      v30 = (unsigned int)v104;
    }
    v31 = &v103[128 * v30];
    for ( i = &v103[v70]; i != v31; v31 += 128 )
    {
      if ( v31 )
      {
        memset(v31, 0, 0x80u);
        *((_DWORD *)v31 + 13) = 8;
        *((_QWORD *)v31 + 5) = v31 + 56;
      }
    }
LABEL_42:
    LODWORD(v104) = v82;
  }
  if ( v82 )
  {
    v72 = v13;
    v33 = v23;
    v34 = v10;
    v35 = 0;
    v73 = v12;
    do
    {
      v36 = *(_QWORD *)(v34 + 24 * (v35 - (*(_DWORD *)(v34 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(*(_QWORD *)v36 + 8LL) == 16 )
      {
        sub_1A41500((__int64)v102, (_QWORD *)a1, v34, v36, (__int64)v24, v15);
        v37 = &v103[128 * v35];
        *v37 = v102[0];
        v38 = v37;
        v37[1] = v102[1];
        v37[2] = v102[2];
        v37[3] = v102[3];
        v39 = v102[4];
        v37[4] = v102[4];
        sub_1A3ED60((__int64)(v37 + 5), (char **)&v102[5], (__int64)v37, v39, v40, v41);
        *((_DWORD *)v38 + 30) = v102[15];
        if ( (unsigned __int64 *)v102[5] != &v102[7] )
          _libc_free(v102[5]);
      }
      else
      {
        *(_QWORD *)&v96[8 * v35] = v36;
      }
      ++v35;
    }
    while ( v33 != v35 );
    v10 = v34;
    v12 = v73;
    v23 = v33;
    v13 = v72;
  }
  v42 = v101;
  v99 = v101;
  v78 = (unsigned int)v76;
  v100 = 0x800000000LL;
  if ( (unsigned int)v76 > 8 )
  {
    sub_16CD150((__int64)&v99, v101, (unsigned int)v76, 8, (int)v24, v15);
    v42 = v99;
  }
  LODWORD(v100) = v76;
  if ( 8LL * (unsigned int)v76 )
    memset(v42, 0, 8LL * (unsigned int)v76);
  v43 = &v102[2];
  v102[0] = (unsigned __int64)&v102[2];
  v102[1] = 0x800000000LL;
  if ( v23 > 8 )
  {
    sub_16CD150((__int64)v102, &v102[2], v23, 8, (int)v24, v15);
    v43 = (unsigned __int64 *)v102[0];
  }
  LODWORD(v102[1]) = v82;
  if ( n )
    memset(v43, 0, n);
  v44 = *(__int64 **)(v12 + 40);
  if ( *(_BYTE *)(v11 + 8) == 16 )
    v11 = **(_QWORD **)(v11 + 16);
  v92[0] = v11;
  v74 = sub_15E26F0(v44, v13, v92, 1);
  v45 = sub_16498A0(v10);
  v48 = *(_QWORD *)(v10 + 48);
  v92[0] = 0;
  v92[3] = v45;
  v49 = *(_QWORD *)(v10 + 40);
  v92[4] = 0;
  v92[1] = v49;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v92[2] = v10 + 24;
  v90.m128_u64[0] = v48;
  if ( v48 )
  {
    sub_1623A60((__int64)&v90, v48, 2);
    if ( v92[0] )
      sub_161E7C0((__int64)v92, v92[0]);
    v92[0] = v90.m128_u64[0];
    if ( v90.m128_u64[0] )
      sub_1623210((__int64)&v90, (unsigned __int8 *)v90.m128_u64[0], (__int64)v92);
  }
  if ( (_DWORD)v76 )
  {
    v77 = v10;
    v50 = 0;
    do
    {
      v51 = 0;
      LODWORD(v102[1]) = 0;
      if ( v82 )
      {
        do
        {
          while ( !sub_14C3B20(v13, v51) )
          {
            v55 = sub_1A3F820((__int64 *)&v103[128 * v51], v50);
            v57 = LODWORD(v102[1]);
            if ( LODWORD(v102[1]) >= HIDWORD(v102[1]) )
            {
              v68 = v55;
              sub_16CD150((__int64)v102, &v102[2], 0, 8, (int)v55, v56);
              v57 = LODWORD(v102[1]);
              v55 = v68;
            }
            ++v51;
            *(_QWORD *)(v102[0] + 8 * v57) = v55;
            ++LODWORD(v102[1]);
            if ( v23 == v51 )
              goto LABEL_77;
          }
          v52 = &v96[8 * v51];
          v53 = LODWORD(v102[1]);
          if ( LODWORD(v102[1]) >= HIDWORD(v102[1]) )
          {
            v69 = &v96[8 * v51];
            sub_16CD150((__int64)v102, &v102[2], 0, 8, (int)v52, v54);
            v53 = LODWORD(v102[1]);
            v52 = v69;
          }
          ++v51;
          *(_QWORD *)(v102[0] + 8 * v53) = *v52;
          ++LODWORD(v102[1]);
        }
        while ( v23 != v51 );
      }
LABEL_77:
      LODWORD(v88) = v50;
      v89 = 265;
      v85[0] = sub_1649960(v77);
      v85[1] = v58;
      v86.m128i_i64[0] = (__int64)v85;
      v86.m128i_i64[1] = (__int64)".i";
      v59 = v89;
      LOWORD(v87) = 773;
      if ( (_BYTE)v89 )
      {
        if ( (_BYTE)v89 == 1 )
        {
          a3 = (__m128)_mm_loadu_si128(&v86);
          v90 = a3;
          v91 = v87;
        }
        else
        {
          v60 = v88;
          if ( HIBYTE(v89) != 1 )
          {
            v60 = &v88;
            v59 = 2;
          }
          v90.m128_u64[1] = (unsigned __int64)v60;
          LOBYTE(v91) = 2;
          v90.m128_u64[0] = (unsigned __int64)&v86;
          BYTE1(v91) = v59;
        }
      }
      else
      {
        LOWORD(v91) = 256;
      }
      v61 = (__int64 *)&v99[8 * v50++];
      *v61 = sub_1285290(v92, *(_QWORD *)(v74 + 24), v74, v102[0], LODWORD(v102[1]), (__int64)&v90, 0);
    }
    while ( v78 != v50 );
    v10 = v77;
  }
  sub_1A41120(a1, v10, &v99, a3, a4, a5, a6, v46, v47, a9, a10);
  if ( v92[0] )
    sub_161E7C0((__int64)v92, v92[0]);
  if ( (unsigned __int64 *)v102[0] != &v102[2] )
    _libc_free(v102[0]);
  if ( v99 != v101 )
    _libc_free((unsigned __int64)v99);
  v62 = v103;
  v63 = (unsigned __int64)&v103[128 * (unsigned __int64)(unsigned int)v104];
  if ( v103 != (_BYTE *)v63 )
  {
    do
    {
      v63 -= 128LL;
      v64 = *(_QWORD *)(v63 + 40);
      if ( v64 != v63 + 56 )
        _libc_free(v64);
    }
    while ( v62 != (_BYTE *)v63 );
    v63 = (unsigned __int64)v103;
  }
  if ( (_BYTE *)v63 != v105 )
    _libc_free(v63);
  if ( v96 != s )
    _libc_free((unsigned __int64)v96);
  return v84;
}
