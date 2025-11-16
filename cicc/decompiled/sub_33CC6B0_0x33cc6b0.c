// Function: sub_33CC6B0
// Address: 0x33cc6b0
//
void __fastcall sub_33CC6B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // rsi
  unsigned __int64 v7; // rcx
  unsigned __int64 v9; // r12
  __int64 v10; // rax
  _QWORD *v11; // r13
  _QWORD *v12; // r14
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  __int64 *v17; // r15
  __int64 *v18; // r13
  __int64 i; // rax
  __int64 v20; // rdi
  unsigned int v21; // ecx
  __int64 v22; // rsi
  __int64 *v23; // r13
  unsigned __int64 v24; // r14
  __int64 v25; // rsi
  __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  __int64 v28; // r13
  unsigned __int64 v29; // r8
  __int64 v30; // r13
  __int64 v31; // r12
  _QWORD *v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  __int64 v35; // rax
  _QWORD *v36; // r12
  _QWORD *v37; // r13
  unsigned __int64 v38; // rdi
  __int64 v39; // rsi
  unsigned __int64 v40; // rdi
  __int64 *v41; // r14
  __int64 *v42; // r12
  __int64 j; // rax
  __int64 v44; // rdi
  unsigned int v45; // ecx
  __int64 *v46; // r12
  unsigned __int64 v47; // r13
  __int64 v48; // rdi
  unsigned __int64 v49; // rdi
  __int64 *v50; // r14
  __int64 v51; // rax
  __int64 *v52; // r12
  __int64 *k; // rax
  __int64 v54; // rdi
  unsigned int v55; // ecx
  __int64 v56; // rsi
  __int64 *v57; // r12
  unsigned __int64 v58; // r13
  __int64 v59; // rsi
  __int64 v60; // rdi
  unsigned __int64 v61; // rdi
  _QWORD *v62; // rax
  __int64 v63; // rsi
  __int64 *v64; // r14
  __int64 *v65; // r12
  __int64 m; // rax
  __int64 v67; // rdi
  unsigned int v68; // ecx
  __int64 *v69; // r12
  unsigned __int64 v70; // r13
  __int64 v71; // rdi
  unsigned __int64 v72; // rdi

  sub_33CC630(a1, a2, a3, a4, a5, a6);
  v9 = *(_QWORD *)(a1 + 720);
  *(_DWORD *)(a1 + 648) = 0;
  if ( v9 )
  {
    v10 = *(unsigned int *)(v9 + 712);
    if ( (_DWORD)v10 )
    {
      v11 = *(_QWORD **)(v9 + 696);
      v12 = &v11[5 * v10];
      do
      {
        if ( *v11 != -8192 && *v11 != -4096 )
        {
          v13 = v11[1];
          if ( (_QWORD *)v13 != v11 + 3 )
            _libc_free(v13);
        }
        v11 += 5;
      }
      while ( v12 != v11 );
      v10 = *(unsigned int *)(v9 + 712);
    }
    sub_C7D6A0(*(_QWORD *)(v9 + 696), 40 * v10, 8);
    v14 = *(_QWORD *)(v9 + 640);
    if ( v14 != v9 + 656 )
      _libc_free(v14);
    v15 = *(_QWORD *)(v9 + 368);
    if ( v15 != v9 + 384 )
      _libc_free(v15);
    v16 = *(_QWORD *)(v9 + 96);
    if ( v16 != v9 + 112 )
      _libc_free(v16);
    v17 = *(__int64 **)(v9 + 16);
    v18 = &v17[*(unsigned int *)(v9 + 24)];
    if ( v17 != v18 )
    {
      for ( i = *(_QWORD *)(v9 + 16); ; i = *(_QWORD *)(v9 + 16) )
      {
        v20 = *v17;
        v21 = (unsigned int)(((__int64)v17 - i) >> 3) >> 7;
        v22 = 4096LL << v21;
        if ( v21 >= 0x1E )
          v22 = 0x40000000000LL;
        ++v17;
        sub_C7D6A0(v20, v22, 16);
        if ( v18 == v17 )
          break;
      }
    }
    v23 = *(__int64 **)(v9 + 64);
    v24 = (unsigned __int64)&v23[2 * *(unsigned int *)(v9 + 72)];
    if ( v23 != (__int64 *)v24 )
    {
      do
      {
        v25 = v23[1];
        v26 = *v23;
        v23 += 2;
        sub_C7D6A0(v26, v25, 16);
      }
      while ( (__int64 *)v24 != v23 );
      v24 = *(_QWORD *)(v9 + 64);
    }
    if ( v24 != v9 + 80 )
      _libc_free(v24);
    v27 = *(_QWORD *)(v9 + 16);
    if ( v27 != v9 + 32 )
      _libc_free(v27);
    j_j___libc_free_0(v9);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1000), 16LL * *(unsigned int *)(a1 + 1016), 8);
  sub_33C9130(*(_QWORD **)(a1 + 960));
  if ( *(_DWORD *)(a1 + 932) )
  {
    v28 = *(unsigned int *)(a1 + 928);
    v29 = *(_QWORD *)(a1 + 920);
    if ( (_DWORD)v28 )
    {
      v30 = 8 * v28;
      v31 = 0;
      do
      {
        v32 = *(_QWORD **)(v29 + v31);
        if ( v32 != (_QWORD *)-8LL && v32 )
        {
          sub_C7D6A0((__int64)v32, *v32 + 17LL, 8);
          v29 = *(_QWORD *)(a1 + 920);
        }
        v31 += 8;
      }
      while ( v30 != v31 );
    }
  }
  else
  {
    v29 = *(_QWORD *)(a1 + 920);
  }
  _libc_free(v29);
  sub_33C8980(*(_QWORD *)(a1 + 888));
  v33 = *(_QWORD *)(a1 + 848);
  if ( v33 )
    j_j___libc_free_0(v33);
  v34 = *(_QWORD *)(a1 + 824);
  if ( v34 )
    j_j___libc_free_0(v34);
  sub_33C8D90(*(_QWORD **)(a1 + 792));
  v35 = *(unsigned int *)(a1 + 752);
  if ( (_DWORD)v35 )
  {
    v36 = *(_QWORD **)(a1 + 736);
    v37 = &v36[10 * v35];
    do
    {
      if ( *v36 != -4096 && *v36 != -8192 )
      {
        v38 = v36[1];
        if ( (_QWORD *)v38 != v36 + 3 )
          _libc_free(v38);
      }
      v36 += 10;
    }
    while ( v37 != v36 );
    v35 = *(unsigned int *)(a1 + 752);
  }
  v39 = 80 * v35;
  sub_C7D6A0(*(_QWORD *)(a1 + 736), 80 * v35, 8);
  v40 = *(_QWORD *)(a1 + 640);
  if ( v40 != a1 + 656 )
    _libc_free(v40);
  v41 = *(__int64 **)(a1 + 560);
  v42 = &v41[*(unsigned int *)(a1 + 568)];
  if ( v41 != v42 )
  {
    for ( j = *(_QWORD *)(a1 + 560); ; j = *(_QWORD *)(a1 + 560) )
    {
      v44 = *v41;
      v45 = (unsigned int)(((__int64)v41 - j) >> 3) >> 7;
      v39 = 4096LL << v45;
      if ( v45 >= 0x1E )
        v39 = 0x40000000000LL;
      ++v41;
      sub_C7D6A0(v44, v39, 16);
      if ( v42 == v41 )
        break;
    }
  }
  v46 = *(__int64 **)(a1 + 608);
  v47 = (unsigned __int64)&v46[2 * *(unsigned int *)(a1 + 616)];
  if ( v46 != (__int64 *)v47 )
  {
    do
    {
      v39 = v46[1];
      v48 = *v46;
      v46 += 2;
      sub_C7D6A0(v48, v39, 16);
    }
    while ( (__int64 *)v47 != v46 );
    v47 = *(_QWORD *)(a1 + 608);
  }
  if ( v47 != a1 + 624 )
    _libc_free(v47);
  v49 = *(_QWORD *)(a1 + 560);
  if ( v49 != a1 + 576 )
    _libc_free(v49);
  sub_C65770((_QWORD *)(a1 + 520), v39);
  v50 = *(__int64 **)(a1 + 440);
  v51 = *(unsigned int *)(a1 + 448);
  *(_QWORD *)(a1 + 416) = 0;
  v52 = &v50[v51];
  if ( v50 != v52 )
  {
    for ( k = v50; ; k = *(__int64 **)(a1 + 440) )
    {
      v54 = *v50;
      v55 = (unsigned int)(v50 - k) >> 7;
      v56 = 4096LL << v55;
      if ( v55 >= 0x1E )
        v56 = 0x40000000000LL;
      ++v50;
      sub_C7D6A0(v54, v56, 16);
      if ( v52 == v50 )
        break;
    }
  }
  v57 = *(__int64 **)(a1 + 488);
  v58 = (unsigned __int64)&v57[2 * *(unsigned int *)(a1 + 496)];
  if ( v57 != (__int64 *)v58 )
  {
    do
    {
      v59 = v57[1];
      v60 = *v57;
      v57 += 2;
      sub_C7D6A0(v60, v59, 16);
    }
    while ( (__int64 *)v58 != v57 );
    v58 = *(_QWORD *)(a1 + 488);
  }
  if ( v58 != a1 + 504 )
    _libc_free(v58);
  v61 = *(_QWORD *)(a1 + 440);
  if ( v61 != a1 + 456 )
    _libc_free(v61);
  v62 = *(_QWORD **)(a1 + 408);
  if ( v62 != (_QWORD *)(a1 + 400) )
  {
    v6 = (unsigned __int64 *)v62[1];
    v7 = *v62 & 0xFFFFFFFFFFFFFFF8LL;
    *v6 = v7 | *v6 & 7;
    *(_QWORD *)(v7 + 8) = v6;
    v62[1] = 0;
    *v62 &= 7uLL;
    BUG();
  }
  v63 = *(_QWORD *)(a1 + 368);
  if ( v63 )
    sub_B91220(a1 + 368, v63);
  v64 = *(__int64 **)(a1 + 208);
  v65 = &v64[*(unsigned int *)(a1 + 216)];
  if ( v64 != v65 )
  {
    for ( m = *(_QWORD *)(a1 + 208); ; m = *(_QWORD *)(a1 + 208) )
    {
      v67 = *v64;
      v68 = (unsigned int)(((__int64)v64 - m) >> 3) >> 7;
      v63 = 4096LL << v68;
      if ( v68 >= 0x1E )
        v63 = 0x40000000000LL;
      ++v64;
      sub_C7D6A0(v67, v63, 16);
      if ( v65 == v64 )
        break;
    }
  }
  v69 = *(__int64 **)(a1 + 256);
  v70 = (unsigned __int64)&v69[2 * *(unsigned int *)(a1 + 264)];
  if ( v69 != (__int64 *)v70 )
  {
    do
    {
      v63 = v69[1];
      v71 = *v69;
      v69 += 2;
      sub_C7D6A0(v71, v63, 16);
    }
    while ( (__int64 *)v70 != v69 );
    v70 = *(_QWORD *)(a1 + 256);
  }
  if ( v70 != a1 + 272 )
    _libc_free(v70);
  v72 = *(_QWORD *)(a1 + 208);
  if ( v72 != a1 + 224 )
    _libc_free(v72);
  sub_C65770((_QWORD *)(a1 + 176), v63);
  sub_33C86F0(*(_QWORD *)(a1 + 144));
}
