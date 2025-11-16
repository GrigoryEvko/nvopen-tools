// Function: sub_277A450
// Address: 0x277a450
//
void __fastcall sub_277A450(__int64 a1)
{
  __int64 *v2; // r14
  __int64 v3; // rax
  __int64 *v4; // r12
  __int64 *i; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 v8; // rsi
  __int64 *v9; // r12
  unsigned __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 *v14; // r14
  __int64 v15; // rax
  __int64 *v16; // r12
  __int64 *j; // rax
  __int64 v18; // rdi
  unsigned int v19; // ecx
  __int64 v20; // rsi
  __int64 *v21; // r12
  unsigned __int64 v22; // r13
  __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 *v26; // r14
  __int64 v27; // rax
  __int64 *v28; // r12
  __int64 *k; // rax
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 v32; // rsi
  __int64 *v33; // r12
  unsigned __int64 v34; // r13
  __int64 v35; // rsi
  __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  __int64 *v38; // r14
  __int64 v39; // rax
  __int64 *v40; // r12
  __int64 *m; // rax
  __int64 v42; // rdi
  unsigned int v43; // ecx
  __int64 v44; // rsi
  __int64 *v45; // r12
  unsigned __int64 v46; // r13
  __int64 v47; // rsi
  __int64 v48; // rdi
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // r13
  _QWORD *v51; // rbx
  _QWORD *v52; // r12
  __int64 v53; // rax
  _QWORD *v54; // rbx
  _QWORD *v55; // r12
  __int64 v56; // rax

  sub_C7D6A0(*(_QWORD *)(a1 + 704), 32LL * *(unsigned int *)(a1 + 720), 8);
  v2 = *(__int64 **)(a1 + 616);
  v3 = *(unsigned int *)(a1 + 624);
  *(_QWORD *)(a1 + 592) = 0;
  v4 = &v2[v3];
  if ( v2 != v4 )
  {
    for ( i = v2; ; i = *(__int64 **)(a1 + 616) )
    {
      v6 = *v2;
      v7 = (unsigned int)(v2 - i) >> 7;
      v8 = 4096LL << v7;
      if ( v7 >= 0x1E )
        v8 = 0x40000000000LL;
      ++v2;
      sub_C7D6A0(v6, v8, 16);
      if ( v4 == v2 )
        break;
    }
  }
  v9 = *(__int64 **)(a1 + 664);
  v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(a1 + 672)];
  if ( v9 != (__int64 *)v10 )
  {
    do
    {
      v11 = v9[1];
      v12 = *v9;
      v9 += 2;
      sub_C7D6A0(v12, v11, 16);
    }
    while ( (__int64 *)v10 != v9 );
    v10 = *(_QWORD *)(a1 + 664);
  }
  if ( v10 != a1 + 680 )
    _libc_free(v10);
  v13 = *(_QWORD *)(a1 + 616);
  if ( v13 != a1 + 632 )
    _libc_free(v13);
  sub_C7D6A0(*(_QWORD *)(a1 + 560), 16LL * *(unsigned int *)(a1 + 576), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 520), 56LL * *(unsigned int *)(a1 + 536), 8);
  v14 = *(__int64 **)(a1 + 432);
  v15 = *(unsigned int *)(a1 + 440);
  *(_QWORD *)(a1 + 408) = 0;
  v16 = &v14[v15];
  if ( v14 != v16 )
  {
    for ( j = v14; ; j = *(__int64 **)(a1 + 432) )
    {
      v18 = *v14;
      v19 = (unsigned int)(v14 - j) >> 7;
      v20 = 4096LL << v19;
      if ( v19 >= 0x1E )
        v20 = 0x40000000000LL;
      ++v14;
      sub_C7D6A0(v18, v20, 16);
      if ( v16 == v14 )
        break;
    }
  }
  v21 = *(__int64 **)(a1 + 480);
  v22 = (unsigned __int64)&v21[2 * *(unsigned int *)(a1 + 488)];
  if ( v21 != (__int64 *)v22 )
  {
    do
    {
      v23 = v21[1];
      v24 = *v21;
      v21 += 2;
      sub_C7D6A0(v24, v23, 16);
    }
    while ( (__int64 *)v22 != v21 );
    v22 = *(_QWORD *)(a1 + 480);
  }
  if ( v22 != a1 + 496 )
    _libc_free(v22);
  v25 = *(_QWORD *)(a1 + 432);
  if ( v25 != a1 + 448 )
    _libc_free(v25);
  sub_C7D6A0(*(_QWORD *)(a1 + 376), 16LL * *(unsigned int *)(a1 + 392), 8);
  v26 = *(__int64 **)(a1 + 288);
  v27 = *(unsigned int *)(a1 + 296);
  *(_QWORD *)(a1 + 264) = 0;
  v28 = &v26[v27];
  if ( v26 != v28 )
  {
    for ( k = v26; ; k = *(__int64 **)(a1 + 288) )
    {
      v30 = *v26;
      v31 = (unsigned int)(v26 - k) >> 7;
      v32 = 4096LL << v31;
      if ( v31 >= 0x1E )
        v32 = 0x40000000000LL;
      ++v26;
      sub_C7D6A0(v30, v32, 16);
      if ( v28 == v26 )
        break;
    }
  }
  v33 = *(__int64 **)(a1 + 336);
  v34 = (unsigned __int64)&v33[2 * *(unsigned int *)(a1 + 344)];
  if ( v33 != (__int64 *)v34 )
  {
    do
    {
      v35 = v33[1];
      v36 = *v33;
      v33 += 2;
      sub_C7D6A0(v36, v35, 16);
    }
    while ( (__int64 *)v34 != v33 );
    v34 = *(_QWORD *)(a1 + 336);
  }
  if ( v34 != a1 + 352 )
    _libc_free(v34);
  v37 = *(_QWORD *)(a1 + 288);
  if ( v37 != a1 + 304 )
    _libc_free(v37);
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 16LL * *(unsigned int *)(a1 + 248), 8);
  v38 = *(__int64 **)(a1 + 144);
  v39 = *(unsigned int *)(a1 + 152);
  *(_QWORD *)(a1 + 120) = 0;
  v40 = &v38[v39];
  if ( v38 != v40 )
  {
    for ( m = v38; ; m = *(__int64 **)(a1 + 144) )
    {
      v42 = *v38;
      v43 = (unsigned int)(v38 - m) >> 7;
      v44 = 4096LL << v43;
      if ( v43 >= 0x1E )
        v44 = 0x40000000000LL;
      ++v38;
      sub_C7D6A0(v42, v44, 16);
      if ( v40 == v38 )
        break;
    }
  }
  v45 = *(__int64 **)(a1 + 192);
  v46 = (unsigned __int64)&v45[2 * *(unsigned int *)(a1 + 200)];
  if ( v45 != (__int64 *)v46 )
  {
    do
    {
      v47 = v45[1];
      v48 = *v45;
      v45 += 2;
      sub_C7D6A0(v48, v47, 16);
    }
    while ( (__int64 *)v46 != v45 );
    v46 = *(_QWORD *)(a1 + 192);
  }
  if ( v46 != a1 + 208 )
    _libc_free(v46);
  v49 = *(_QWORD *)(a1 + 144);
  if ( v49 != a1 + 160 )
    _libc_free(v49);
  v50 = *(_QWORD *)(a1 + 112);
  if ( v50 )
  {
    sub_2779280(*(_QWORD **)(v50 + 728));
    v51 = *(_QWORD **)(v50 + 504);
    v52 = &v51[3 * *(unsigned int *)(v50 + 512)];
    if ( v51 != v52 )
    {
      do
      {
        v53 = *(v52 - 1);
        v52 -= 3;
        if ( v53 != 0 && v53 != -4096 && v53 != -8192 )
          sub_BD60C0(v52);
      }
      while ( v51 != v52 );
      v52 = *(_QWORD **)(v50 + 504);
    }
    if ( v52 != (_QWORD *)(v50 + 520) )
      _libc_free((unsigned __int64)v52);
    if ( !*(_BYTE *)(v50 + 436) )
      _libc_free(*(_QWORD *)(v50 + 416));
    v54 = *(_QWORD **)(v50 + 8);
    v55 = &v54[3 * *(unsigned int *)(v50 + 16)];
    if ( v54 != v55 )
    {
      do
      {
        v56 = *(v55 - 1);
        v55 -= 3;
        if ( v56 != -4096 && v56 != 0 && v56 != -8192 )
          sub_BD60C0(v55);
      }
      while ( v54 != v55 );
      v55 = *(_QWORD **)(v50 + 8);
    }
    if ( v55 != (_QWORD *)(v50 + 24) )
      _libc_free((unsigned __int64)v55);
    j_j___libc_free_0(v50);
  }
}
