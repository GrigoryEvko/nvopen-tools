// Function: sub_1D17470
// Address: 0x1d17470
//
_QWORD *__fastcall sub_1D17470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 *v6; // rsi
  unsigned __int64 v7; // rcx
  __int64 v8; // r15
  __int64 v10; // r13
  __int64 v11; // rax
  _QWORD *v12; // r12
  _QWORD *v13; // r14
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 *v18; // r12
  unsigned __int64 *v19; // r14
  unsigned __int64 v20; // rdi
  unsigned __int64 *v21; // r12
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // r12
  __int64 v26; // r13
  __int64 v27; // rdi
  unsigned __int64 v28; // r8
  __int64 v29; // r13
  __int64 v30; // r13
  __int64 v31; // r12
  unsigned __int64 v32; // rdi
  __int64 v33; // rdi
  __int64 v34; // rdi
  unsigned __int64 *v35; // r12
  unsigned __int64 *v36; // r13
  unsigned __int64 v37; // rdi
  unsigned __int64 *v38; // r12
  unsigned __int64 v39; // r13
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  unsigned __int64 *v43; // r12
  unsigned __int64 *v44; // r13
  unsigned __int64 v45; // rdi
  unsigned __int64 *v46; // r12
  unsigned __int64 v47; // r13
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rdi
  unsigned __int64 *v50; // r12
  __int64 v51; // rax
  unsigned __int64 *v52; // r13
  unsigned __int64 v53; // rdi
  unsigned __int64 *v54; // r12
  unsigned __int64 v55; // r13
  unsigned __int64 v56; // rdi
  unsigned __int64 v57; // rdi
  _QWORD *result; // rax
  __int64 v59; // rsi

  sub_1D173F0(a1, a2, a3, a4, a5, a6);
  v10 = *(_QWORD *)(a1 + 648);
  *(_DWORD *)(a1 + 472) = 0;
  if ( v10 )
  {
    v11 = *(unsigned int *)(v10 + 720);
    if ( (_DWORD)v11 )
    {
      v12 = *(_QWORD **)(v10 + 704);
      v13 = &v12[5 * v11];
      do
      {
        if ( *v12 != -16 && *v12 != -8 )
        {
          v14 = v12[1];
          if ( (_QWORD *)v14 != v12 + 3 )
            _libc_free(v14);
        }
        v12 += 5;
      }
      while ( v13 != v12 );
    }
    j___libc_free_0(*(_QWORD *)(v10 + 704));
    v15 = *(_QWORD *)(v10 + 648);
    if ( v15 != v10 + 664 )
      _libc_free(v15);
    v16 = *(_QWORD *)(v10 + 376);
    if ( v16 != v10 + 392 )
      _libc_free(v16);
    v17 = *(_QWORD *)(v10 + 104);
    if ( v17 != v10 + 120 )
      _libc_free(v17);
    v18 = *(unsigned __int64 **)(v10 + 16);
    v19 = &v18[*(unsigned int *)(v10 + 24)];
    while ( v19 != v18 )
    {
      v20 = *v18++;
      _libc_free(v20);
    }
    v21 = *(unsigned __int64 **)(v10 + 64);
    v22 = (unsigned __int64)&v21[2 * *(unsigned int *)(v10 + 72)];
    if ( v21 != (unsigned __int64 *)v22 )
    {
      do
      {
        v23 = *v21;
        v21 += 2;
        _libc_free(v23);
      }
      while ( (unsigned __int64 *)v22 != v21 );
      v22 = *(_QWORD *)(v10 + 64);
    }
    if ( v22 != v10 + 80 )
      _libc_free(v22);
    v24 = *(_QWORD *)(v10 + 16);
    if ( v24 != v10 + 32 )
      _libc_free(v24);
    j_j___libc_free_0(v10, 728);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 880));
  v25 = *(_QWORD *)(a1 + 840);
  while ( v25 )
  {
    v26 = v25;
    sub_1D13F90(*(_QWORD **)(v25 + 24));
    v27 = *(_QWORD *)(v25 + 32);
    v25 = *(_QWORD *)(v25 + 16);
    if ( v27 != v26 + 48 )
      j_j___libc_free_0(v27, *(_QWORD *)(v26 + 48) + 1LL);
    j_j___libc_free_0(v26, 80);
  }
  v28 = *(_QWORD *)(a1 + 792);
  if ( *(_DWORD *)(a1 + 804) )
  {
    v29 = *(unsigned int *)(a1 + 800);
    if ( (_DWORD)v29 )
    {
      v30 = 8 * v29;
      v31 = 0;
      do
      {
        v32 = *(_QWORD *)(v28 + v31);
        if ( v32 != -8 && v32 )
        {
          _libc_free(v32);
          v28 = *(_QWORD *)(a1 + 792);
        }
        v31 += 8;
      }
      while ( v30 != v31 );
    }
  }
  _libc_free(v28);
  sub_1D13CC0(*(_QWORD *)(a1 + 760));
  v33 = *(_QWORD *)(a1 + 720);
  if ( v33 )
    j_j___libc_free_0(v33, *(_QWORD *)(a1 + 736) - v33);
  v34 = *(_QWORD *)(a1 + 696);
  if ( v34 )
    j_j___libc_free_0(v34, *(_QWORD *)(a1 + 712) - v34);
  *(_QWORD *)(a1 + 672) = &unk_49F9978;
  sub_16BD9D0(a1 + 672);
  v35 = *(unsigned __int64 **)(a1 + 560);
  v36 = &v35[*(unsigned int *)(a1 + 568)];
  while ( v36 != v35 )
  {
    v37 = *v35++;
    _libc_free(v37);
  }
  v38 = *(unsigned __int64 **)(a1 + 608);
  v39 = (unsigned __int64)&v38[2 * *(unsigned int *)(a1 + 616)];
  if ( v38 != (unsigned __int64 *)v39 )
  {
    do
    {
      v40 = *v38;
      v38 += 2;
      _libc_free(v40);
    }
    while ( (unsigned __int64 *)v39 != v38 );
    v39 = *(_QWORD *)(a1 + 608);
  }
  if ( v39 != a1 + 624 )
    _libc_free(v39);
  v41 = *(_QWORD *)(a1 + 560);
  if ( v41 != a1 + 576 )
    _libc_free(v41);
  v42 = *(_QWORD *)(a1 + 464);
  if ( v42 != a1 + 480 )
    _libc_free(v42);
  v43 = *(unsigned __int64 **)(a1 + 376);
  v44 = &v43[*(unsigned int *)(a1 + 384)];
  while ( v44 != v43 )
  {
    v45 = *v43++;
    _libc_free(v45);
  }
  v46 = *(unsigned __int64 **)(a1 + 424);
  v47 = (unsigned __int64)&v46[2 * *(unsigned int *)(a1 + 432)];
  if ( v46 != (unsigned __int64 *)v47 )
  {
    do
    {
      v48 = *v46;
      v46 += 2;
      _libc_free(v48);
    }
    while ( (unsigned __int64 *)v47 != v46 );
    v47 = *(_QWORD *)(a1 + 424);
  }
  if ( v47 != a1 + 440 )
    _libc_free(v47);
  v49 = *(_QWORD *)(a1 + 376);
  if ( v49 != a1 + 392 )
    _libc_free(v49);
  *(_QWORD *)(a1 + 320) = &unk_49F9918;
  sub_16BD9D0(a1 + 320);
  v50 = *(unsigned __int64 **)(a1 + 232);
  v51 = *(unsigned int *)(a1 + 240);
  *(_QWORD *)(a1 + 208) = 0;
  v52 = &v50[v51];
  while ( v52 != v50 )
  {
    v53 = *v50++;
    _libc_free(v53);
  }
  v54 = *(unsigned __int64 **)(a1 + 280);
  v55 = (unsigned __int64)&v54[2 * *(unsigned int *)(a1 + 288)];
  if ( v54 != (unsigned __int64 *)v55 )
  {
    do
    {
      v56 = *v54;
      v54 += 2;
      _libc_free(v56);
    }
    while ( (unsigned __int64 *)v55 != v54 );
    v55 = *(_QWORD *)(a1 + 280);
  }
  if ( v55 != a1 + 296 )
    _libc_free(v55);
  v57 = *(_QWORD *)(a1 + 232);
  if ( v57 != a1 + 248 )
    _libc_free(v57);
  result = *(_QWORD **)(a1 + 200);
  if ( result != (_QWORD *)(a1 + 192) )
  {
    v6 = (unsigned __int64 *)result[1];
    v7 = *result & 0xFFFFFFFFFFFFFFF8LL;
    *v6 = v7 | *v6 & 7;
    *(_QWORD *)(v7 + 8) = v6;
    *result &= 7uLL;
    result[1] = 0;
    nullsub_686(0, v8, 0);
    BUG();
  }
  v59 = *(_QWORD *)(a1 + 160);
  if ( v59 )
    return (_QWORD *)sub_161E7C0(a1 + 160, v59);
  return result;
}
