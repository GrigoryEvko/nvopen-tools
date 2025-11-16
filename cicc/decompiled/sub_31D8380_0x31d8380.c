// Function: sub_31D8380
// Address: 0x31d8380
//
__int64 __fastcall sub_31D8380(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r15
  __int64 v7; // rbx
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rbx
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  __int64 v18; // rbx
  unsigned __int64 v19; // r12
  __int64 v20; // rdi
  __int64 v21; // rbx
  unsigned __int64 v22; // r12
  __int64 v23; // rdi
  __int64 v24; // rax
  _QWORD *v25; // r12
  _QWORD *v26; // rbx
  unsigned __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // rsi
  _QWORD *v30; // r12
  _QWORD *v31; // rbx
  __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  __int64 v36; // rdi

  *(_QWORD *)a1 = &unk_4A35060;
  v3 = *(_QWORD *)(a1 + 784);
  if ( v3 != a1 + 800 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 768);
  if ( v4 )
  {
    sub_C7D6A0(*(_QWORD *)(v4 + 16), 24LL * *(unsigned int *)(v4 + 32), 8);
    a2 = 40;
    j_j___libc_free_0(v4);
  }
  v5 = *(_QWORD *)(a1 + 752);
  if ( v5 )
    sub_31D7970(v5, a2);
  v6 = *(_QWORD *)(a1 + 744);
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 24);
    v8 = v7 + 8LL * *(unsigned int *)(v6 + 32);
    if ( v7 != v8 )
    {
      do
      {
        v9 = *(_QWORD *)(v8 - 8);
        v8 -= 8LL;
        if ( v9 )
        {
          v10 = *(_QWORD *)(v9 + 24);
          if ( v10 != v9 + 40 )
            _libc_free(v10);
          j_j___libc_free_0(v9);
        }
      }
      while ( v7 != v8 );
      v8 = *(_QWORD *)(v6 + 24);
    }
    if ( v8 != v6 + 40 )
      _libc_free(v8);
    if ( *(_QWORD *)v6 != v6 + 16 )
      _libc_free(*(_QWORD *)v6);
    j_j___libc_free_0(v6);
  }
  v11 = *(_QWORD *)(a1 + 728);
  if ( v11 != a1 + 744 )
    _libc_free(v11);
  sub_C7D6A0(*(_QWORD *)(a1 + 704), 16LL * *(unsigned int *)(a1 + 720), 8);
  v12 = *(_QWORD *)(a1 + 680);
  if ( a1 + 696 != v12 )
    _libc_free(v12);
  v13 = 16LL * *(unsigned int *)(a1 + 672);
  sub_C7D6A0(*(_QWORD *)(a1 + 656), v13, 8);
  v14 = *(_QWORD *)(a1 + 632);
  v15 = *(_QWORD *)(a1 + 624);
  if ( v14 != v15 )
  {
    do
    {
      v16 = *(_QWORD *)(v15 + 128);
      if ( v16 != v15 + 144 )
        _libc_free(v16);
      v17 = *(_QWORD *)(v15 + 16);
      if ( v17 != v15 + 32 )
        _libc_free(v17);
      v15 += 192LL;
    }
    while ( v14 != v15 );
    v15 = *(_QWORD *)(a1 + 624);
  }
  if ( v15 )
  {
    v13 = *(_QWORD *)(a1 + 640) - v15;
    j_j___libc_free_0(v15);
  }
  v18 = *(_QWORD *)(a1 + 576);
  v19 = v18 + 8LL * *(unsigned int *)(a1 + 584);
  if ( v18 != v19 )
  {
    do
    {
      v20 = *(_QWORD *)(v19 - 8);
      v19 -= 8LL;
      if ( v20 )
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v20 + 8LL))(v20, v13);
    }
    while ( v18 != v19 );
    v19 = *(_QWORD *)(a1 + 576);
  }
  if ( v19 != a1 + 592 )
    _libc_free(v19);
  v21 = *(_QWORD *)(a1 + 552);
  v22 = v21 + 8LL * *(unsigned int *)(a1 + 560);
  if ( v21 != v22 )
  {
    do
    {
      v23 = *(_QWORD *)(v22 - 8);
      v22 -= 8LL;
      if ( v23 )
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v23 + 8LL))(v23, v13);
    }
    while ( v21 != v22 );
    v22 = *(_QWORD *)(a1 + 552);
  }
  if ( v22 != a1 + 568 )
    _libc_free(v22);
  v24 = *(unsigned int *)(a1 + 528);
  if ( (_DWORD)v24 )
  {
    v25 = *(_QWORD **)(a1 + 512);
    v26 = &v25[9 * v24];
    do
    {
      if ( *v25 != -4096 && *v25 != -8192 )
      {
        v27 = v25[1];
        if ( (_QWORD *)v27 != v25 + 3 )
          _libc_free(v27);
      }
      v25 += 9;
    }
    while ( v26 != v25 );
    v24 = *(unsigned int *)(a1 + 528);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 512), 72 * v24, 8);
  v28 = *(_QWORD *)(a1 + 496);
  if ( v28 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v28 + 8LL))(v28);
  v29 = *(unsigned int *)(a1 + 480);
  if ( (_DWORD)v29 )
  {
    v30 = *(_QWORD **)(a1 + 464);
    v31 = &v30[2 * v29];
    do
    {
      if ( *v30 != -8192 && *v30 != -4096 )
      {
        v32 = v30[1];
        if ( v32 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 8LL))(v32);
      }
      v30 += 2;
    }
    while ( v31 != v30 );
    v29 = *(unsigned int *)(a1 + 480);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 464), 16 * v29, 8);
  v33 = *(_QWORD *)(a1 + 448);
  if ( v33 )
    sub_31D8060(v33);
  sub_C7D6A0(*(_QWORD *)(a1 + 416), 16LL * *(unsigned int *)(a1 + 432), 8);
  v34 = *(_QWORD *)(a1 + 384);
  if ( v34 != a1 + 400 )
    _libc_free(v34);
  sub_C7D6A0(*(_QWORD *)(a1 + 360), 16LL * *(unsigned int *)(a1 + 376), 8);
  v35 = *(_QWORD *)(a1 + 336);
  if ( a1 + 352 != v35 )
    _libc_free(v35);
  sub_C7D6A0(*(_QWORD *)(a1 + 312), 12LL * *(unsigned int *)(a1 + 328), 4);
  v36 = *(_QWORD *)(a1 + 224);
  if ( v36 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 56LL))(v36);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
