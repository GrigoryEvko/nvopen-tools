// Function: sub_2F4E350
// Address: 0x2f4e350
//
void __fastcall sub_2F4E350(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rsi
  unsigned __int64 v4; // rdi
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  unsigned __int64 v10; // rdi
  __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  _QWORD *v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  void (__fastcall *v26)(unsigned __int64); // rax
  unsigned __int64 v27; // rdi
  __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  void (__fastcall *v31)(__int64, __int64, __int64); // rax
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  __int64 v37; // r12
  __int64 v38; // rax
  __int64 i; // rbx
  unsigned __int64 v40; // rdi

  *(_QWORD *)a1 = off_4A2B1A8;
  *(_QWORD *)(a1 + 760) = &unk_4A2B218;
  v2 = *(_QWORD *)(a1 + 28984);
  if ( v2 != a1 + 29000 )
    _libc_free(v2);
  v3 = 8LL * *(unsigned int *)(a1 + 28976);
  sub_C7D6A0(*(_QWORD *)(a1 + 28960), v3, 8);
  v4 = *(_QWORD *)(a1 + 28800);
  if ( v4 != a1 + 28816 )
    _libc_free(v4);
  v5 = *(_QWORD **)(a1 + 24176);
  v6 = &v5[18 * *(unsigned int *)(a1 + 24184)];
  if ( v5 != v6 )
  {
    do
    {
      v6 -= 18;
      v7 = v6[12];
      if ( (_QWORD *)v7 != v6 + 14 )
        _libc_free(v7);
      v8 = v6[3];
      if ( (_QWORD *)v8 != v6 + 5 )
        _libc_free(v8);
      v6[2] = 0;
      v9 = v6[1];
      if ( v9 )
        --*(_DWORD *)(v9 + 8);
    }
    while ( v5 != v6 );
    v6 = *(_QWORD **)(a1 + 24176);
  }
  if ( v6 != (_QWORD *)(a1 + 24192) )
    _libc_free((unsigned __int64)v6);
  v10 = *(_QWORD *)(a1 + 24096);
  if ( v10 != a1 + 24112 )
    _libc_free(v10);
  v11 = a1 + 24096;
  _libc_free(*(_QWORD *)(a1 + 1032));
  do
  {
    v11 -= 720;
    v12 = *(_QWORD *)(v11 + 512);
    if ( v12 != v11 + 528 )
      _libc_free(v12);
    v13 = *(_QWORD *)(v11 + 48);
    v14 = v13 + 112LL * *(unsigned int *)(v11 + 56);
    if ( v13 != v14 )
    {
      do
      {
        v14 -= 112LL;
        v15 = *(_QWORD *)(v14 + 8);
        if ( v15 != v14 + 24 )
          _libc_free(v15);
      }
      while ( v13 != v14 );
      v13 = *(_QWORD *)(v11 + 48);
    }
    if ( v13 != v11 + 64 )
      _libc_free(v13);
  }
  while ( v11 != a1 + 1056 );
  v16 = *(_QWORD *)(a1 + 1000);
  if ( v16 )
    sub_2F4CF00(v16);
  v17 = *(_QWORD **)(a1 + 992);
  if ( v17 )
  {
    v18 = v17[78];
    if ( (_QWORD *)v18 != v17 + 80 )
      _libc_free(v18);
    v19 = v17[35];
    if ( (_QWORD *)v19 != v17 + 37 )
      _libc_free(v19);
    v20 = v17[25];
    if ( (_QWORD *)v20 != v17 + 27 )
      _libc_free(v20);
    v21 = v17[7];
    if ( (_QWORD *)v21 != v17 + 9 )
      _libc_free(v21);
    v3 = 704;
    j_j___libc_free_0((unsigned __int64)v17);
  }
  v22 = *(_QWORD *)(a1 + 976);
  if ( v22 )
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v22 + 8LL))(v22, v3);
  v23 = *(_QWORD *)(a1 + 968);
  if ( v23 )
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v23 + 8LL))(v23, v3);
  if ( *(_BYTE *)(a1 + 960) )
  {
    v24 = *(_QWORD *)(a1 + 920);
    if ( v24 != a1 + 936 )
      _libc_free(v24);
  }
  v25 = *(_QWORD *)(a1 + 912);
  if ( v25 )
  {
    v26 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v25 + 8LL);
    if ( v26 == sub_2F4C270 )
    {
      v3 = 56;
      j_j___libc_free_0(v25);
    }
    else
    {
      ((void (__fastcall *)(unsigned __int64, __int64))v26)(v25, v3);
    }
  }
  v27 = *(_QWORD *)(a1 + 880);
  if ( v27 )
  {
    v3 = *(_QWORD *)(a1 + 896) - v27;
    j_j___libc_free_0(v27);
  }
  v28 = *(_QWORD *)(a1 + 872);
  if ( v28 )
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v28 + 16LL))(v28, v3);
  v29 = *(_QWORD *)(a1 + 728);
  *(_QWORD *)a1 = &unk_4A3A030;
  sub_2F4E180(v29);
  v30 = *(_QWORD *)(a1 + 688);
  if ( v30 != a1 + 704 )
    _libc_free(v30);
  if ( !*(_BYTE *)(a1 + 428) )
    _libc_free(*(_QWORD *)(a1 + 408));
  v31 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 384);
  if ( v31 )
    v31(a1 + 368, a1 + 368, 3);
  v32 = *(_QWORD *)(a1 + 344);
  if ( v32 )
    j_j___libc_free_0_0(v32);
  v33 = *(_QWORD *)(a1 + 272);
  if ( v33 != a1 + 288 )
    _libc_free(v33);
  v34 = *(_QWORD *)(a1 + 200);
  if ( v34 != a1 + 216 )
    _libc_free(v34);
  v35 = *(_QWORD *)(a1 + 136);
  if ( v35 != a1 + 160 )
    _libc_free(v35);
  v36 = *(_QWORD *)(a1 + 80);
  if ( v36 != a1 + 104 )
    _libc_free(v36);
  v37 = *(_QWORD *)(a1 + 48);
  if ( v37 )
  {
    v38 = 24LL * *(_QWORD *)(v37 - 8);
    for ( i = v37 + v38; v37 != i; i -= 24 )
    {
      v40 = *(_QWORD *)(i - 8);
      if ( v40 )
        j_j___libc_free_0_0(v40);
    }
    j_j_j___libc_free_0_0(v37 - 8);
  }
}
