// Function: sub_391A900
// Address: 0x391a900
//
void __fastcall sub_391A900(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rbx
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  __int64 v16; // rbx
  _QWORD *v17; // r12
  _QWORD *v18; // rbx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 v23; // rdi

  v2 = *(unsigned int *)(a1 + 704);
  v3 = *(_QWORD *)(a1 + 696);
  *(_QWORD *)a1 = off_4A3EC58;
  v4 = v3 + (v2 << 6);
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 64LL;
      v5 = *(_QWORD *)(v4 + 40);
      if ( v5 != v4 + 56 )
        _libc_free(v5);
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 696);
  }
  if ( v4 != a1 + 712 )
    _libc_free(v4);
  v6 = *(_QWORD *)(a1 + 616);
  if ( v6 != a1 + 632 )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 344);
  v8 = v7 + ((unsigned __int64)*(unsigned int *)(a1 + 352) << 6);
  if ( v7 != v8 )
  {
    do
    {
      v8 -= 64LL;
      v9 = *(_QWORD *)(v8 + 32);
      if ( v9 != v8 + 48 )
        _libc_free(v9);
      v10 = *(_QWORD *)(v8 + 8);
      if ( v10 != v8 + 24 )
        _libc_free(v10);
    }
    while ( v7 != v8 );
    v8 = *(_QWORD *)(a1 + 344);
  }
  if ( v8 != a1 + 360 )
    _libc_free(v8);
  v11 = *(unsigned int *)(a1 + 336);
  if ( (_DWORD)v11 )
  {
    v12 = *(_QWORD *)(a1 + 320);
    v13 = v12 + 72 * v11;
    do
    {
      v14 = *(_QWORD *)(v12 + 32);
      if ( v14 != v12 + 48 )
        _libc_free(v14);
      v15 = *(_QWORD *)(v12 + 8);
      if ( v15 != v12 + 24 )
        _libc_free(v15);
      v12 += 72;
    }
    while ( v13 != v12 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 320));
  j___libc_free_0(*(_QWORD *)(a1 + 288));
  v16 = *(unsigned int *)(a1 + 272);
  if ( (_DWORD)v16 )
  {
    v17 = *(_QWORD **)(a1 + 256);
    v18 = &v17[4 * v16];
    do
    {
      if ( *v17 != -16 && *v17 != -8 )
      {
        v19 = v17[1];
        if ( v19 )
          j_j___libc_free_0(v19);
      }
      v17 += 4;
    }
    while ( v18 != v17 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 256));
  v20 = *(_QWORD *)(a1 + 224);
  if ( v20 )
    j_j___libc_free_0(v20);
  j___libc_free_0(*(_QWORD *)(a1 + 200));
  j___libc_free_0(*(_QWORD *)(a1 + 168));
  j___libc_free_0(*(_QWORD *)(a1 + 136));
  j___libc_free_0(*(_QWORD *)(a1 + 104));
  v21 = *(_QWORD *)(a1 + 64);
  if ( v21 )
    j_j___libc_free_0(v21);
  v22 = *(_QWORD *)(a1 + 32);
  if ( v22 )
    j_j___libc_free_0(v22);
  v23 = *(_QWORD *)(a1 + 24);
  if ( v23 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
  nullsub_1935();
}
