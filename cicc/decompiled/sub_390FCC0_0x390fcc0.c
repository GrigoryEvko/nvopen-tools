// Function: sub_390FCC0
// Address: 0x390fcc0
//
void __fastcall sub_390FCC0(__int64 a1)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // r8
  __int64 v13; // r13
  __int64 v14; // r13
  __int64 v15; // r12
  unsigned __int64 v16; // rdi

  if ( !*(_BYTE *)(a1 + 64) )
  {
    v2 = *(_QWORD *)(a1 + 56);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 112);
      if ( v3 != v2 + 128 )
        _libc_free(v3);
      v4 = *(_QWORD *)(v2 + 64);
      if ( v4 != v2 + 80 )
        _libc_free(v4);
      nullsub_1930();
      j_j___libc_free_0(v2);
    }
  }
  v5 = *(_QWORD *)(a1 + 296);
  v6 = *(_QWORD *)(a1 + 288);
  if ( v5 != v6 )
  {
    do
    {
      v7 = *(_QWORD *)(v6 + 32);
      v6 += 56LL;
      j___libc_free_0(v7);
    }
    while ( v5 != v6 );
    v6 = *(_QWORD *)(a1 + 288);
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  v8 = *(_QWORD *)(a1 + 264);
  if ( v8 )
    j_j___libc_free_0(v8);
  v9 = *(_QWORD *)(a1 + 232);
  while ( v9 )
  {
    sub_390F8B0(*(_QWORD *)(v9 + 24));
    v10 = v9;
    v9 = *(_QWORD *)(v9 + 16);
    j_j___libc_free_0(v10);
  }
  v11 = *(_QWORD *)(a1 + 72);
  if ( v11 != a1 + 88 )
    _libc_free(v11);
  v12 = *(_QWORD *)(a1 + 24);
  if ( *(_DWORD *)(a1 + 36) )
  {
    v13 = *(unsigned int *)(a1 + 32);
    if ( (_DWORD)v13 )
    {
      v14 = 8 * v13;
      v15 = 0;
      do
      {
        v16 = *(_QWORD *)(v12 + v15);
        if ( v16 != -8 && v16 )
        {
          _libc_free(v16);
          v12 = *(_QWORD *)(a1 + 24);
        }
        v15 += 8;
      }
      while ( v14 != v15 );
    }
  }
  _libc_free(v12);
}
