// Function: sub_D896C0
// Address: 0xd896c0
//
__int64 __fastcall sub_D896C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // r12
  _QWORD *v5; // r13
  _QWORD *v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdi
  _QWORD *v9; // r13
  _QWORD *v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // r13
  _QWORD *v16; // rdi
  _QWORD *v17; // rdi
  _QWORD *v18; // rdi
  _QWORD *v19; // rdi

  v3 = *(unsigned int *)(a1 + 1352);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 1336);
    v5 = &v4[9 * v3];
    do
    {
      if ( *v4 != -8192 && *v4 != -4096 )
      {
        v6 = (_QWORD *)v4[1];
        if ( v6 != v4 + 3 )
          _libc_free(v6, a2);
      }
      v4 += 9;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 1352);
  }
  v7 = 72 * v3;
  sub_C7D6A0(*(_QWORD *)(a1 + 1336), 72 * v3, 8);
  v8 = *(_QWORD *)(a1 + 1256);
  if ( v8 != a1 + 1272 )
    _libc_free(v8, v7);
  v9 = *(_QWORD **)(a1 + 664);
  v10 = &v9[9 * *(unsigned int *)(a1 + 672)];
  if ( v9 != v10 )
  {
    do
    {
      v10 -= 9;
      if ( (_QWORD *)*v10 != v10 + 2 )
        _libc_free(*v10, v7);
    }
    while ( v9 != v10 );
    v10 = *(_QWORD **)(a1 + 664);
  }
  if ( v10 != (_QWORD *)(a1 + 680) )
    _libc_free(v10, v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 640), 16LL * *(unsigned int *)(a1 + 656), 8);
  v11 = 16LL * *(unsigned int *)(a1 + 600);
  sub_C7D6A0(*(_QWORD *)(a1 + 584), v11, 8);
  v12 = *(_QWORD *)(a1 + 48);
  if ( v12 != a1 + 64 )
    _libc_free(v12, v11);
  v13 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v13 )
  {
    v14 = *(_QWORD **)(a1 + 24);
    v15 = &v14[37 * v13];
    do
    {
      if ( *v14 != -8192 && *v14 != -4096 )
      {
        v16 = (_QWORD *)v14[28];
        if ( v16 != v14 + 30 )
          _libc_free(v16, v11);
        v17 = (_QWORD *)v14[19];
        if ( v17 != v14 + 21 )
          _libc_free(v17, v11);
        v18 = (_QWORD *)v14[10];
        if ( v18 != v14 + 12 )
          _libc_free(v18, v11);
        v19 = (_QWORD *)v14[1];
        if ( v19 != v14 + 3 )
          _libc_free(v19, v11);
      }
      v14 += 37;
    }
    while ( v15 != v14 );
    v13 = *(unsigned int *)(a1 + 40);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 24), 296 * v13, 8);
}
