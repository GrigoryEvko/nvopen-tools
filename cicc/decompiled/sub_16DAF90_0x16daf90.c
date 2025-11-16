// Function: sub_16DAF90
// Address: 0x16daf90
//
void __fastcall sub_16DAF90(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 v5; // r8
  __int64 v6; // r13
  __int64 v7; // r13
  __int64 v8; // r12
  unsigned __int64 v9; // rdi
  _QWORD *v10; // r13
  _QWORD *v11; // r12
  _QWORD *v12; // rdi
  _QWORD *v13; // rdi
  _QWORD *v14; // r13
  _QWORD *v15; // r12
  _QWORD *v16; // rdi
  _QWORD *v17; // rdi

  v2 = a1 + 11656;
  v3 = *(_QWORD *)(a1 + 11640);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 11600);
  if ( v4 != a1 + 11616 )
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 11616) + 1LL);
  v5 = *(_QWORD *)(a1 + 11552);
  if ( *(_DWORD *)(a1 + 11564) )
  {
    v6 = *(unsigned int *)(a1 + 11560);
    if ( (_DWORD)v6 )
    {
      v7 = 8 * v6;
      v8 = 0;
      do
      {
        v9 = *(_QWORD *)(v5 + v8);
        if ( v9 != -8 && v9 )
        {
          _libc_free(v9);
          v5 = *(_QWORD *)(a1 + 11552);
        }
        v8 += 8;
      }
      while ( v7 != v8 );
    }
  }
  _libc_free(v5);
  v10 = *(_QWORD **)(a1 + 1296);
  v11 = &v10[10 * *(unsigned int *)(a1 + 1304)];
  if ( v10 != v11 )
  {
    do
    {
      v11 -= 10;
      v12 = (_QWORD *)v11[6];
      if ( v12 != v11 + 8 )
        j_j___libc_free_0(v12, v11[8] + 1LL);
      v13 = (_QWORD *)v11[2];
      if ( v13 != v11 + 4 )
        j_j___libc_free_0(v13, v11[4] + 1LL);
    }
    while ( v10 != v11 );
    v11 = *(_QWORD **)(a1 + 1296);
  }
  if ( v11 != (_QWORD *)(a1 + 1312) )
    _libc_free((unsigned __int64)v11);
  v14 = *(_QWORD **)a1;
  v15 = (_QWORD *)(*(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v15 )
  {
    do
    {
      v15 -= 10;
      v16 = (_QWORD *)v15[6];
      if ( v16 != v15 + 8 )
        j_j___libc_free_0(v16, v15[8] + 1LL);
      v17 = (_QWORD *)v15[2];
      if ( v17 != v15 + 4 )
        j_j___libc_free_0(v17, v15[4] + 1LL);
    }
    while ( v14 != v15 );
    v15 = *(_QWORD **)a1;
  }
  if ( v15 != (_QWORD *)(a1 + 16) )
    _libc_free((unsigned __int64)v15);
}
