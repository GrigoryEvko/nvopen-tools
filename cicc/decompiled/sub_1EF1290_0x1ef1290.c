// Function: sub_1EF1290
// Address: 0x1ef1290
//
__int64 __fastcall sub_1EF1290(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // r13
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // r13

  v2 = *(unsigned int *)(a1 + 504);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 488);
    v4 = &v3[9 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[1];
        if ( (_QWORD *)v5 != v3 + 3 )
          _libc_free(v5);
      }
      v3 += 9;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 488));
  v6 = *(_QWORD *)(a1 + 400);
  if ( v6 != a1 + 416 )
    _libc_free(v6);
  _libc_free(*(_QWORD *)(a1 + 376));
  v7 = *(_QWORD *)(a1 + 168);
  v8 = v7 + 24LL * *(unsigned int *)(a1 + 176);
  if ( v7 != v8 )
  {
    do
    {
      v9 = *(_QWORD *)(v8 - 24);
      v8 -= 24LL;
      _libc_free(v9);
    }
    while ( v7 != v8 );
    v8 = *(_QWORD *)(a1 + 168);
  }
  if ( v8 != a1 + 184 )
    _libc_free(v8);
  j___libc_free_0(*(_QWORD *)(a1 + 144));
  j___libc_free_0(*(_QWORD *)(a1 + 88));
  j___libc_free_0(*(_QWORD *)(a1 + 56));
  v10 = *(unsigned int *)(a1 + 32);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 16);
    v12 = &v11[13 * v10];
    do
    {
      if ( *v11 != -16 && *v11 != -8 )
      {
        _libc_free(v11[10]);
        _libc_free(v11[7]);
        _libc_free(v11[4]);
        _libc_free(v11[1]);
      }
      v11 += 13;
    }
    while ( v12 != v11 );
  }
  return j___libc_free_0(*(_QWORD *)(a1 + 16));
}
