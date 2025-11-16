// Function: sub_1E09EE0
// Address: 0x1e09ee0
//
void __fastcall sub_1E09EE0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // r13
  __int64 v13; // r12
  unsigned __int64 v14; // rdi
  __int64 v15; // rdi
  _QWORD *v16; // r13
  _QWORD *v17; // r12
  unsigned __int64 v18; // rdi

  v2 = *(_QWORD *)(a1 + 360);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 376) - v2);
  v3 = *(_QWORD *)(a1 + 328);
  if ( v3 != a1 + 344 )
    _libc_free(v3);
  _libc_free(*(_QWORD *)(a1 + 304));
  _libc_free(*(_QWORD *)(a1 + 280));
  v4 = *(_QWORD *)(a1 + 272);
  if ( v4 )
    j_j___libc_free_0_0(v4);
  v5 = *(_QWORD *)(a1 + 232);
  if ( v5 != a1 + 248 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 208);
  v7 = v6 + 40LL * *(unsigned int *)(a1 + 216);
  if ( v6 != v7 )
  {
    do
    {
      v7 -= 40LL;
      v8 = *(_QWORD *)(v7 + 8);
      if ( v8 != v7 + 24 )
        _libc_free(v8);
    }
    while ( v6 != v7 );
    v7 = *(_QWORD *)(a1 + 208);
  }
  if ( v7 != a1 + 224 )
    _libc_free(v7);
  v9 = *(_QWORD *)(a1 + 160);
  if ( v9 != a1 + 176 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 120);
  if ( *(_DWORD *)(a1 + 132) )
  {
    v11 = *(unsigned int *)(a1 + 128);
    if ( (_DWORD)v11 )
    {
      v12 = 8 * v11;
      v13 = 0;
      do
      {
        v14 = *(_QWORD *)(v10 + v13);
        if ( v14 && v14 != -8 )
        {
          _libc_free(v14);
          v10 = *(_QWORD *)(a1 + 120);
        }
        v13 += 8;
      }
      while ( v12 != v13 );
    }
  }
  _libc_free(v10);
  v15 = *(_QWORD *)(a1 + 80);
  if ( v15 != a1 + 96 )
    j_j___libc_free_0(v15, *(_QWORD *)(a1 + 96) + 1LL);
  v16 = *(_QWORD **)(a1 + 64);
  v17 = &v16[4 * *(unsigned int *)(a1 + 72)];
  if ( v16 != v17 )
  {
    do
    {
      v17 -= 4;
      if ( (_QWORD *)*v17 != v17 + 2 )
        j_j___libc_free_0(*v17, v17[2] + 1LL);
    }
    while ( v16 != v17 );
    v17 = *(_QWORD **)(a1 + 64);
  }
  if ( v17 != (_QWORD *)(a1 + 80) )
    _libc_free((unsigned __int64)v17);
  v18 = *(_QWORD *)(a1 + 24);
  if ( v18 != a1 + 40 )
    _libc_free(v18);
}
