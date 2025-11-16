// Function: sub_2E78DE0
// Address: 0x2e78de0
//
void __fastcall sub_2E78DE0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // r13
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // r8
  __int64 v13; // r13
  __int64 v14; // r13
  __int64 v15; // r12
  _QWORD *v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 *v18; // r13
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi

  v2 = *(_QWORD *)(a1 + 488);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 456);
  if ( v3 != a1 + 472 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 384);
  if ( v4 != a1 + 400 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 312);
  if ( v5 != a1 + 328 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 304);
  if ( v6 )
    j_j___libc_free_0_0(v6);
  v7 = *(_QWORD *)(a1 + 264);
  if ( v7 != a1 + 280 )
    _libc_free(v7);
  v8 = *(_QWORD *)(a1 + 240);
  v9 = v8 + 40LL * *(unsigned int *)(a1 + 248);
  if ( v8 != v9 )
  {
    do
    {
      v9 -= 40LL;
      v10 = *(_QWORD *)(v9 + 8);
      if ( v10 != v9 + 24 )
        _libc_free(v10);
    }
    while ( v8 != v9 );
    v9 = *(_QWORD *)(a1 + 240);
  }
  if ( v9 != a1 + 256 )
    _libc_free(v9);
  v11 = *(_QWORD *)(a1 + 184);
  if ( v11 != a1 + 208 )
    _libc_free(v11);
  v12 = *(_QWORD *)(a1 + 152);
  if ( *(_DWORD *)(a1 + 164) )
  {
    v13 = *(unsigned int *)(a1 + 160);
    if ( (_DWORD)v13 )
    {
      v14 = 8 * v13;
      v15 = 0;
      do
      {
        v16 = *(_QWORD **)(v12 + v15);
        if ( v16 && v16 != (_QWORD *)-8LL )
        {
          sub_C7D6A0((__int64)v16, *v16 + 9LL, 8);
          v12 = *(_QWORD *)(a1 + 152);
        }
        v15 += 8;
      }
      while ( v14 != v15 );
    }
  }
  _libc_free(v12);
  v17 = *(_QWORD *)(a1 + 112);
  if ( v17 != a1 + 128 )
    j_j___libc_free_0(v17);
  v18 = *(unsigned __int64 **)(a1 + 96);
  v19 = &v18[4 * *(unsigned int *)(a1 + 104)];
  if ( v18 != v19 )
  {
    do
    {
      v19 -= 4;
      if ( (unsigned __int64 *)*v19 != v19 + 2 )
        j_j___libc_free_0(*v19);
    }
    while ( v18 != v19 );
    v19 = *(unsigned __int64 **)(a1 + 96);
  }
  if ( v19 != (unsigned __int64 *)(a1 + 112) )
    _libc_free((unsigned __int64)v19);
  v20 = *(_QWORD *)(a1 + 56);
  if ( v20 != a1 + 72 )
    _libc_free(v20);
  if ( !*(_BYTE *)(a1 + 36) )
    _libc_free(*(_QWORD *)(a1 + 16));
}
