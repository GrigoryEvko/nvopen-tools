// Function: sub_39A20E0
// Address: 0x39a20e0
//
void __fastcall sub_39A20E0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // r13
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 *v8; // r12
  unsigned __int64 *v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi

  *(_QWORD *)a1 = &unk_4A3FD10;
  v2 = *(_QWORD *)(a1 + 392);
  if ( v2 != a1 + 408 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 376);
  v4 = *(_QWORD *)(a1 + 368);
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(_QWORD *)(v4 + 8);
      if ( v5 != v4 + 24 )
        _libc_free(v5);
      v4 += 88LL;
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 368);
  }
  if ( v4 )
    j_j___libc_free_0(v4);
  j___libc_free_0(*(_QWORD *)(a1 + 344));
  j___libc_free_0(*(_QWORD *)(a1 + 312));
  v6 = *(_QWORD *)(a1 + 280);
  if ( v6 )
    j_j___libc_free_0(v6);
  v7 = *(_QWORD *)(a1 + 256);
  if ( v7 )
    j_j___libc_free_0(v7);
  j___libc_free_0(*(_QWORD *)(a1 + 232));
  v8 = *(unsigned __int64 **)(a1 + 104);
  v9 = &v8[*(unsigned int *)(a1 + 112)];
  while ( v9 != v8 )
  {
    v10 = *v8++;
    _libc_free(v10);
  }
  v11 = *(unsigned __int64 **)(a1 + 152);
  v12 = (unsigned __int64)&v11[2 * *(unsigned int *)(a1 + 160)];
  if ( v11 != (unsigned __int64 *)v12 )
  {
    do
    {
      v13 = *v11;
      v11 += 2;
      _libc_free(v13);
    }
    while ( (unsigned __int64 *)v12 != v11 );
    v12 = *(_QWORD *)(a1 + 152);
  }
  if ( v12 != a1 + 168 )
    _libc_free(v12);
  v14 = *(_QWORD *)(a1 + 104);
  if ( v14 != a1 + 120 )
    _libc_free(v14);
}
