// Function: sub_254FD60
// Address: 0x254fd60
//
__int64 __fastcall sub_254FD60(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // rbx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r14
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi

  v2 = *(unsigned int *)(a1 + 144);
  v3 = *(_QWORD *)(a1 + 136);
  *(_QWORD *)a1 = off_4A1B158;
  *(_QWORD *)(a1 + 88) = &unk_4A1B1E8;
  v4 = v3 + 16 * v2;
  while ( v3 != v4 )
  {
    v5 = *(_QWORD *)(v3 + 8);
    v6 = *(_QWORD *)(v5 + 56);
    if ( v6 != v5 + 72 )
      _libc_free(v6);
    v3 += 16;
    sub_C7D6A0(*(_QWORD *)(v5 + 32), 8LL * *(unsigned int *)(v5 + 48), 8);
  }
  v7 = *(_QWORD *)(a1 + 184);
  v8 = v7 + 16LL * *(unsigned int *)(a1 + 192);
  if ( v8 != v7 )
  {
    do
    {
      v9 = *(_QWORD *)(v7 + 8);
      v10 = *(_QWORD *)(v9 + 56);
      if ( v10 != v9 + 72 )
        _libc_free(v10);
      v7 += 16LL;
      sub_C7D6A0(*(_QWORD *)(v9 + 32), 8LL * *(unsigned int *)(v9 + 48), 8);
    }
    while ( v7 != v8 );
    v7 = *(_QWORD *)(a1 + 184);
  }
  if ( v7 != a1 + 200 )
    _libc_free(v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 160), 16LL * *(unsigned int *)(a1 + 176), 8);
  v11 = *(_QWORD *)(a1 + 136);
  if ( a1 + 152 != v11 )
    _libc_free(v11);
  sub_C7D6A0(*(_QWORD *)(a1 + 112), 16LL * *(unsigned int *)(a1 + 128), 8);
  *(_QWORD *)a1 = &unk_4A16C00;
  return sub_254FD20(a1 + 8);
}
