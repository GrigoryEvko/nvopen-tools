// Function: sub_2553210
// Address: 0x2553210
//
__int64 __fastcall sub_2553210(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // rbx
  unsigned __int64 v6; // rdi
  __int64 v7; // r12
  unsigned __int64 v8; // r14
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi

  v2 = *(unsigned int *)(a1 + 56);
  v3 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 - 88) = off_4A1B158;
  *(_QWORD *)a1 = &unk_4A1B1E8;
  v4 = v3 + 16 * v2;
  while ( v4 != v3 )
  {
    v5 = *(_QWORD *)(v3 + 8);
    v6 = *(_QWORD *)(v5 + 56);
    if ( v6 != v5 + 72 )
      _libc_free(v6);
    v3 += 16;
    sub_C7D6A0(*(_QWORD *)(v5 + 32), 8LL * *(unsigned int *)(v5 + 48), 8);
  }
  v7 = *(_QWORD *)(a1 + 96);
  v8 = v7 + 16LL * *(unsigned int *)(a1 + 104);
  if ( v7 != v8 )
  {
    do
    {
      v9 = *(_QWORD *)(v7 + 8);
      v10 = *(_QWORD *)(v9 + 56);
      if ( v10 != v9 + 72 )
        _libc_free(v10);
      v7 += 16;
      sub_C7D6A0(*(_QWORD *)(v9 + 32), 8LL * *(unsigned int *)(v9 + 48), 8);
    }
    while ( v8 != v7 );
    v8 = *(_QWORD *)(a1 + 96);
  }
  if ( v8 != a1 + 112 )
    _libc_free(v8);
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 16LL * *(unsigned int *)(a1 + 88), 8);
  v11 = *(_QWORD *)(a1 + 48);
  if ( a1 + 64 != v11 )
    _libc_free(v11);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 40), 8);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  return sub_254FD20(a1 - 80);
}
