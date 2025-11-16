// Function: sub_25530C0
// Address: 0x25530c0
//
void __fastcall sub_25530C0(__int64 a1)
{
  unsigned __int64 v1; // r14
  __int64 v3; // r15
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // rbx
  unsigned __int64 v7; // rdi
  __int64 v8; // r12
  unsigned __int64 v9; // r15
  __int64 v10; // rbx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi

  v1 = a1 - 88;
  v3 = *(unsigned int *)(a1 + 56);
  v4 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 - 88) = off_4A1B158;
  *(_QWORD *)a1 = &unk_4A1B1E8;
  v5 = v4 + 16 * v3;
  while ( v5 != v4 )
  {
    v6 = *(_QWORD *)(v4 + 8);
    v7 = *(_QWORD *)(v6 + 56);
    if ( v7 != v6 + 72 )
      _libc_free(v7);
    v4 += 16;
    sub_C7D6A0(*(_QWORD *)(v6 + 32), 8LL * *(unsigned int *)(v6 + 48), 8);
  }
  v8 = *(_QWORD *)(a1 + 96);
  v9 = v8 + 16LL * *(unsigned int *)(a1 + 104);
  if ( v8 != v9 )
  {
    do
    {
      v10 = *(_QWORD *)(v8 + 8);
      v11 = *(_QWORD *)(v10 + 56);
      if ( v11 != v10 + 72 )
        _libc_free(v11);
      v8 += 16;
      sub_C7D6A0(*(_QWORD *)(v10 + 32), 8LL * *(unsigned int *)(v10 + 48), 8);
    }
    while ( v9 != v8 );
    v9 = *(_QWORD *)(a1 + 96);
  }
  if ( v9 != a1 + 112 )
    _libc_free(v9);
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 16LL * *(unsigned int *)(a1 + 88), 8);
  v12 = *(_QWORD *)(a1 + 48);
  if ( a1 + 64 != v12 )
    _libc_free(v12);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 40), 8);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  sub_254FD20(a1 - 80);
  j_j___libc_free_0(v1);
}
