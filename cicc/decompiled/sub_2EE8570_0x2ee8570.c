// Function: sub_2EE8570
// Address: 0x2ee8570
//
void __fastcall sub_2EE8570(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rdi

  v1 = a1 + 424;
  *(_QWORD *)a1 = &unk_4A2A380;
  v3 = *(_QWORD *)(a1 + 424);
  if ( v3 != a1 + 440 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 408);
  if ( v1 != v4 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 384), 16LL * *(unsigned int *)(a1 + 400), 8);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = v5 + 88LL * *(unsigned int *)(a1 + 16);
  if ( v5 != v6 )
  {
    do
    {
      v6 -= 88LL;
      v7 = *(_QWORD *)(v6 + 40);
      if ( v7 != v6 + 56 )
        _libc_free(v7);
    }
    while ( v5 != v6 );
    v6 = *(_QWORD *)(a1 + 8);
  }
  if ( v6 != a1 + 24 )
    _libc_free(v6);
}
