// Function: sub_2309B40
// Address: 0x2309b40
//
void __fastcall sub_2309B40(unsigned __int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  __int64 v5; // r14
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rdi

  v2 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)a1 = &unk_4A10E78;
  v3 = v2 + 88LL * *(unsigned int *)(a1 + 168);
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 88LL;
      v4 = *(_QWORD *)(v3 + 8);
      if ( v4 != v3 + 24 )
        _libc_free(v4);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 160);
  }
  if ( v3 != a1 + 176 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 136), 16LL * *(unsigned int *)(a1 + 152), 8);
  v5 = *(_QWORD *)(a1 + 112);
  v6 = v5 + 88LL * *(unsigned int *)(a1 + 120);
  if ( v5 != v6 )
  {
    do
    {
      v6 -= 88LL;
      v7 = *(_QWORD *)(v6 + 8);
      if ( v7 != v6 + 24 )
        _libc_free(v7);
    }
    while ( v5 != v6 );
    v6 = *(_QWORD *)(a1 + 112);
  }
  if ( a1 + 128 != v6 )
    _libc_free(v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 88), 16LL * *(unsigned int *)(a1 + 104), 8);
  j_j___libc_free_0(a1);
}
