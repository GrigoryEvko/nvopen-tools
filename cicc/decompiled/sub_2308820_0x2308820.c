// Function: sub_2308820
// Address: 0x2308820
//
void __fastcall sub_2308820(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *(_QWORD *)a1 = &unk_4A11978;
  v2 = *(_QWORD *)(a1 + 2136);
  if ( v2 != a1 + 2152 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 2112), 8LL * *(unsigned int *)(a1 + 2128), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 2080), 16LL * *(unsigned int *)(a1 + 2096), 8);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 != a1 + 24 )
    _libc_free(v3);
  j_j___libc_free_0(a1);
}
