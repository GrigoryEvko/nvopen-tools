// Function: sub_2308500
// Address: 0x2308500
//
void __fastcall sub_2308500(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rsi
  __int64 v4; // rdi
  __int64 v5; // rbx
  unsigned __int64 v6; // rdi

  *(_QWORD *)a1 = &unk_4A11978;
  v2 = *(_QWORD *)(a1 + 2136);
  if ( v2 != a1 + 2152 )
    _libc_free(v2);
  v3 = *(unsigned int *)(a1 + 2128);
  v4 = *(_QWORD *)(a1 + 2112);
  v5 = a1 + 24;
  sub_C7D6A0(v4, 8 * v3, 8);
  sub_C7D6A0(*(_QWORD *)(v5 + 2056), 16LL * *(unsigned int *)(v5 + 2072), 8);
  v6 = *(_QWORD *)(v5 - 16);
  if ( v6 != v5 )
    _libc_free(v6);
}
