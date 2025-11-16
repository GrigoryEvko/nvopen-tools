// Function: sub_3707B10
// Address: 0x3707b10
//
__int64 __fastcall sub_3707B10(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *(_QWORD *)a1 = &unk_4A3C938;
  v2 = *(_QWORD *)(a1 + 120);
  if ( v2 != a1 + 136 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 72);
  if ( v3 != a1 + 88 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 48), 12LL * *(unsigned int *)(a1 + 64), 1);
  return sub_3708F80(a1 + 16);
}
