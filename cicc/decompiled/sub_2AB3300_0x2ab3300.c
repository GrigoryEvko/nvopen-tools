// Function: sub_2AB3300
// Address: 0x2ab3300
//
void __fastcall sub_2AB3300(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1 + 88;
  v3 = *(_QWORD *)(a1 + 72);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 24);
  if ( v4 != a1 + 40 )
    _libc_free(v4);
}
