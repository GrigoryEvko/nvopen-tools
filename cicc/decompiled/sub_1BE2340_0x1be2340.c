// Function: sub_1BE2340
// Address: 0x1be2340
//
void __fastcall sub_1BE2340(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1 + 96;
  v3 = *(_QWORD *)(a1 + 80);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 48);
  if ( v4 != a1 + 64 )
    _libc_free(v4);
}
