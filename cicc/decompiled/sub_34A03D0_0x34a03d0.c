// Function: sub_34A03D0
// Address: 0x34a03d0
//
void __fastcall sub_34A03D0(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1 + 136;
  v3 = *(_QWORD *)(a1 + 120);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 8);
  if ( v4 != a1 + 24 )
    _libc_free(v4);
}
