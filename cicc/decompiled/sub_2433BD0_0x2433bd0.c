// Function: sub_2433BD0
// Address: 0x2433bd0
//
void __fastcall sub_2433BD0(__int64 a1)
{
  unsigned __int64 v1; // r8

  v1 = *(_QWORD *)(a1 + 16);
  if ( v1 != a1 + 32 )
    _libc_free(v1);
}
