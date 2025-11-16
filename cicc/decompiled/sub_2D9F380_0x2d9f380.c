// Function: sub_2D9F380
// Address: 0x2d9f380
//
void __fastcall sub_2D9F380(__int64 a1)
{
  unsigned __int64 v1; // r8

  v1 = *(_QWORD *)(a1 + 56);
  if ( v1 != a1 + 72 )
    _libc_free(v1);
}
