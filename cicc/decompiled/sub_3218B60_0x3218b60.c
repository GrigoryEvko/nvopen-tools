// Function: sub_3218B60
// Address: 0x3218b60
//
void __fastcall sub_3218B60(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // rdi

  v2 = *a2;
  if ( *a2 )
  {
    v3 = *(_QWORD *)(v2 + 8);
    if ( v3 != v2 + 24 )
      _libc_free(v3);
    j_j___libc_free_0(v2);
  }
}
