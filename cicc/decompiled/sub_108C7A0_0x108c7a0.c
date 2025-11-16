// Function: sub_108C7A0
// Address: 0x108c7a0
//
__int64 __fastcall sub_108C7A0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rdi
  __int64 v5; // rdi

  v2 = a1[8];
  *a1 = off_497C080;
  if ( v2 )
  {
    v4 = *(_QWORD *)(v2 + 64);
    if ( v4 != v2 + 80 )
      _libc_free(v4, a2);
    v5 = *(_QWORD *)(v2 + 32);
    if ( v5 != v2 + 48 )
      _libc_free(v5, a2);
    j_j___libc_free_0(v2, 96);
  }
  return j_j___libc_free_0(a1, 80);
}
