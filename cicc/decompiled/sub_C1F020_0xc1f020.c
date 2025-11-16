// Function: sub_C1F020
// Address: 0xc1f020
//
__int64 __fastcall sub_C1F020(__int64 a1)
{
  __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 24) - v2);
  return j_j___libc_free_0(a1, 88);
}
