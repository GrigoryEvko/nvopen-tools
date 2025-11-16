// Function: sub_1E046F0
// Address: 0x1e046f0
//
__int64 __fastcall sub_1E046F0(__int64 a1)
{
  __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 24);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 40) - v2);
  return j_j___libc_free_0(a1, 56);
}
