// Function: sub_13C2D40
// Address: 0x13c2d40
//
__int64 __fastcall sub_13C2D40(__int64 a1)
{
  __int64 v1; // r12

  v1 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = 0;
  if ( v1 )
  {
    sub_13C2BD0(v1);
    j_j___libc_free_0(v1, 352);
  }
  return 0;
}
