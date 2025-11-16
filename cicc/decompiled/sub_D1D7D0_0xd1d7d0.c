// Function: sub_D1D7D0
// Address: 0xd1d7d0
//
__int64 __fastcall sub_D1D7D0(__int64 a1)
{
  __int64 v1; // r12

  v1 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = 0;
  if ( v1 )
  {
    sub_D1D5E0(v1);
    j_j___libc_free_0(v1, 360);
  }
  return 0;
}
