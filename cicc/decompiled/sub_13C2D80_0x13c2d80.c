// Function: sub_13C2D80
// Address: 0x13c2d80
//
__int64 __fastcall sub_13C2D80(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[20];
  *a1 = &unk_49EA4B0;
  if ( v1 )
  {
    sub_13C2BD0(v1);
    j_j___libc_free_0(v1, 352);
  }
  return sub_1636790(a1);
}
