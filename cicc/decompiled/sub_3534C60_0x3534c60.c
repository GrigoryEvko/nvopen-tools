// Function: sub_3534C60
// Address: 0x3534c60
//
__int64 __fastcall sub_3534C60(_QWORD *a1)
{
  unsigned __int64 v1; // r13
  unsigned __int64 v3; // rdi

  v1 = a1[25];
  *a1 = off_4A38FF8;
  if ( v1 )
  {
    sub_3112140(v1 + 16);
    v3 = *(_QWORD *)(v1 + 16);
    if ( v3 != v1 + 64 )
      j_j___libc_free_0(v3);
    j_j___libc_free_0(v1);
  }
  return sub_BB9260((__int64)a1);
}
