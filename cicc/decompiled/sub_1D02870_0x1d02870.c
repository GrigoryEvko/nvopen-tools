// Function: sub_1D02870
// Address: 0x1d02870
//
__int64 __fastcall sub_1D02870(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi

  *a1 = &off_49F9520;
  v2 = a1[18];
  if ( v2 )
    j_j___libc_free_0(v2, a1[20] - v2);
  v3 = a1[15];
  if ( v3 )
    j_j___libc_free_0(v3, a1[17] - v3);
  v4 = a1[12];
  if ( v4 )
    j_j___libc_free_0(v4, a1[14] - v4);
  v5 = a1[2];
  if ( v5 )
    j_j___libc_free_0(v5, a1[4] - v5);
  return j_j___libc_free_0(a1, 176);
}
