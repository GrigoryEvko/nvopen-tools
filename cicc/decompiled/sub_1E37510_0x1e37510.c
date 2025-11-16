// Function: sub_1E37510
// Address: 0x1e37510
//
__int64 __fastcall sub_1E37510(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = off_49FC080;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, a1[22] - v2);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 184);
}
