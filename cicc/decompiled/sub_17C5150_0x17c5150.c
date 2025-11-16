// Function: sub_17C5150
// Address: 0x17c5150
//
__int64 __fastcall sub_17C5150(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi
  _QWORD *v5; // rdi
  _QWORD *v6; // rdi

  *a1 = off_49F0318;
  v2 = a1[46];
  if ( v2 )
    j_j___libc_free_0(v2, a1[48] - v2);
  v3 = a1[41];
  if ( v3 )
    j_j___libc_free_0(v3, a1[43] - v3);
  v4 = a1[38];
  if ( v4 )
    j_j___libc_free_0(v4, a1[40] - v4);
  j___libc_free_0(a1[35]);
  v5 = (_QWORD *)a1[26];
  if ( v5 != a1 + 28 )
    j_j___libc_free_0(v5, a1[28] + 1LL);
  v6 = (_QWORD *)a1[21];
  if ( v6 != a1 + 23 )
    j_j___libc_free_0(v6, a1[23] + 1LL);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 416);
}
