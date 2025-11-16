// Function: sub_18022E0
// Address: 0x18022e0
//
__int64 __fastcall sub_18022E0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49F0848;
  v2 = (_QWORD *)a1[28];
  if ( v2 != a1 + 30 )
    j_j___libc_free_0(v2, a1[30] + 1LL);
  j___libc_free_0(a1[22]);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 384);
}
