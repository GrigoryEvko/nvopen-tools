// Function: sub_B3A260
// Address: 0xb3a260
//
__int64 __fastcall sub_B3A260(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49DA260;
  v2 = (_QWORD *)a1[23];
  if ( v2 != a1 + 25 )
    j_j___libc_free_0(v2, a1[25] + 1LL);
  return sub_BB9260(a1);
}
