// Function: sub_B3A310
// Address: 0xb3a310
//
__int64 __fastcall sub_B3A310(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49DA260;
  v2 = (_QWORD *)a1[23];
  if ( v2 != a1 + 25 )
    j_j___libc_free_0(v2, a1[25] + 1LL);
  sub_BB9260(a1);
  return j_j___libc_free_0(a1, 224);
}
