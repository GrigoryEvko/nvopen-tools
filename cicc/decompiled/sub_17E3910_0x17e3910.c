// Function: sub_17E3910
// Address: 0x17e3910
//
__int64 __fastcall sub_17E3910(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49F0580;
  v2 = (_QWORD *)a1[20];
  if ( v2 != a1 + 22 )
    j_j___libc_free_0(v2, a1[22] + 1LL);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 192);
}
