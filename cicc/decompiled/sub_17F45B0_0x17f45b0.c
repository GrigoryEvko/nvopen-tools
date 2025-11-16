// Function: sub_17F45B0
// Address: 0x17f45b0
//
__int64 __fastcall sub_17F45B0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  _QWORD *v4; // rdi

  *a1 = off_49F06F8;
  v2 = a1[81];
  if ( (_QWORD *)v2 != a1 + 83 )
    _libc_free(v2);
  v3 = a1[59];
  if ( (_QWORD *)v3 != a1 + 61 )
    _libc_free(v3);
  v4 = (_QWORD *)a1[47];
  if ( v4 != a1 + 49 )
    j_j___libc_free_0(v4, a1[49] + 1LL);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 840);
}
