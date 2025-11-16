// Function: sub_1CEB1D0
// Address: 0x1ceb1d0
//
void *__fastcall sub_1CEB1D0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi

  *a1 = off_49F8E60;
  v2 = a1[23];
  if ( v2 )
    j_j___libc_free_0(v2, a1[25] - v2);
  v3 = a1[20];
  if ( v3 )
    j_j___libc_free_0(v3, a1[22] - v3);
  return sub_1636790(a1);
}
