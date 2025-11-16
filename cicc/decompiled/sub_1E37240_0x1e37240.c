// Function: sub_1E37240
// Address: 0x1e37240
//
void *__fastcall sub_1E37240(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = off_49FC080;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, a1[22] - v2);
  return sub_1636790(a1);
}
