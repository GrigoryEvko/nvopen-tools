// Function: sub_2113C50
// Address: 0x2113c50
//
void *__fastcall sub_2113C50(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = off_4A010C0;
  v2 = a1[23];
  if ( v2 )
    j_j___libc_free_0(v2, a1[25] - v2);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
