// Function: sub_1A912B0
// Address: 0x1a912b0
//
void *__fastcall sub_1A912B0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = off_49F5D48;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, a1[22] - v2);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
