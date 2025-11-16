// Function: sub_384D000
// Address: 0x384d000
//
void *__fastcall sub_384D000(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A3DCA0;
  v2 = a1[20];
  if ( (_QWORD *)v2 != a1 + 22 )
    j_j___libc_free_0(v2);
  *a1 = &unk_4A3DD58;
  return sub_16366C0(a1);
}
