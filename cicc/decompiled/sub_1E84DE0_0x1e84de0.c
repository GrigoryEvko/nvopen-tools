// Function: sub_1E84DE0
// Address: 0x1e84de0
//
void *__fastcall sub_1E84DE0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49FCE40;
  v2 = (_QWORD *)a1[29];
  if ( v2 != a1 + 31 )
    j_j___libc_free_0(v2, a1[31] + 1LL);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
