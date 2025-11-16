// Function: sub_20E3080
// Address: 0x20e3080
//
void *__fastcall sub_20E3080(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = off_4A008A0;
  v2 = a1[29];
  if ( v2 )
    j_j___libc_free_0(v2, a1[31] - v2);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
