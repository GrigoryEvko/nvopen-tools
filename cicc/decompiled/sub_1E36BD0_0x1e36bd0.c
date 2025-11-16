// Function: sub_1E36BD0
// Address: 0x1e36bd0
//
void *__fastcall sub_1E36BD0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49FBF58;
  v2 = a1[29];
  if ( v2 )
    j_j___libc_free_0(v2, 16);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
