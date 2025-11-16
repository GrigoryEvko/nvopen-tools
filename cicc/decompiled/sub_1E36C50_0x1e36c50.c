// Function: sub_1E36C50
// Address: 0x1e36c50
//
__int64 __fastcall sub_1E36C50(_QWORD *a1)
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
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 240);
}
