// Function: sub_183B4B0
// Address: 0x183b4b0
//
__int64 __fastcall sub_183B4B0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi

  v2 = a1[15];
  if ( v2 != a1[14] )
    _libc_free(v2);
  v3 = a1[10];
  *a1 = &unk_49F0D08;
  if ( v3 )
    j_j___libc_free_0(v3, a1[12] - v3);
  v4 = a1[6];
  if ( v4 )
    j_j___libc_free_0(v4, a1[8] - v4);
  v5 = a1[2];
  if ( v5 )
    j_j___libc_free_0(v5, a1[4] - v5);
  return j_j___libc_free_0(a1, 400);
}
