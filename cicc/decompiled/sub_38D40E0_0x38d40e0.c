// Function: sub_38D40E0
// Address: 0x38d40e0
//
__int64 __fastcall sub_38D40E0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // r13

  *a1 = &unk_4A3E0F8;
  v2 = a1[36];
  if ( (_QWORD *)v2 != a1 + 38 )
    _libc_free(v2);
  v3 = a1[33];
  if ( v3 )
  {
    sub_390A9D0(a1[33]);
    j_j___libc_free_0(v3);
  }
  return sub_38DCBC0(a1);
}
