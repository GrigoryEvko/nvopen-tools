// Function: sub_1D5AB30
// Address: 0x1d5ab30
//
__int64 __fastcall sub_1D5AB30(_QWORD *a1)
{
  __int64 v1; // r13
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v1 = a1[12];
  *a1 = off_49856A8;
  if ( v1 )
  {
    v3 = *(_QWORD *)(v1 + 16);
    if ( v3 != v1 + 32 )
      _libc_free(v3);
    j_j___libc_free_0(v1, 96);
  }
  v4 = a1[6];
  if ( (_QWORD *)v4 != a1 + 8 )
    _libc_free(v4);
  return j_j___libc_free_0(a1, 112);
}
