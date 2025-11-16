// Function: sub_1D470C0
// Address: 0x1d470c0
//
__int64 __fastcall sub_1D470C0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = a1[27];
  qword_4FC1B10[2] = 0;
  if ( (_QWORD *)v2 != a1 + 29 )
    _libc_free(v2);
  v3 = a1[12];
  if ( v3 != a1[11] )
    _libc_free(v3);
  return j_j___libc_free_0(a1, 688);
}
