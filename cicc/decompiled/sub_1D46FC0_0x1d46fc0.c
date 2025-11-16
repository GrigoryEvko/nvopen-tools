// Function: sub_1D46FC0
// Address: 0x1d46fc0
//
__int64 __fastcall sub_1D46FC0(__int64 a1)
{
  unsigned __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 24);
  qword_4FC1B10[2] = 0;
  if ( v2 != a1 + 40 )
    _libc_free(v2);
  return j_j___libc_free_0(a1, 488);
}
