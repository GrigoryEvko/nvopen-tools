// Function: sub_20FF870
// Address: 0x20ff870
//
__int64 __fastcall sub_20FF870(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *(_QWORD *)(a1[3] + 8LL) = 0;
  v2 = a1[21];
  if ( v2 != a1[20] )
    _libc_free(v2);
  v3 = a1[12];
  if ( v3 != a1[11] )
    _libc_free(v3);
  return j_j___libc_free_0(a1, 224);
}
