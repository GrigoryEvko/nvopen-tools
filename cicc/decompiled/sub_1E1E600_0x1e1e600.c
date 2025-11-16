// Function: sub_1E1E600
// Address: 0x1e1e600
//
void __fastcall sub_1E1E600(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = a1[13];
  if ( v2 )
    j_j___libc_free_0(v2, a1[15] - v2);
  v3 = a1[2];
  if ( v3 != a1[1] )
    _libc_free(v3);
}
