// Function: sub_1C4A3F0
// Address: 0x1c4a3f0
//
__int64 __fastcall sub_1C4A3F0(_QWORD *a1)
{
  unsigned __int64 *v1; // r13
  unsigned __int64 v3; // rdi
  __int64 v4; // rdi

  v1 = (unsigned __int64 *)a1[6];
  *a1 = off_49854A8;
  if ( v1 )
  {
    if ( (unsigned __int64 *)*v1 != v1 + 2 )
      _libc_free(*v1);
    j_j___libc_free_0(v1, 32);
  }
  j___libc_free_0(a1[11]);
  v3 = a1[7];
  if ( (_QWORD *)v3 != a1 + 9 )
    _libc_free(v3);
  v4 = a1[2];
  *a1 = off_4985448;
  j___libc_free_0(v4);
  return j_j___libc_free_0(a1, 112);
}
