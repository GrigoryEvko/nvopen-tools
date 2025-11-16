// Function: sub_1F5AE50
// Address: 0x1f5ae50
//
__int64 __fastcall sub_1F5AE50(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = a1 + 41;
  v3 = a1[39];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[36];
  if ( (_QWORD *)v4 != a1 + 38 )
    _libc_free(v4);
  v5 = a1[33];
  if ( (_QWORD *)v5 != a1 + 35 )
    _libc_free(v5);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 336);
}
