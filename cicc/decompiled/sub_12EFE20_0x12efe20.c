// Function: sub_12EFE20
// Address: 0x12efe20
//
__int64 __fastcall sub_12EFE20(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rdi
  __int64 v5; // rdi

  v3 = a1 + 27;
  v4 = (_QWORD *)a1[25];
  if ( v4 != v3 )
    _libc_free(v4, a2);
  v5 = a1[12];
  if ( v5 != a1[11] )
    _libc_free(v5, a2);
  return j_j___libc_free_0(a1, 608);
}
