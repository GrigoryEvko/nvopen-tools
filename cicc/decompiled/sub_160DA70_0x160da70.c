// Function: sub_160DA70
// Address: 0x160da70
//
__int64 __fastcall sub_160DA70(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1 + 27;
  v3 = a1[25];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[12];
  if ( v4 != a1[11] )
    _libc_free(v4);
  return j_j___libc_free_0(a1, 608);
}
