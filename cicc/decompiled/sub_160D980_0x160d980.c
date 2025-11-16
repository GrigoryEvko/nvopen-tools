// Function: sub_160D980
// Address: 0x160d980
//
__int64 __fastcall sub_160D980(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 32;
  v3 = *(_QWORD *)(a1 + 16);
  if ( v3 != v2 )
    _libc_free(v3);
  return j_j___libc_free_0(a1, 416);
}
