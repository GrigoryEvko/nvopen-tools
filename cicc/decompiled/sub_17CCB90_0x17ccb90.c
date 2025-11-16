// Function: sub_17CCB90
// Address: 0x17ccb90
//
__int64 __fastcall sub_17CCB90(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 64;
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 != v2 )
    _libc_free(v3);
  return j_j___libc_free_0(a1, 192);
}
