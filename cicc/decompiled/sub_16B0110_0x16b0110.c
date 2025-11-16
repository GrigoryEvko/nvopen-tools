// Function: sub_16B0110
// Address: 0x16b0110
//
__int64 __fastcall sub_16B0110(__int64 a1)
{
  unsigned __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 96);
  if ( v2 != *(_QWORD *)(a1 + 88) )
    _libc_free(v2);
  return j_j___libc_free_0(a1, 200);
}
