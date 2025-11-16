// Function: sub_16C91C0
// Address: 0x16c91c0
//
__int64 __fastcall sub_16C91C0(__int64 a1)
{
  unsigned __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 96);
  if ( v2 != *(_QWORD *)(a1 + 88) )
    _libc_free(v2);
  return j_j___libc_free_0(a1, 208);
}
