// Function: sub_16CCF00
// Address: 0x16ccf00
//
__int64 __fastcall sub_16CCF00(__int64 a1, int a2, __int64 a3)
{
  unsigned __int64 v5; // rdi

  v5 = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 8) != v5 )
    _libc_free(v5);
  return sub_16CCE60(a1, a2, a3);
}
