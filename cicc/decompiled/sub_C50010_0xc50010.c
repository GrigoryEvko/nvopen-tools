// Function: sub_C50010
// Address: 0xc50010
//
__int64 __fastcall sub_C50010(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi

  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104), a2);
  v3 = *(_QWORD *)(a1 + 72);
  if ( v3 != a1 + 88 )
    _libc_free(v3, a2);
  return j_j___libc_free_0(a1, 144);
}
