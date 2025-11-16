// Function: sub_CA40B0
// Address: 0xca40b0
//
__int64 __fastcall sub_CA40B0(__int64 a1, __int64 a2)
{
  bool v3; // zf
  __int64 v5; // rdi
  __int64 v6; // rdi

  v3 = *(_BYTE *)(a1 + 328) == 0;
  *(_QWORD *)a1 = off_49DCB98;
  if ( !v3 )
  {
    *(_BYTE *)(a1 + 328) = 0;
    if ( (*(_BYTE *)(a1 + 320) & 1) == 0 )
    {
      v5 = *(_QWORD *)(a1 + 168);
      if ( v5 != a1 + 192 )
        _libc_free(v5, a2);
      v6 = *(_QWORD *)(a1 + 16);
      if ( v6 != a1 + 40 )
        _libc_free(v6, a2);
    }
  }
  nullsub_170();
  return j_j___libc_free_0(a1, 336);
}
