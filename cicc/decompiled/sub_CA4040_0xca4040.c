// Function: sub_CA4040
// Address: 0xca4040
//
void __fastcall sub_CA4040(__int64 a1, __int64 a2)
{
  bool v3; // zf
  __int64 v4; // rdi
  __int64 v5; // rdi

  v3 = *(_BYTE *)(a1 + 328) == 0;
  *(_QWORD *)a1 = off_49DCB98;
  if ( !v3 )
  {
    *(_BYTE *)(a1 + 328) = 0;
    if ( (*(_BYTE *)(a1 + 320) & 1) == 0 )
    {
      v4 = *(_QWORD *)(a1 + 168);
      if ( v4 != a1 + 192 )
        _libc_free(v4, a2);
      v5 = *(_QWORD *)(a1 + 16);
      if ( v5 != a1 + 40 )
        _libc_free(v5, a2);
    }
  }
  nullsub_170();
}
