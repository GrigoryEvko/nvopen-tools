// Function: sub_CE8560
// Address: 0xce8560
//
__int64 __fastcall sub_CE8560(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r8
  unsigned __int8 v3; // dl
  __int64 v4; // rax
  __int64 v5; // rax

  if ( !*(_QWORD *)(a1 + 48) && (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return 0;
  v1 = sub_B91F50(a1, "nv.used_bytes_mask", 0x12u);
  LODWORD(v2) = 0;
  if ( v1 )
  {
    v3 = *(_BYTE *)(v1 - 16);
    if ( (v3 & 2) != 0 )
      v4 = *(_QWORD *)(v1 - 32);
    else
      v4 = v1 - 8LL * ((v3 >> 2) & 0xF) - 16;
    v5 = *(_QWORD *)(*(_QWORD *)v4 + 136LL);
    v2 = *(_QWORD **)(v5 + 24);
    if ( *(_DWORD *)(v5 + 32) > 0x40u )
      return (unsigned int)*v2;
  }
  return (unsigned int)v2;
}
