// Function: sub_2CDD760
// Address: 0x2cdd760
//
__int64 __fastcall sub_2CDD760(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v5; // rax

  while ( 1 )
  {
    v3 = *(unsigned __int16 *)(a1 + 2);
    if ( v3 == 50 )
      break;
    if ( v3 != 34 && v3 != 49 )
      return 0;
    a1 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    if ( *(_BYTE *)a1 != 5 )
    {
      if ( *(_BYTE *)a1 != 22 || !(unsigned __int8)sub_CE9220(a2) || !unk_50142AD )
        return 0;
      return (unsigned __int8)sub_B2D680(a1) ^ 1u;
    }
  }
  v5 = *(_QWORD *)(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + 8LL);
  if ( *(_BYTE *)(v5 + 8) != 14 )
    return 0;
  return *(_DWORD *)(v5 + 8) >> 8;
}
