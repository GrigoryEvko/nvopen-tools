// Function: sub_B92110
// Address: 0xb92110
//
__int64 __fastcall sub_B92110(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int8 v2; // dl
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return 0;
  v1 = sub_B91C10(a1, 28);
  if ( !v1 )
    return 0;
  v2 = *(_BYTE *)(v1 - 16);
  if ( (v2 & 2) != 0 )
    v3 = *(_QWORD *)(v1 - 32);
  else
    v3 = v1 - 8LL * ((v2 >> 2) & 0xF) - 16;
  v4 = *(_QWORD *)(*(_QWORD *)v3 + 136LL);
  result = *(_QWORD *)(v4 + 24);
  if ( *(_DWORD *)(v4 + 32) > 0x40u )
    return *(_QWORD *)result;
  return result;
}
