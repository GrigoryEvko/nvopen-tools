// Function: sub_B90600
// Address: 0xb90600
//
__int64 __fastcall sub_B90600(__int64 a1)
{
  unsigned __int8 v1; // al

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
  {
    if ( *(_DWORD *)(a1 - 24) > 9u )
      return *(_QWORD *)(*(_QWORD *)(a1 - 32) + 72LL);
    return 0;
  }
  if ( ((*(_WORD *)(a1 - 16) >> 6) & 0xFu) <= 9 )
    return 0;
  return *(_QWORD *)(a1 - 16 - 8LL * ((v1 >> 2) & 0xF) + 72);
}
