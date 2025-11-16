// Function: sub_1A7A220
// Address: 0x1a7a220
//
__int64 __fastcall sub_1A7A220(__int64 a1)
{
  __int64 v3; // rdx

  while ( 1 )
  {
    if ( !a1 )
      BUG();
    if ( *(_BYTE *)(a1 - 8) != 78 )
      break;
    v3 = *(_QWORD *)(a1 - 48);
    if ( *(_BYTE *)(v3 + 16) || (*(_BYTE *)(v3 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v3 + 36) - 35) > 3 )
      break;
    a1 = *(_QWORD *)(a1 + 8);
  }
  return a1 - 24;
}
