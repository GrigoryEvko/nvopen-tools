// Function: sub_2997170
// Address: 0x2997170
//
__int64 __fastcall sub_2997170(__int64 a1)
{
  __int64 v3; // rdx

  while ( 1 )
  {
    if ( !a1 )
      BUG();
    if ( *(_BYTE *)(a1 - 24) != 85 )
      break;
    v3 = *(_QWORD *)(a1 - 56);
    if ( !v3
      || *(_BYTE *)v3
      || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a1 + 56)
      || (*(_BYTE *)(v3 + 33) & 0x20) == 0
      || (unsigned int)(*(_DWORD *)(v3 + 36) - 68) > 3 )
    {
      break;
    }
    a1 = *(_QWORD *)(a1 + 8);
  }
  return a1 - 24;
}
