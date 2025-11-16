// Function: sub_BC8980
// Address: 0xbc8980
//
__int64 __fastcall sub_BC8980(__int64 a1)
{
  int v1; // ebx

  if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
    v1 = *(_DWORD *)(a1 - 24);
  else
    v1 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  return v1 - (unsigned int)sub_BC8810(a1);
}
