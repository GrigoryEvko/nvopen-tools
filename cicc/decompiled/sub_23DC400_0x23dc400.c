// Function: sub_23DC400
// Address: 0x23dc400
//
bool __fastcall sub_23DC400(__int64 a1)
{
  if ( (unsigned int)*(unsigned __int8 *)(a1 + 8) - 17 <= 1 )
  {
    a1 = **(_QWORD **)(a1 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(a1 + 8) - 17 <= 1 )
      a1 = **(_QWORD **)(a1 + 16);
  }
  return (((*(_DWORD *)(a1 + 8) >> 8) - 3) & 0xFFFFFFFD) == 0;
}
