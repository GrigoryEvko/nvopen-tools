// Function: sub_EA1630
// Address: 0xea1630
//
__int64 __fastcall sub_EA1630(__int64 a1)
{
  unsigned int v1; // r8d

  v1 = 0;
  if ( (unsigned __int16)((*(_WORD *)(a1 + 12) & 7) - 1) <= 6u )
    return (unsigned int)dword_3F82A80[(unsigned __int16)((*(_WORD *)(a1 + 12) & 7) - 1)];
  return v1;
}
