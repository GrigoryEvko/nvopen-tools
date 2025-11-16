// Function: sub_7E0590
// Address: 0x7e0590
//
__int16 __fastcall sub_7E0590(__int64 a1)
{
  __int16 result; // ax
  bool v2; // zf
  __int64 v3; // rdx

  result = *(unsigned __int8 *)(a1 + 25);
  if ( (result & 2) != 0 )
  {
    result = result & 0xFFFC | 1;
    v2 = *(_BYTE *)(a1 + 24) == 1;
    *(_BYTE *)(a1 + 25) = result;
    if ( v2 )
    {
      result = *(unsigned __int8 *)(a1 + 56) - 94;
      if ( ((*(_BYTE *)(a1 + 56) - 94) & 0xFD) == 0 )
      {
        v3 = *(_QWORD *)(a1 + 72);
        result = *(_WORD *)(v3 + 24) & 0x1FF;
        if ( result == 5 )
          *(_BYTE *)(v3 + 25) |= 1u;
      }
    }
  }
  return result;
}
