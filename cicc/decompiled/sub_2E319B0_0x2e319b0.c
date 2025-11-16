// Function: sub_2E319B0
// Address: 0x2e319b0
//
__int64 __fastcall sub_2E319B0(__int64 a1, char a2)
{
  __int64 v4; // rsi
  __int64 result; // rax
  __int16 v6; // dx

  v4 = a1 + 48;
  result = *(_QWORD *)(a1 + 56);
  if ( result != v4 )
  {
    while ( 1 )
    {
      v6 = *(_WORD *)(result + 68);
      if ( (unsigned __int16)(v6 - 14) > 4u && (v6 != 24 || !a2) )
        break;
      if ( (*(_BYTE *)result & 4) != 0 )
      {
        result = *(_QWORD *)(result + 8);
        if ( v4 == result )
          return result;
      }
      else
      {
        while ( (*(_BYTE *)(result + 44) & 8) != 0 )
          result = *(_QWORD *)(result + 8);
        result = *(_QWORD *)(result + 8);
        if ( v4 == result )
          return result;
      }
    }
  }
  return result;
}
