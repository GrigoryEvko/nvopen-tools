// Function: sub_1DD6100
// Address: 0x1dd6100
//
__int64 __fastcall sub_1DD6100(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rcx

  result = *(_QWORD *)(a1 + 32);
  v2 = a1 + 24;
  if ( result != a1 + 24 )
  {
    while ( (unsigned __int16)(**(_WORD **)(result + 16) - 12) <= 1u )
    {
      if ( (*(_BYTE *)result & 4) != 0 )
      {
        result = *(_QWORD *)(result + 8);
        if ( v2 == result )
          return result;
      }
      else
      {
        while ( (*(_BYTE *)(result + 46) & 8) != 0 )
          result = *(_QWORD *)(result + 8);
        result = *(_QWORD *)(result + 8);
        if ( v2 == result )
          return result;
      }
    }
  }
  return result;
}
