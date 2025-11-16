// Function: sub_1E6BEE0
// Address: 0x1e6bee0
//
__int64 __fastcall sub_1E6BEE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = a1;
  if ( a2 != a1 )
  {
    while ( (unsigned __int16)(**(_WORD **)(result + 16) - 12) <= 1u )
    {
      if ( (*(_BYTE *)result & 4) != 0 )
      {
        result = *(_QWORD *)(result + 8);
        if ( a2 == result )
          return result;
      }
      else
      {
        while ( (*(_BYTE *)(result + 46) & 8) != 0 )
          result = *(_QWORD *)(result + 8);
        result = *(_QWORD *)(result + 8);
        if ( a2 == result )
          return result;
      }
    }
  }
  return result;
}
