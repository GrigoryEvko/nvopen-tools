// Function: sub_2EC2050
// Address: 0x2ec2050
//
__int64 __fastcall sub_2EC2050(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int16 v3; // dx

  result = a1;
  if ( a1 != a2 )
  {
    while ( 1 )
    {
      v3 = *(_WORD *)(result + 68);
      if ( (unsigned __int16)(v3 - 14) > 4u && v3 != 24 )
        break;
      if ( (*(_BYTE *)result & 4) != 0 )
      {
        result = *(_QWORD *)(result + 8);
        if ( a2 == result )
          return result;
      }
      else
      {
        while ( (*(_BYTE *)(result + 44) & 8) != 0 )
          result = *(_QWORD *)(result + 8);
        result = *(_QWORD *)(result + 8);
        if ( a2 == result )
          return result;
      }
    }
  }
  return result;
}
