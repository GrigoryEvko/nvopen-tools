// Function: sub_B9DD00
// Address: 0xb9dd00
//
__int64 __fastcall sub_B9DD00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v6; // r8

  if ( a1 && a2 )
  {
    if ( *(_BYTE *)a3 != 85 || *(_BYTE *)a4 != 85 )
      return 0;
    result = *(_QWORD *)(a3 - 32);
    if ( !result )
      return result;
    if ( *(_BYTE *)result )
      return 0;
    v5 = *(_QWORD *)(result + 24);
    result = 0;
    if ( v5 == *(_QWORD *)(a3 + 80) )
    {
      result = *(_QWORD *)(a4 - 32);
      if ( result )
      {
        if ( !*(_BYTE *)result )
        {
          v6 = *(_QWORD *)(result + 24);
          result = 0;
          if ( v6 == *(_QWORD *)(a4 + 80) )
            return sub_B9DAC0(a1, a2, a3);
          return result;
        }
        return 0;
      }
    }
  }
  else
  {
    result = a2;
    if ( a1 )
      return a1;
  }
  return result;
}
