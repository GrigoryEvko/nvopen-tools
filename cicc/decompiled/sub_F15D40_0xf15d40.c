// Function: sub_F15D40
// Address: 0xf15d40
//
__int64 __fastcall sub_F15D40(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rcx

  result = *(_QWORD *)(a1 + 16);
  v2 = *(_QWORD *)(a1 + 24);
  if ( result != v2 )
  {
    while ( 1 )
    {
      if ( *(_QWORD *)result == -4096 )
      {
        if ( *(_QWORD *)(result + 8) || *(_BYTE *)(result + 32) )
          return result;
      }
      else if ( *(_QWORD *)result != -8192
             || *(_QWORD *)(result + 8)
             || !*(_BYTE *)(result + 32)
             || *(_QWORD *)(result + 16)
             || *(_QWORD *)(result + 24) )
      {
        return result;
      }
      if ( !*(_QWORD *)(result + 40) )
      {
        result += 56;
        *(_QWORD *)(a1 + 16) = result;
        if ( result != v2 )
          continue;
      }
      return result;
    }
  }
  return result;
}
