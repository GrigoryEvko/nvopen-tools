// Function: sub_B2B6F0
// Address: 0xb2b6f0
//
__int64 __fastcall sub_B2B6F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = 0;
  if ( a2 )
  {
    if ( a2 != 4 )
    {
      if ( a2 == 13 )
      {
        if ( *(_QWORD *)a1 == 0x6576726573657270LL && *(_DWORD *)(a1 + 8) == 1734964013 && *(_BYTE *)(a1 + 12) == 110 )
          return 1;
        if ( *(_QWORD *)a1 == 0x6576697469736F70LL && *(_DWORD *)(a1 + 8) == 1919253037 && *(_BYTE *)(a1 + 12) == 111 )
          return 2;
      }
      else if ( a2 == 7 && *(_DWORD *)a1 == 1634629988 && *(_WORD *)(a1 + 4) == 26989 && *(_BYTE *)(a1 + 6) == 99 )
      {
        return 3;
      }
      return 0xFFFFFFFFLL;
    }
    if ( *(_DWORD *)a1 != 1701143913 )
      return 0xFFFFFFFFLL;
  }
  return result;
}
