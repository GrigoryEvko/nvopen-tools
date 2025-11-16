// Function: sub_2E85500
// Address: 0x2e85500
//
__int64 __fastcall sub_2E85500(__int64 a1, __int64 a2, int a3)
{
  signed __int64 v3; // rax
  __int64 v4; // rax
  __int64 result; // rax

  v3 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 3);
  if ( v3 >> 2 <= 0 )
  {
LABEL_13:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        if ( v3 != 1 )
          return a2;
LABEL_28:
        result = a2;
        if ( !*(_BYTE *)a1 && a3 == *(_DWORD *)(a1 + 8) )
          return a1;
        return result;
      }
      if ( !*(_BYTE *)a1 )
      {
        result = a1;
        if ( a3 == *(_DWORD *)(a1 + 8) )
          return result;
      }
      a1 += 40;
    }
    if ( !*(_BYTE *)a1 )
    {
      result = a1;
      if ( a3 == *(_DWORD *)(a1 + 8) )
        return result;
    }
    a1 += 40;
    goto LABEL_28;
  }
  v4 = a1 + 160 * (v3 >> 2);
  while ( 1 )
  {
    if ( !*(_BYTE *)a1 && a3 == *(_DWORD *)(a1 + 8) )
      return a1;
    if ( !*(_BYTE *)(a1 + 40) && a3 == *(_DWORD *)(a1 + 48) )
      return a1 + 40;
    if ( !*(_BYTE *)(a1 + 80) && a3 == *(_DWORD *)(a1 + 88) )
      return a1 + 80;
    if ( !*(_BYTE *)(a1 + 120) && a3 == *(_DWORD *)(a1 + 128) )
      return a1 + 120;
    a1 += 160;
    if ( a1 == v4 )
    {
      v3 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 3);
      goto LABEL_13;
    }
  }
}
