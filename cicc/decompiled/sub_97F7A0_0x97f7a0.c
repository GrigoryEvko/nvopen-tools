// Function: sub_97F7A0
// Address: 0x97f7a0
//
__int64 __fastcall sub_97F7A0(__int64 a1, int a2, __int64 a3, int *a4)
{
  __int64 result; // rax
  char v5; // dl
  int v6; // eax

  result = 0;
  if ( a2 == 24 )
  {
    v5 = *(_BYTE *)(a3 + 8);
    if ( v5 == 3 )
    {
      v6 = 276;
    }
    else
    {
      if ( v5 != 2 )
        return result;
      v6 = 277;
    }
    *a4 = v6;
    return 1;
  }
  return result;
}
