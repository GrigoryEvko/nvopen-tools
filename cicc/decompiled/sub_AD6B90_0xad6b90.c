// Function: sub_AD6B90
// Address: 0xad6b90
//
__int64 __fastcall sub_AD6B90(__int64 a1, unsigned __int8 (__fastcall *a2)(__int64, __int64), __int64 a3)
{
  __int64 v3; // r14
  int v6; // r14d
  unsigned int v7; // r15d
  __int64 v8; // rsi

  v3 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 > 1 )
    return 0;
  if ( !a2(a3, a1) )
  {
    if ( *(_BYTE *)a1 != 14 && *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) != 18 )
    {
      v6 = *(_DWORD *)(v3 + 32);
      if ( v6 )
      {
        v7 = 0;
        while ( 1 )
        {
          v8 = sub_AD69F0((unsigned __int8 *)a1, v7);
          if ( v8 )
          {
            if ( a2(a3, v8) )
              break;
          }
          if ( ++v7 == v6 )
            return 0;
        }
        return 1;
      }
    }
    return 0;
  }
  return 1;
}
