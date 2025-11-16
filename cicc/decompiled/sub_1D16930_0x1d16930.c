// Function: sub_1D16930
// Address: 0x1d16930
//
__int64 __fastcall sub_1D16930(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rsi
  unsigned __int64 v5; // rdx

  result = 0;
  if ( *(_WORD *)(a1 + 24) == 104 )
  {
    v2 = *(_QWORD *)(a1 + 32);
    v3 = v2 + 40LL * *(unsigned int *)(a1 + 56);
    if ( v2 == v3 )
    {
      return 1;
    }
    else
    {
      v4 = 0x1000200000800LL;
      while ( 1 )
      {
        v5 = *(unsigned __int16 *)(*(_QWORD *)v2 + 24LL);
        if ( (unsigned __int16)v5 > 0x30u || !_bittest64(&v4, v5) )
          break;
        v2 += 40;
        if ( v3 == v2 )
          return 1;
      }
      return 0;
    }
  }
  return result;
}
