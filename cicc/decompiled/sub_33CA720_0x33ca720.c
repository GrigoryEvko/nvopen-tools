// Function: sub_33CA720
// Address: 0x33ca720
//
__int64 __fastcall sub_33CA720(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rsi
  unsigned __int64 v5; // rdx

  result = 0;
  if ( *(_DWORD *)(a1 + 24) == 156 )
  {
    v2 = *(_QWORD *)(a1 + 40);
    v3 = v2 + 40LL * *(unsigned int *)(a1 + 64);
    if ( v2 == v3 )
    {
      return 1;
    }
    else
    {
      v4 = 0x8001000001000LL;
      while ( 1 )
      {
        v5 = *(unsigned int *)(*(_QWORD *)v2 + 24LL);
        if ( (unsigned int)v5 > 0x33 || !_bittest64(&v4, v5) )
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
