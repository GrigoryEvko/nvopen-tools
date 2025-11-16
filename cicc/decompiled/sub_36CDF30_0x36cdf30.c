// Function: sub_36CDF30
// Address: 0x36cdf30
//
__int64 __fastcall sub_36CDF30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v8; // rdx
  __int64 result; // rax
  _BYTE *v10; // rdi

  v5 = *(_QWORD *)(a1 - 32);
  if ( !v5 || *(_BYTE *)v5 || (v6 = *(_QWORD *)(a1 + 80), *(_QWORD *)(v5 + 24) != v6) )
    BUG();
  v8 = *(unsigned int *)(v5 + 36);
  if ( (unsigned int)v8 <= 0x2342 )
  {
    result = 3;
    if ( (unsigned int)v8 > 0x2340 )
      return result;
    if ( (_DWORD)v8 == 8604 )
    {
      v10 = *(_BYTE **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      if ( *v10 == 17 && sub_AD7890((__int64)v10, a2, v8, v6, a5) )
        return 6;
    }
    else if ( (_DWORD)v8 == 8605 )
    {
      return 4294967292LL;
    }
    return sub_36CDE90(a1) == 0 ? -7 : -3;
  }
  result = 1;
  if ( (unsigned int)(v8 - 9027) > 1 )
    return sub_36CDE90(a1) == 0 ? -7 : -3;
  return result;
}
