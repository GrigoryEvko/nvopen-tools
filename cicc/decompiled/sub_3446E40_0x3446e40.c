// Function: sub_3446E40
// Address: 0x3446e40
//
__int64 __fastcall sub_3446E40(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rsi
  __int64 v3; // rdx
  char v4; // cl
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rax
  __int64 v7; // rdx

  result = *(unsigned int *)(a1 + 8);
  v2 = *(_QWORD *)a1;
  v3 = 1LL << ((unsigned __int8)result - 1);
  if ( (unsigned int)result > 0x40 )
  {
    if ( (*(_QWORD *)(v2 + 8LL * ((unsigned int)(result - 1) >> 6)) & v3) != 0 )
      return sub_C44500(a1);
  }
  else if ( (v3 & v2) != 0 )
  {
    goto LABEL_3;
  }
  result = *(unsigned int *)(a1 + 24);
  v2 = *(_QWORD *)(a1 + 16);
  v7 = 1LL << ((unsigned __int8)result - 1);
  if ( (unsigned int)result > 0x40 )
  {
    if ( (*(_QWORD *)(v2 + 8LL * ((unsigned int)(result - 1) >> 6)) & v7) == 0 )
      return 1;
    a1 += 16;
    return sub_C44500(a1);
  }
  if ( (v7 & v2) == 0 )
    return 1;
LABEL_3:
  if ( (_DWORD)result )
  {
    v4 = 64 - result;
    result = 64;
    v5 = ~(v2 << v4);
    if ( v5 )
    {
      _BitScanReverse64(&v6, v5);
      return (unsigned int)v6 ^ 0x3F;
    }
  }
  return result;
}
