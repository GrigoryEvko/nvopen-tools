// Function: sub_C57500
// Address: 0xc57500
//
unsigned __int64 __fastcall sub_C57500(__int64 a1, __int64 *a2)
{
  __int64 *v4; // r13
  __int64 v5; // rcx
  unsigned __int64 result; // rax
  __int64 **v7; // rsi
  __int64 v8; // rdi
  unsigned __int64 v9; // rdx

  if ( a2 != sub_C57470() )
  {
    v4 = **(__int64 ***)(a1 + 72);
    if ( v4 == sub_C57470() )
    {
      result = *(_QWORD *)(a1 + 72);
      *(_QWORD *)result = a2;
      return result;
    }
  }
  v5 = *(unsigned int *)(a1 + 80);
  result = *(_QWORD *)(a1 + 72);
  v7 = (__int64 **)(result + 8 * v5);
  v8 = (8 * v5) >> 3;
  if ( (8 * v5) >> 5 )
  {
    v9 = result + 32 * ((8 * v5) >> 5);
    while ( a2 != *(__int64 **)result )
    {
      if ( a2 == *(__int64 **)(result + 8) )
      {
        result += 8LL;
        break;
      }
      if ( a2 == *(__int64 **)(result + 16) )
      {
        result += 16LL;
        break;
      }
      if ( a2 == *(__int64 **)(result + 24) )
      {
        result += 24LL;
        break;
      }
      result += 32LL;
      if ( v9 == result )
      {
        v8 = (__int64)((__int64)v7 - result) >> 3;
        goto LABEL_13;
      }
    }
LABEL_10:
    if ( v7 != (__int64 **)result )
      return result;
    goto LABEL_17;
  }
LABEL_13:
  if ( v8 != 2 )
  {
    if ( v8 != 3 )
    {
      if ( v8 != 1 )
        goto LABEL_17;
      goto LABEL_16;
    }
    if ( a2 == *(__int64 **)result )
      goto LABEL_10;
    result += 8LL;
  }
  if ( a2 == *(__int64 **)result )
    goto LABEL_10;
  result += 8LL;
LABEL_16:
  if ( a2 == *(__int64 **)result )
    goto LABEL_10;
LABEL_17:
  result = *(unsigned int *)(a1 + 84);
  if ( v5 + 1 > result )
  {
    sub_C8D5F0(a1 + 72, a1 + 88, v5 + 1, 8);
    result = *(_QWORD *)(a1 + 72);
    v7 = (__int64 **)(result + 8LL * *(unsigned int *)(a1 + 80));
  }
  *v7 = a2;
  ++*(_DWORD *)(a1 + 80);
  return result;
}
