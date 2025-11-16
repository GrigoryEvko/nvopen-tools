// Function: sub_1ED8CF0
// Address: 0x1ed8cf0
//
__int64 *__fastcall sub_1ED8CF0(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 *v3; // r8
  unsigned int v4; // r9d
  __int64 *v5; // rcx

  result = *(__int64 **)(a1 + 568);
  if ( *(__int64 **)(a1 + 576) != result )
    return sub_16CCBA0(a1 + 560, a2);
  v3 = &result[*(unsigned int *)(a1 + 588)];
  v4 = *(_DWORD *)(a1 + 588);
  if ( result == v3 )
  {
LABEL_12:
    if ( v4 >= *(_DWORD *)(a1 + 584) )
      return sub_16CCBA0(a1 + 560, a2);
    *(_DWORD *)(a1 + 588) = v4 + 1;
    *v3 = a2;
    ++*(_QWORD *)(a1 + 560);
  }
  else
  {
    v5 = 0;
    while ( a2 != *result )
    {
      if ( *result == -2 )
        v5 = result;
      if ( v3 == ++result )
      {
        if ( !v5 )
          goto LABEL_12;
        *v5 = a2;
        --*(_DWORD *)(a1 + 592);
        ++*(_QWORD *)(a1 + 560);
        return result;
      }
    }
  }
  return result;
}
