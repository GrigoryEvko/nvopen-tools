// Function: sub_1B04D00
// Address: 0x1b04d00
//
_QWORD *__fastcall sub_1B04D00(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  _QWORD *v5; // rdx

  result = *(_QWORD **)(a1 + 8);
  if ( *(_QWORD **)(a1 + 16) == result )
  {
    v5 = &result[*(unsigned int *)(a1 + 28)];
    if ( result == v5 )
    {
LABEL_13:
      result = v5;
    }
    else
    {
      while ( a2 != *result )
      {
        if ( v5 == ++result )
          goto LABEL_13;
      }
    }
  }
  else
  {
    result = sub_16CC9F0(a1, a2);
    if ( a2 == *result )
    {
      v3 = *(_QWORD *)(a1 + 16);
      if ( v3 == *(_QWORD *)(a1 + 8) )
        v4 = *(unsigned int *)(a1 + 28);
      else
        v4 = *(unsigned int *)(a1 + 24);
      v5 = (_QWORD *)(v3 + 8 * v4);
    }
    else
    {
      result = *(_QWORD **)(a1 + 16);
      if ( result != *(_QWORD **)(a1 + 8) )
        return result;
      result += *(unsigned int *)(a1 + 28);
      v5 = result;
    }
  }
  if ( v5 != result )
  {
    *result = -2;
    ++*(_DWORD *)(a1 + 32);
  }
  return result;
}
