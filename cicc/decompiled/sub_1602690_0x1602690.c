// Function: sub_1602690
// Address: 0x1602690
//
_QWORD *__fastcall sub_1602690(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // rdx

  v2 = *a1;
  result = *(_QWORD **)(*a1 + 8LL);
  if ( *(_QWORD **)(*a1 + 16LL) == result )
  {
    v6 = &result[*(unsigned int *)(v2 + 28)];
    if ( result == v6 )
    {
LABEL_13:
      result = v6;
    }
    else
    {
      while ( a2 != *result )
      {
        if ( v6 == ++result )
          goto LABEL_13;
      }
    }
  }
  else
  {
    result = (_QWORD *)sub_16CC9F0(*a1, a2);
    if ( a2 == *result )
    {
      v4 = *(_QWORD *)(v2 + 16);
      if ( v4 == *(_QWORD *)(v2 + 8) )
        v5 = *(unsigned int *)(v2 + 28);
      else
        v5 = *(unsigned int *)(v2 + 24);
      v6 = (_QWORD *)(v4 + 8 * v5);
    }
    else
    {
      result = *(_QWORD **)(v2 + 16);
      if ( result != *(_QWORD **)(v2 + 8) )
        return result;
      result += *(unsigned int *)(v2 + 28);
      v6 = result;
    }
  }
  if ( v6 != result )
  {
    *result = -2;
    ++*(_DWORD *)(v2 + 32);
  }
  return result;
}
