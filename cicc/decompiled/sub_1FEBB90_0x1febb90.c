// Function: sub_1FEBB90
// Address: 0x1febb90
//
_QWORD *__fastcall sub_1FEBB90(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  _QWORD *result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  _QWORD *v7; // rdx

  v2 = *a1;
  v3 = *a2;
  result = *(_QWORD **)(*a1 + 8);
  if ( *(_QWORD **)(*a1 + 16) == result )
  {
    v7 = &result[*(unsigned int *)(v2 + 28)];
    if ( result == v7 )
    {
LABEL_13:
      result = v7;
    }
    else
    {
      while ( v3 != *result )
      {
        if ( v7 == ++result )
          goto LABEL_13;
      }
    }
  }
  else
  {
    result = sub_16CC9F0(*a1, *a2);
    if ( v3 == *result )
    {
      v5 = *(_QWORD *)(v2 + 16);
      if ( v5 == *(_QWORD *)(v2 + 8) )
        v6 = *(unsigned int *)(v2 + 28);
      else
        v6 = *(unsigned int *)(v2 + 24);
      v7 = (_QWORD *)(v5 + 8 * v6);
    }
    else
    {
      result = *(_QWORD **)(v2 + 16);
      if ( result != *(_QWORD **)(v2 + 8) )
        return result;
      result += *(unsigned int *)(v2 + 28);
      v7 = result;
    }
  }
  if ( v7 != result )
  {
    *result = -2;
    ++*(_DWORD *)(v2 + 32);
  }
  return result;
}
