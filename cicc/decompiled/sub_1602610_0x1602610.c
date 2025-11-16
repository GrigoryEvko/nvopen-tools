// Function: sub_1602610
// Address: 0x1602610
//
_QWORD *__fastcall sub_1602610(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  _QWORD *result; // rax
  _QWORD *v4; // r8
  unsigned int v5; // r9d
  _QWORD *v6; // rcx

  v2 = *a1;
  result = *(_QWORD **)(v2 + 8);
  if ( *(_QWORD **)(v2 + 16) != result )
    return (_QWORD *)sub_16CCBA0(v2, a2);
  v4 = &result[*(unsigned int *)(v2 + 28)];
  v5 = *(_DWORD *)(v2 + 28);
  if ( result == v4 )
  {
LABEL_12:
    if ( v5 >= *(_DWORD *)(v2 + 24) )
      return (_QWORD *)sub_16CCBA0(v2, a2);
    *(_DWORD *)(v2 + 28) = v5 + 1;
    *v4 = a2;
    ++*(_QWORD *)v2;
  }
  else
  {
    v6 = 0;
    while ( a2 != *result )
    {
      if ( *result == -2 )
        v6 = result;
      if ( v4 == ++result )
      {
        if ( !v6 )
          goto LABEL_12;
        *v6 = a2;
        --*(_DWORD *)(v2 + 32);
        ++*(_QWORD *)v2;
        return result;
      }
    }
  }
  return result;
}
