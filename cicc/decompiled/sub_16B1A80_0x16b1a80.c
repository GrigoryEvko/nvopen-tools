// Function: sub_16B1A80
// Address: 0x16b1a80
//
_QWORD *__fastcall sub_16B1A80(__int64 a1)
{
  __int64 v2; // rdi
  _QWORD *result; // rax
  _QWORD *v4; // rsi
  unsigned int v5; // r8d
  _QWORD *v6; // rcx

  if ( !qword_4FA01E0 )
    sub_16C1EA0(&qword_4FA01E0, sub_16B89A0, sub_16B0D50);
  v2 = qword_4FA01E0;
  result = *(_QWORD **)(qword_4FA01E0 + 80);
  if ( *(_QWORD **)(qword_4FA01E0 + 88) != result )
    return (_QWORD *)sub_16CCBA0(qword_4FA01E0 + 72, a1);
  v4 = &result[*(unsigned int *)(qword_4FA01E0 + 100)];
  v5 = *(_DWORD *)(qword_4FA01E0 + 100);
  if ( result == v4 )
  {
LABEL_14:
    if ( v5 >= *(_DWORD *)(qword_4FA01E0 + 96) )
      return (_QWORD *)sub_16CCBA0(qword_4FA01E0 + 72, a1);
    *(_DWORD *)(qword_4FA01E0 + 100) = v5 + 1;
    *v4 = a1;
    ++*(_QWORD *)(v2 + 72);
  }
  else
  {
    v6 = 0;
    while ( a1 != *result )
    {
      if ( *result == -2 )
        v6 = result;
      if ( v4 == ++result )
      {
        if ( !v6 )
          goto LABEL_14;
        *v6 = a1;
        --*(_DWORD *)(v2 + 104);
        ++*(_QWORD *)(v2 + 72);
        return result;
      }
    }
  }
  return result;
}
