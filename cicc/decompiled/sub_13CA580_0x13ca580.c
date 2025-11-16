// Function: sub_13CA580
// Address: 0x13ca580
//
_QWORD *__fastcall sub_13CA580(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // r8
  unsigned int v4; // r9d
  _QWORD *v5; // rcx

  result = *(_QWORD **)(a1 + 88);
  if ( *(_QWORD **)(a1 + 96) != result )
    return (_QWORD *)sub_16CCBA0(a1 + 80, a2);
  v3 = &result[*(unsigned int *)(a1 + 108)];
  v4 = *(_DWORD *)(a1 + 108);
  if ( result == v3 )
  {
LABEL_12:
    if ( v4 >= *(_DWORD *)(a1 + 104) )
      return (_QWORD *)sub_16CCBA0(a1 + 80, a2);
    *(_DWORD *)(a1 + 108) = v4 + 1;
    *v3 = a2;
    ++*(_QWORD *)(a1 + 80);
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
        --*(_DWORD *)(a1 + 112);
        ++*(_QWORD *)(a1 + 80);
        return result;
      }
    }
  }
  return result;
}
