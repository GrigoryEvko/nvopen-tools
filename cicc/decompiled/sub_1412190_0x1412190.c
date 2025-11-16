// Function: sub_1412190
// Address: 0x1412190
//
_QWORD *__fastcall sub_1412190(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // r8
  unsigned int v4; // r9d
  _QWORD *v5; // rcx
  int v6; // eax

  result = *(_QWORD **)(a1 + 8);
  if ( result != *(_QWORD **)(a1 + 16) )
    return (_QWORD *)sub_16CCBA0(a1, a2);
  v3 = &result[*(unsigned int *)(a1 + 28)];
  v4 = *(_DWORD *)(a1 + 28);
  if ( result == v3 )
  {
LABEL_12:
    if ( v4 >= *(_DWORD *)(a1 + 24) )
      return (_QWORD *)sub_16CCBA0(a1, a2);
    *(_DWORD *)(a1 + 28) = v4 + 1;
    *v3 = a2;
    v6 = *(_DWORD *)(a1 + 28);
    ++*(_QWORD *)a1;
    return (_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * (unsigned int)(v6 - 1));
  }
  else
  {
    v5 = 0;
    while ( *result != a2 )
    {
      if ( *result == -2 )
        v5 = result;
      if ( ++result == v3 )
      {
        if ( !v5 )
          goto LABEL_12;
        *v5 = a2;
        --*(_DWORD *)(a1 + 32);
        ++*(_QWORD *)a1;
        return v5;
      }
    }
  }
  return result;
}
