// Function: sub_B6E750
// Address: 0xb6e750
//
_QWORD *__fastcall sub_B6E750(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  _QWORD *result; // rax
  __int64 v4; // rcx
  _QWORD *v5; // rdx

  v2 = *a1;
  if ( !*(_BYTE *)(v2 + 28) )
    return (_QWORD *)sub_C8CC70(v2, a2);
  result = *(_QWORD **)(v2 + 8);
  v4 = *(unsigned int *)(v2 + 20);
  v5 = &result[v4];
  if ( result == v5 )
  {
LABEL_7:
    if ( (unsigned int)v4 >= *(_DWORD *)(v2 + 16) )
      return (_QWORD *)sub_C8CC70(v2, a2);
    *(_DWORD *)(v2 + 20) = v4 + 1;
    *v5 = a2;
    ++*(_QWORD *)v2;
  }
  else
  {
    while ( a2 != *result )
    {
      if ( v5 == ++result )
        goto LABEL_7;
    }
  }
  return result;
}
