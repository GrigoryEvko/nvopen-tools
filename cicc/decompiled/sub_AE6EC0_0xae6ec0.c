// Function: sub_AE6EC0
// Address: 0xae6ec0
//
_QWORD *__fastcall sub_AE6EC0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rcx
  _QWORD *v4; // rdx
  int v5; // eax

  if ( !*(_BYTE *)(a1 + 28) )
    return (_QWORD *)sub_C8CC70(a1, a2);
  result = *(_QWORD **)(a1 + 8);
  v3 = *(unsigned int *)(a1 + 20);
  v4 = &result[v3];
  if ( result == v4 )
  {
LABEL_7:
    if ( (unsigned int)v3 >= *(_DWORD *)(a1 + 16) )
      return (_QWORD *)sub_C8CC70(a1, a2);
    *(_DWORD *)(a1 + 20) = v3 + 1;
    *v4 = a2;
    v5 = *(_DWORD *)(a1 + 20);
    ++*(_QWORD *)a1;
    return (_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * (unsigned int)(v5 - 1));
  }
  else
  {
    while ( *result != a2 )
    {
      if ( ++result == v4 )
        goto LABEL_7;
    }
  }
  return result;
}
