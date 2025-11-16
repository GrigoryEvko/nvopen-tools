// Function: sub_AA8820
// Address: 0xaa8820
//
_QWORD *__fastcall sub_AA8820(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rcx
  _QWORD *v4; // rdx

  if ( !*(_BYTE *)(a1 + 44) )
    return (_QWORD *)sub_C8CC70(a1 + 16, a2);
  result = *(_QWORD **)(a1 + 24);
  v3 = *(unsigned int *)(a1 + 36);
  v4 = &result[v3];
  if ( result == v4 )
  {
LABEL_7:
    if ( (unsigned int)v3 >= *(_DWORD *)(a1 + 32) )
      return (_QWORD *)sub_C8CC70(a1 + 16, a2);
    *(_DWORD *)(a1 + 36) = v3 + 1;
    *v4 = a2;
    ++*(_QWORD *)(a1 + 16);
  }
  else
  {
    while ( a2 != *result )
    {
      if ( v4 == ++result )
        goto LABEL_7;
    }
  }
  return result;
}
