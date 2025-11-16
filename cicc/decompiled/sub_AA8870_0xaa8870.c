// Function: sub_AA8870
// Address: 0xaa8870
//
_QWORD *__fastcall sub_AA8870(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // rdi
  _QWORD *v6; // rdx
  _QWORD *result; // rax
  __int64 v8; // rcx

  if ( *(_BYTE *)(a1 + 44) )
  {
    v5 = *(_QWORD **)(a1 + 24);
    v6 = &v5[*(unsigned int *)(a1 + 36)];
    result = v5;
    if ( v5 != v6 )
    {
      while ( a2 != *result )
      {
        if ( v6 == ++result )
          return result;
      }
      v8 = (unsigned int)(*(_DWORD *)(a1 + 36) - 1);
      *(_DWORD *)(a1 + 36) = v8;
      *result = v5[v8];
      ++*(_QWORD *)(a1 + 16);
    }
  }
  else
  {
    result = (_QWORD *)sub_C8CA60(a1 + 16, a2, a3, a4);
    if ( result )
    {
      *result = -2;
      ++*(_DWORD *)(a1 + 40);
      ++*(_QWORD *)(a1 + 16);
    }
  }
  return result;
}
