// Function: sub_33259A0
// Address: 0x33259a0
//
__int64 *__fastcall sub_33259A0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rsi
  __int64 *v4; // rdi
  __int64 *v5; // rdx
  __int64 *result; // rax
  __int64 v7; // rcx

  v2 = *a1;
  v3 = *a2;
  if ( *(_BYTE *)(*a1 + 28) )
  {
    v4 = *(__int64 **)(v2 + 8);
    v5 = &v4[*(unsigned int *)(v2 + 20)];
    result = v4;
    if ( v4 != v5 )
    {
      while ( v3 != *result )
      {
        if ( v5 == ++result )
          return result;
      }
      v7 = (unsigned int)(*(_DWORD *)(v2 + 20) - 1);
      *(_DWORD *)(v2 + 20) = v7;
      *result = v4[v7];
      ++*(_QWORD *)v2;
    }
  }
  else
  {
    result = sub_C8CA60(*a1, v3);
    if ( result )
    {
      *result = -2;
      ++*(_DWORD *)(v2 + 24);
      ++*(_QWORD *)v2;
    }
  }
  return result;
}
