// Function: sub_1E29BE0
// Address: 0x1e29be0
//
__int64 *__fastcall sub_1E29BE0(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 *v3; // r13
  __int64 *v4; // rbx
  __int64 *j; // r14
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // rcx
  __int64 *i; // [rsp+8h] [rbp-38h]

  result = *(__int64 **)(a1 + 40);
  v3 = *(__int64 **)(a1 + 32);
  for ( i = result; i != v3; ++v3 )
  {
    result = (__int64 *)*v3;
    v4 = *(__int64 **)(*v3 + 96);
    for ( j = *(__int64 **)(*v3 + 88); v4 != j; ++*(_DWORD *)(a2 + 8) )
    {
      while ( 1 )
      {
        result = (__int64 *)sub_1DA1810(a1 + 56, *j);
        if ( !(_DWORD)result )
          break;
        if ( v4 == ++j )
          goto LABEL_9;
      }
      result = (__int64 *)*(unsigned int *)(a2 + 8);
      if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v6, v7);
        result = (__int64 *)*(unsigned int *)(a2 + 8);
      }
      v8 = *j++;
      *(_QWORD *)(*(_QWORD *)a2 + 8LL * (_QWORD)result) = v8;
    }
LABEL_9:
    ;
  }
  return result;
}
