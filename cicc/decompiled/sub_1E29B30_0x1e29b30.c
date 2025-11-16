// Function: sub_1E29B30
// Address: 0x1e29b30
//
__int64 *__fastcall sub_1E29B30(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 *v3; // r13
  __int64 v4; // r14
  __int64 *v5; // rbx
  __int64 *v6; // r15
  int v7; // r8d
  int v8; // r9d
  __int64 *i; // [rsp+0h] [rbp-40h]

  result = *(__int64 **)(a1 + 40);
  v3 = *(__int64 **)(a1 + 32);
  for ( i = result; i != v3; ++v3 )
  {
    v4 = *v3;
    v5 = *(__int64 **)(*v3 + 96);
    v6 = *(__int64 **)(*v3 + 88);
    if ( v5 != v6 )
    {
      while ( 1 )
      {
        result = (__int64 *)sub_1DA1810(a1 + 56, *v6);
        if ( !(_DWORD)result )
          break;
        if ( v5 == ++v6 )
          goto LABEL_9;
      }
      result = (__int64 *)*(unsigned int *)(a2 + 8);
      if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v7, v8);
        result = (__int64 *)*(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8LL * (_QWORD)result) = v4;
      ++*(_DWORD *)(a2 + 8);
    }
LABEL_9:
    ;
  }
  return result;
}
