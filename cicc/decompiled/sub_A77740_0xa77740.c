// Function: sub_A77740
// Address: 0xa77740
//
__int64 __fastcall sub_A77740(__int64 a1, const void *a2, __int64 a3)
{
  __int64 *v4; // rdi
  __int64 v5; // rax
  __int64 *v6; // r13
  __int64 v8; // rdx
  int v9; // eax
  const void *v10; // [rsp+0h] [rbp-20h] BYREF
  __int64 v11; // [rsp+8h] [rbp-18h]

  v4 = *(__int64 **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 16);
  v10 = a2;
  v11 = a3;
  v6 = sub_A77430(v4, (__int64)&v4[v5], (__int64)&v10);
  if ( v6 == (__int64 *)(*(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16))
    || !(unsigned __int8)sub_A721E0(v6, v10, v11) )
  {
    return a1;
  }
  v8 = *(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16);
  v9 = *(_DWORD *)(a1 + 16);
  if ( (__int64 *)v8 != v6 + 1 )
  {
    memmove(v6, v6 + 1, v8 - (_QWORD)(v6 + 1));
    v9 = *(_DWORD *)(a1 + 16);
  }
  *(_DWORD *)(a1 + 16) = v9 - 1;
  return a1;
}
