// Function: sub_A77390
// Address: 0xa77390
//
__int64 __fastcall sub_A77390(__int64 a1, int a2)
{
  __int64 *v3; // rdi
  __int64 v4; // rax
  __int64 *v5; // r13
  __int64 v7; // rdx
  int v8; // eax
  int v9[5]; // [rsp+Ch] [rbp-14h] BYREF

  v3 = *(__int64 **)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 16);
  v9[0] = a2;
  v5 = sub_A771C0(v3, (__int64)&v3[v4], v9);
  if ( v5 == (__int64 *)(*(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16)) || !sub_A71B30(v5, v9[0]) )
    return a1;
  v7 = *(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16);
  v8 = *(_DWORD *)(a1 + 16);
  if ( (__int64 *)v7 != v5 + 1 )
  {
    memmove(v5, v5 + 1, v7 - (_QWORD)(v5 + 1));
    v8 = *(_DWORD *)(a1 + 16);
  }
  *(_DWORD *)(a1 + 16) = v8 - 1;
  return a1;
}
