// Function: sub_16BD760
// Address: 0x16bd760
//
char *__fastcall sub_16BD760(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // r14
  __int64 v6; // r8
  unsigned int v7; // ebx
  unsigned __int64 v8; // r15
  __int64 v9; // rax
  char *v10; // r8
  size_t v11; // r9
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // [rsp+8h] [rbp-38h]

  v2 = *a2;
  v3 = a2[1];
  v4 = *a2 + 3;
  v5 = 4LL * *(unsigned int *)(a1 + 8);
  a2[10] += v5;
  v6 = (v4 & 0xFFFFFFFFFFFFFFFCLL) - v2;
  if ( v5 + v6 <= (unsigned __int64)(v3 - v2) )
  {
    v10 = (char *)(v2 + v6);
    *a2 = (__int64)&v10[v5];
  }
  else if ( (unsigned __int64)(v5 + 3) > 0x1000 )
  {
    v13 = malloc(v5 + 3);
    if ( !v13 )
      sub_16BD1C0("Allocation failed", 1u);
    v14 = *((unsigned int *)a2 + 18);
    if ( (unsigned int)v14 >= *((_DWORD *)a2 + 19) )
    {
      sub_16CD150(a2 + 8, a2 + 10, 0, 16);
      v14 = *((unsigned int *)a2 + 18);
    }
    v15 = (__int64 *)(a2[8] + 16 * v14);
    *v15 = v13;
    v15[1] = v5 + 3;
    ++*((_DWORD *)a2 + 18);
    v10 = (char *)((v13 + 3) & 0xFFFFFFFFFFFFFFFCLL);
  }
  else
  {
    v7 = *((_DWORD *)a2 + 6);
    v8 = 0x40000000000LL;
    if ( v7 >> 7 < 0x1E )
      v8 = 4096LL << (v7 >> 7);
    v9 = malloc(v8);
    if ( !v9 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v7 = *((_DWORD *)a2 + 6);
      v9 = 0;
    }
    if ( v7 >= *((_DWORD *)a2 + 7) )
    {
      v16 = v9;
      sub_16CD150(a2 + 2, a2 + 4, 0, 8);
      v7 = *((_DWORD *)a2 + 6);
      v9 = v16;
    }
    *(_QWORD *)(a2[2] + 8LL * v7) = v9;
    ++*((_DWORD *)a2 + 6);
    a2[1] = v9 + v8;
    v10 = (char *)((v9 + 3) & 0xFFFFFFFFFFFFFFFCLL);
    *a2 = (__int64)&v10[v5];
  }
  v11 = 4LL * *(unsigned int *)(a1 + 8);
  if ( v11 )
    return (char *)memmove(v10, *(const void **)a1, v11);
  return v10;
}
