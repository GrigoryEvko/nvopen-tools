// Function: sub_16E5DA0
// Address: 0x16e5da0
//
char *__fastcall sub_16E5DA0(__int64 a1, char **a2)
{
  unsigned __int64 v2; // r12
  char *v4; // r13
  char *v5; // rax
  unsigned __int64 v6; // r15
  __int64 v7; // rax
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rdx
  size_t v11; // rdx
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rax
  char *v15; // rax
  unsigned int v16; // [rsp+Ch] [rbp-34h]

  v2 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return 0;
  v4 = *a2;
  v5 = a2[1];
  a2[10] += v2;
  if ( v2 > v5 - v4 )
  {
    if ( v2 > 0x1000 )
    {
      v4 = (char *)malloc(v2);
      if ( !v4 )
        sub_16BD1C0("Allocation failed", 1u);
      v14 = *((unsigned int *)a2 + 18);
      if ( (unsigned int)v14 >= *((_DWORD *)a2 + 19) )
      {
        sub_16CD150((__int64)(a2 + 8), a2 + 10, 0, 16, v12, v13);
        v14 = *((unsigned int *)a2 + 18);
      }
      v15 = &a2[8][16 * v14];
      *(_QWORD *)v15 = v4;
      *((_QWORD *)v15 + 1) = v2;
      ++*((_DWORD *)a2 + 18);
      goto LABEL_13;
    }
    v6 = 0x40000000000LL;
    v16 = *((_DWORD *)a2 + 6);
    if ( v16 >> 7 < 0x1E )
      v6 = 4096LL << (v16 >> 7);
    v7 = malloc(v6);
    v10 = v16;
    v4 = (char *)v7;
    if ( !v7 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v10 = *((unsigned int *)a2 + 6);
    }
    if ( *((_DWORD *)a2 + 7) <= (unsigned int)v10 )
    {
      sub_16CD150((__int64)(a2 + 2), a2 + 4, 0, 8, v8, v9);
      v10 = *((unsigned int *)a2 + 6);
    }
    *(_QWORD *)&a2[2][8 * v10] = v4;
    ++*((_DWORD *)a2 + 6);
    a2[1] = &v4[v6];
  }
  *a2 = &v4[v2];
LABEL_13:
  v11 = *(_QWORD *)(a1 + 8);
  if ( v11 )
    memmove(v4, *(const void **)a1, v11);
  return v4;
}
