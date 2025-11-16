// Function: sub_A77250
// Address: 0xa77250
//
char __fastcall sub_A77250(__int64 a1, int a2, __int64 a3)
{
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // rcx
  char *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // r14
  unsigned __int64 v14; // r8
  __int64 v15; // rdi
  unsigned __int64 v16; // rdx
  char *v17; // rsi
  char *v18; // rdx
  int v20[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(__int64 **)a1;
  v6 = *(unsigned int *)(a1 + 8);
  v20[0] = a2;
  v7 = sub_A771C0(v5, (__int64)&v5[v6], v20);
  v8 = *(unsigned int *)(a1 + 8);
  v9 = *(__int64 **)a1;
  v10 = (char *)v7;
  if ( v7 == (__int64 *)(*(_QWORD *)a1 + 8 * v8) )
  {
    v14 = *(unsigned int *)(a1 + 12);
    v16 = v8 + 1;
  }
  else
  {
    LOBYTE(v11) = sub_A71B30(v7, v20[0]);
    if ( (_BYTE)v11 )
    {
LABEL_11:
      *(_QWORD *)v10 = a3;
      return v11;
    }
    v12 = *(unsigned int *)(a1 + 8);
    v13 = *(__int64 **)a1;
    v14 = *(unsigned int *)(a1 + 12);
    v15 = v12;
    LODWORD(v11) = *(_DWORD *)(a1 + 8);
    v9 = *(__int64 **)a1;
    v16 = v12 + 1;
    v17 = (char *)(*(_QWORD *)a1 + v15 * 8);
    if ( v10 != v17 )
    {
      if ( v16 > v14 )
      {
        sub_C8D5F0(a1, a1 + 16, v16, 8);
        v11 = *(unsigned int *)(a1 + 8);
        v15 = v11;
        v10 = (char *)(*(_QWORD *)a1 + v10 - (char *)v13);
        v13 = *(__int64 **)a1;
        v17 = (char *)(*(_QWORD *)a1 + 8 * v11);
      }
      v18 = (char *)&v13[v15 - 1];
      if ( v17 )
      {
        *(_QWORD *)v17 = *(_QWORD *)v18;
        v13 = *(__int64 **)a1;
        v11 = *(unsigned int *)(a1 + 8);
        v15 = v11;
        v18 = (char *)(*(_QWORD *)a1 + 8 * v11 - 8);
      }
      if ( v10 != v18 )
      {
        memmove((char *)v13 + v15 * 8 - (v18 - v10), v10, v18 - v10);
        LODWORD(v11) = *(_DWORD *)(a1 + 8);
      }
      LODWORD(v11) = v11 + 1;
      *(_DWORD *)(a1 + 8) = v11;
      goto LABEL_11;
    }
  }
  if ( v16 > v14 )
  {
    sub_C8D5F0(a1, a1 + 16, v16, 8);
    v9 = *(__int64 **)a1;
  }
  v11 = *(unsigned int *)(a1 + 8);
  v9[v11] = a3;
  ++*(_DWORD *)(a1 + 8);
  return v11;
}
