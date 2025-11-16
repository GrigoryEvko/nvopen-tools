// Function: sub_A77520
// Address: 0xa77520
//
__int64 __fastcall sub_A77520(__int64 a1, const void *a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // rdi
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // rcx
  char *v11; // r12
  __int64 result; // rax
  __int64 v13; // rdx
  __int64 *v14; // r14
  unsigned __int64 v15; // r8
  __int64 v16; // rdi
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  char *v19; // rsi
  char *v20; // rdx
  const void *v21; // [rsp+0h] [rbp-30h] BYREF
  __int64 v22; // [rsp+8h] [rbp-28h]

  v6 = *(__int64 **)a1;
  v7 = *(unsigned int *)(a1 + 8);
  v21 = a2;
  v22 = a3;
  v8 = sub_A77430(v6, (__int64)&v6[v7], (__int64)&v21);
  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(__int64 **)a1;
  v11 = (char *)v8;
  if ( v8 == (__int64 *)(*(_QWORD *)a1 + 8 * v9) )
  {
    v15 = *(unsigned int *)(a1 + 12);
    v18 = v9 + 1;
  }
  else
  {
    result = sub_A721E0(v8, v21, v22);
    if ( (_BYTE)result )
    {
LABEL_11:
      *(_QWORD *)v11 = a4;
      return result;
    }
    v13 = *(unsigned int *)(a1 + 8);
    v14 = *(__int64 **)a1;
    v15 = *(unsigned int *)(a1 + 12);
    v16 = v13;
    LODWORD(v17) = *(_DWORD *)(a1 + 8);
    v10 = *(__int64 **)a1;
    v18 = v13 + 1;
    v19 = (char *)(*(_QWORD *)a1 + v16 * 8);
    if ( v11 != v19 )
    {
      if ( v18 > v15 )
      {
        sub_C8D5F0(a1, a1 + 16, v18, 8);
        v17 = *(unsigned int *)(a1 + 8);
        v16 = v17;
        v11 = (char *)(*(_QWORD *)a1 + v11 - (char *)v14);
        v14 = *(__int64 **)a1;
        v19 = (char *)(*(_QWORD *)a1 + 8 * v17);
      }
      v20 = (char *)&v14[v16 - 1];
      if ( v19 )
      {
        *(_QWORD *)v19 = *(_QWORD *)v20;
        v14 = *(__int64 **)a1;
        v17 = *(unsigned int *)(a1 + 8);
        v16 = v17;
        v20 = (char *)(*(_QWORD *)a1 + 8 * v17 - 8);
      }
      if ( v11 != v20 )
      {
        memmove((char *)v14 + v16 * 8 - (v20 - v11), v11, v20 - v11);
        LODWORD(v17) = *(_DWORD *)(a1 + 8);
      }
      result = (unsigned int)(v17 + 1);
      *(_DWORD *)(a1 + 8) = result;
      goto LABEL_11;
    }
  }
  if ( v18 > v15 )
  {
    sub_C8D5F0(a1, a1 + 16, v18, 8);
    v10 = *(__int64 **)a1;
  }
  result = *(unsigned int *)(a1 + 8);
  v10[result] = a4;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
