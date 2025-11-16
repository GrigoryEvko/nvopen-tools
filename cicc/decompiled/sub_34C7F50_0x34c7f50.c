// Function: sub_34C7F50
// Address: 0x34c7f50
//
unsigned __int64 __fastcall sub_34C7F50(__int64 a1, __int64 a2, const void **a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  unsigned __int64 result; // rax
  __int64 v10; // r9
  unsigned __int64 v11; // rbx
  __int64 v12; // r12
  __int64 v13; // r9
  void *v14; // rdi
  unsigned int v15; // r14d
  size_t v16; // rdx
  char *v17; // r13
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+0h] [rbp-40h]
  __int64 v20; // [rsp+8h] [rbp-38h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  v4 = a1;
  v7 = *(unsigned int *)(a1 + 8);
  v8 = *(unsigned int *)(a1 + 12);
  result = *(_QWORD *)a1;
  v10 = a2 + v7;
  if ( a2 + v7 > v8 )
  {
    if ( result > (unsigned __int64)a3 || (v8 = result + 40 * v7, (unsigned __int64)a3 >= v8) )
    {
      sub_34C7D80(a1, a2 + v7, v8, a4, a1, v10);
      v4 = a1;
      result = *(_QWORD *)a1;
      v7 = *(unsigned int *)(a1 + 8);
    }
    else
    {
      v17 = (char *)a3 - result;
      sub_34C7D80(a1, a2 + v7, v8, a4, a1, v10);
      v4 = a1;
      result = *(_QWORD *)a1;
      v7 = *(unsigned int *)(a1 + 8);
      a3 = (const void **)&v17[*(_QWORD *)a1];
    }
  }
  v11 = result + 40 * v7;
  if ( a2 )
  {
    v12 = a2;
    v13 = (__int64)(a3 + 1);
    do
    {
      while ( 1 )
      {
        if ( v11 )
        {
          result = *(unsigned int *)a3;
          v14 = (void *)(v11 + 24);
          *(_DWORD *)(v11 + 16) = 0;
          *(_QWORD *)(v11 + 8) = v11 + 24;
          *(_DWORD *)v11 = result;
          *(_DWORD *)(v11 + 20) = 4;
          v15 = *((_DWORD *)a3 + 4);
          if ( v15 )
          {
            if ( v11 + 8 != v13 )
              break;
          }
        }
        v11 += 40LL;
        if ( !--v12 )
          goto LABEL_11;
      }
      v16 = 4LL * v15;
      if ( v15 <= 4
        || (v19 = v13,
            v21 = v4,
            result = sub_C8D5F0(v11 + 8, (const void *)(v11 + 24), v15, 4u, v4, v13),
            v14 = *(void **)(v11 + 8),
            v4 = v21,
            v13 = v19,
            (v16 = 4LL * *((unsigned int *)a3 + 4)) != 0) )
      {
        v18 = v13;
        v20 = v4;
        result = (unsigned __int64)memcpy(v14, a3[1], v16);
        *(_DWORD *)(v11 + 16) = v15;
        v13 = v18;
        v4 = v20;
      }
      else
      {
        *(_DWORD *)(v11 + 16) = v15;
      }
      v11 += 40LL;
      --v12;
    }
    while ( v12 );
LABEL_11:
    LODWORD(v7) = *(_DWORD *)(v4 + 8);
  }
  *(_DWORD *)(v4 + 8) = v7 + a2;
  return result;
}
