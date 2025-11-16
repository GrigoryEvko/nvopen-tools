// Function: sub_30279F0
// Address: 0x30279f0
//
int *__fastcall sub_30279F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // ebx
  int v7; // r12d
  __int64 v8; // rax
  int *result; // rax
  unsigned int v10; // ebx
  int v11; // edx
  int v12; // r13d
  unsigned __int64 *v13; // rax
  _WORD *v14; // rdx
  unsigned int v15; // [rsp+Ch] [rbp-34h]

  v6 = *(_DWORD *)a1;
  v7 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 168) + 208LL) + 8LL);
  v8 = *(unsigned int *)(a1 + 40);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 4u, a5, a6);
    v8 = *(unsigned int *)(a1 + 40);
  }
  v15 = 0;
  *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * v8) = v6;
  result = *(int **)(a1 + 32);
  v10 = 0;
  v11 = *(_DWORD *)a1;
  ++*(_DWORD *)(a1 + 40);
  v12 = *result;
  if ( v11 )
  {
LABEL_4:
    if ( v12 == v10 )
    {
      while ( 1 )
      {
        v10 += v7;
        sub_30278D0(a1, v15, a2);
        result = *(int **)(a1 + 32);
        v12 = result[++v15];
        if ( *(_DWORD *)a1 <= v10 )
          break;
LABEL_8:
        if ( !v10 )
          goto LABEL_4;
        v14 = *(_WORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v14 <= 1u )
        {
          sub_CB6200(a2, (unsigned __int8 *)", ", 2u);
          goto LABEL_4;
        }
        *v14 = 8236;
        *(_QWORD *)(a2 + 32) += 2LL;
        if ( v12 != v10 )
          goto LABEL_5;
      }
    }
    else
    {
LABEL_5:
      v13 = (unsigned __int64 *)(*(_QWORD *)(a1 + 8) + v10);
      if ( v7 == 4 )
        result = (int *)sub_CB59D0(a2, *(unsigned int *)v13);
      else
        result = (int *)sub_CB59D0(a2, *v13);
      v10 += v7;
      if ( *(_DWORD *)a1 > v10 )
        goto LABEL_8;
    }
  }
  return result;
}
