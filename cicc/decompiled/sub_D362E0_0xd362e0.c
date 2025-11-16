// Function: sub_D362E0
// Address: 0xd362e0
//
_BYTE *__fastcall sub_D362E0(unsigned int *a1, __int64 a2, unsigned int a3, _QWORD *a4)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  size_t v11; // rax
  char *v12; // rdi
  char *v13; // rsi
  size_t v14; // rdx
  unsigned __int64 v15; // rax
  char *v16; // rdx
  unsigned __int64 v17; // rax
  unsigned int v18; // ebx
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // r12
  _BYTE *result; // rax
  char *src; // [rsp+8h] [rbp-38h]
  char *srca; // [rsp+8h] [rbp-38h]

  v8 = sub_CB69B0(a2, a3);
  v9 = (int)a1[2];
  v10 = v8;
  if ( !off_4C5D120[v9] )
    goto LABEL_11;
  src = off_4C5D120[v9];
  v11 = strlen(src);
  v12 = *(char **)(v10 + 32);
  v13 = src;
  v14 = v11;
  v15 = *(_QWORD *)(v10 + 24) - (_QWORD)v12;
  if ( v14 > v15 )
  {
    v10 = sub_CB6200(v10, (unsigned __int8 *)src, v14);
LABEL_11:
    v12 = *(char **)(v10 + 32);
    v15 = *(_QWORD *)(v10 + 24) - (_QWORD)v12;
LABEL_12:
    if ( v15 <= 1 )
      goto LABEL_5;
    goto LABEL_13;
  }
  if ( !v14 )
    goto LABEL_12;
  srca = (char *)v14;
  memcpy(v12, v13, v14);
  v16 = &srca[*(_QWORD *)(v10 + 32)];
  v17 = *(_QWORD *)(v10 + 24) - (_QWORD)v16;
  *(_QWORD *)(v10 + 32) = v16;
  v12 = v16;
  if ( v17 <= 1 )
  {
LABEL_5:
    sub_CB6200(v10, (unsigned __int8 *)":\n", 2u);
    goto LABEL_6;
  }
LABEL_13:
  *(_WORD *)v12 = 2618;
  *(_QWORD *)(v10 + 32) += 2LL;
LABEL_6:
  v18 = a3 + 2;
  v19 = sub_CB69B0(a2, v18);
  sub_A69870(*(_QWORD *)(*a4 + 8LL * *a1), (_BYTE *)v19, 0);
  v20 = *(_QWORD *)(v19 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v19 + 24) - v20) <= 4 )
  {
    sub_CB6200(v19, " -> \n", 5u);
  }
  else
  {
    *(_DWORD *)v20 = 540945696;
    *(_BYTE *)(v20 + 4) = 10;
    *(_QWORD *)(v19 + 32) += 5LL;
  }
  v21 = sub_CB69B0(a2, v18);
  sub_A69870(*(_QWORD *)(*a4 + 8LL * a1[1]), (_BYTE *)v21, 0);
  result = *(_BYTE **)(v21 + 32);
  if ( *(_BYTE **)(v21 + 24) == result )
    return (_BYTE *)sub_CB6200(v21, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(v21 + 32);
  return result;
}
