// Function: sub_E17770
// Address: 0xe17770
//
char *__fastcall sub_E17770(__int64 a1, __int64 a2)
{
  int v4; // r12d
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  char *v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  char *v11; // rdi
  __int64 v12; // rsi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  char *v16; // rax
  char *result; // rax

  v4 = *(_DWORD *)(a2 + 32);
  *(_DWORD *)(a2 + 32) = 0;
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_QWORD *)(a2 + 16);
  v7 = *(char **)a2;
  if ( v5 + 9 > v6 )
  {
    v8 = v5 + 1001;
    v9 = 2 * v6;
    if ( v8 <= v9 )
      *(_QWORD *)(a2 + 16) = v9;
    else
      *(_QWORD *)(a2 + 16) = v8;
    v10 = realloc(v7);
    *(_QWORD *)a2 = v10;
    v7 = (char *)v10;
    if ( !v10 )
      goto LABEL_15;
    v5 = *(_QWORD *)(a2 + 8);
  }
  v11 = &v7[v5];
  *(_QWORD *)v11 = 0x6574616C706D6574LL;
  v11[8] = 60;
  *(_QWORD *)(a2 + 8) += 9LL;
  sub_E161C0((_QWORD *)(a1 + 24), (char **)a2);
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *(_QWORD *)(a2 + 16);
  if ( v12 + 11 <= v13 )
  {
    v16 = *(char **)a2;
    goto LABEL_12;
  }
  v14 = v12 + 1003;
  v15 = 2 * v13;
  if ( v14 > v15 )
    *(_QWORD *)(a2 + 16) = v14;
  else
    *(_QWORD *)(a2 + 16) = v15;
  v16 = (char *)realloc(*(void **)a2);
  *(_QWORD *)a2 = v16;
  if ( !v16 )
LABEL_15:
    abort();
  v12 = *(_QWORD *)(a2 + 8);
LABEL_12:
  result = &v16[v12];
  qmemcpy(result, "> typename ", 11);
  *(_DWORD *)(a2 + 32) = v4;
  *(_QWORD *)(a2 + 8) += 11LL;
  return result;
}
