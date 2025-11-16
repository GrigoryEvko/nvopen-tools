// Function: sub_E17370
// Address: 0xe17370
//
char *__fastcall sub_E17370(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  char *result; // rax
  __int64 v15; // rsi

  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(char **)a2;
  if ( v4 + 12 > v5 )
  {
    v7 = v4 + 1004;
    v8 = 2 * v5;
    if ( v7 > v8 )
      *(_QWORD *)(a2 + 16) = v7;
    else
      *(_QWORD *)(a2 + 16) = v8;
    v9 = realloc(v6);
    *(_QWORD *)a2 = v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_14;
    v4 = *(_QWORD *)(a2 + 8);
  }
  qmemcpy(&v6[v4], " [enable_if:", 12);
  *(_QWORD *)(a2 + 8) += 12LL;
  sub_E161C0((_QWORD *)(a1 + 16), (char **)a2);
  v10 = *(_QWORD *)(a2 + 8);
  v11 = *(_QWORD *)(a2 + 16);
  if ( v10 + 1 > v11 )
  {
    v12 = v10 + 993;
    v13 = 2 * v11;
    if ( v12 > v13 )
      *(_QWORD *)(a2 + 16) = v12;
    else
      *(_QWORD *)(a2 + 16) = v13;
    result = (char *)realloc(*(void **)a2);
    *(_QWORD *)a2 = result;
    if ( result )
    {
      v15 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(a2 + 8) = v15 + 1;
      result[v15] = 93;
      return result;
    }
LABEL_14:
    abort();
  }
  result = *(char **)a2;
  *(_QWORD *)(a2 + 8) = v10 + 1;
  result[v10] = 93;
  return result;
}
