// Function: sub_E17670
// Address: 0xe17670
//
char *__fastcall sub_E17670(__int64 a1, __int64 a2)
{
  int v4; // r13d
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  char *v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rax
  char *result; // rax

  v4 = *(_DWORD *)(a2 + 32);
  *(_DWORD *)(a2 + 32) = 0;
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_QWORD *)(a2 + 16);
  v7 = *(char **)a2;
  if ( v5 + 1 > v6 )
  {
    v8 = v5 + 993;
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
  v7[v5] = 60;
  ++*(_QWORD *)(a2 + 8);
  sub_E161C0((_QWORD *)(a1 + 16), (char **)a2);
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(_QWORD *)(a2 + 16);
  if ( v11 + 1 <= v12 )
  {
    result = *(char **)a2;
    goto LABEL_12;
  }
  v13 = v11 + 993;
  v14 = 2 * v12;
  if ( v13 > v14 )
    *(_QWORD *)(a2 + 16) = v13;
  else
    *(_QWORD *)(a2 + 16) = v14;
  result = (char *)realloc(*(void **)a2);
  *(_QWORD *)a2 = result;
  if ( !result )
LABEL_15:
    abort();
  v11 = *(_QWORD *)(a2 + 8);
LABEL_12:
  result[v11] = 62;
  *(_DWORD *)(a2 + 32) = v4;
  ++*(_QWORD *)(a2 + 8);
  return result;
}
