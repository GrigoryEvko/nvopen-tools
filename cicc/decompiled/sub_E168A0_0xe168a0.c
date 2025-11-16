// Function: sub_E168A0
// Address: 0xe168a0
//
char *__fastcall sub_E168A0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rax
  char *result; // rax
  __int64 v16; // rsi

  ++*(_DWORD *)(a2 + 32);
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(char **)a2;
  v7 = v4 + 1;
  if ( v4 + 1 > v5 )
  {
    v8 = v4 + 993;
    v9 = 2 * v5;
    if ( v8 > v9 )
      *(_QWORD *)(a2 + 16) = v8;
    else
      *(_QWORD *)(a2 + 16) = v9;
    v10 = realloc(v6);
    *(_QWORD *)a2 = v10;
    v6 = (char *)v10;
    if ( !v10 )
      goto LABEL_14;
    v4 = *(_QWORD *)(a2 + 8);
    v7 = v4 + 1;
  }
  *(_QWORD *)(a2 + 8) = v7;
  v6[v4] = 91;
  sub_E161C0((_QWORD *)(a1 + 16), (char **)a2);
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  if ( v11 + 1 > v12 )
  {
    v13 = v11 + 993;
    v14 = 2 * v12;
    if ( v13 > v14 )
      *(_QWORD *)(a2 + 16) = v13;
    else
      *(_QWORD *)(a2 + 16) = v14;
    result = (char *)realloc(*(void **)a2);
    *(_QWORD *)a2 = result;
    if ( result )
    {
      v16 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(a2 + 8) = v16 + 1;
      result[v16] = 93;
      return result;
    }
LABEL_14:
    abort();
  }
  result = *(char **)a2;
  *(_QWORD *)(a2 + 8) = v11 + 1;
  result[v11] = 93;
  return result;
}
