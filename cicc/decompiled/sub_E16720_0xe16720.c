// Function: sub_E16720
// Address: 0xe16720
//
char *__fastcall sub_E16720(__int64 a1, __int64 a2)
{
  _BYTE *v4; // r13
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  char *v7; // rdi
  __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax
  char *result; // rax

  if ( *(_BYTE *)(a1 + 40) )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
  }
  v4 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 32LL))(v4, a2);
  if ( (v4[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 40LL))(v4, a2);
  if ( *(_BYTE *)(a1 + 40) )
  {
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_QWORD *)(a2 + 16);
  ++*(_DWORD *)(a2 + 32);
  v7 = *(char **)a2;
  v8 = v5 + 1;
  if ( v5 + 1 > v6 )
  {
    v9 = v5 + 993;
    v10 = 2 * v6;
    if ( v9 > v10 )
      *(_QWORD *)(a2 + 16) = v9;
    else
      *(_QWORD *)(a2 + 16) = v10;
    v11 = realloc(v7);
    *(_QWORD *)a2 = v11;
    v7 = (char *)v11;
    if ( !v11 )
      goto LABEL_21;
    v5 = *(_QWORD *)(a2 + 8);
    v8 = v5 + 1;
  }
  *(_QWORD *)(a2 + 8) = v8;
  v7[v5] = 40;
  sub_E161C0((_QWORD *)(a1 + 24), (char **)a2);
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  v14 = v12 + 1;
  if ( v12 + 1 <= v13 )
  {
    result = *(char **)a2;
    goto LABEL_18;
  }
  v15 = v12 + 993;
  v16 = 2 * v13;
  if ( v15 > v16 )
    *(_QWORD *)(a2 + 16) = v15;
  else
    *(_QWORD *)(a2 + 16) = v16;
  result = (char *)realloc(*(void **)a2);
  *(_QWORD *)a2 = result;
  if ( !result )
LABEL_21:
    abort();
  v12 = *(_QWORD *)(a2 + 8);
  v14 = v12 + 1;
LABEL_18:
  *(_QWORD *)(a2 + 8) = v14;
  result[v12] = 41;
  return result;
}
