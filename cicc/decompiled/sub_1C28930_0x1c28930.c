// Function: sub_1C28930
// Address: 0x1c28930
//
__int64 __fastcall sub_1C28930(__int64 *a1, char *a2, size_t a3, __int64 a4)
{
  __int64 v7; // r12
  void *v8; // rdx
  const char *v9; // rax
  size_t v10; // rdx
  _DWORD *v11; // rdi
  char *v12; // rsi
  unsigned __int64 v13; // rax
  _BYTE *v14; // rdi
  __int64 result; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  size_t v18; // [rsp+8h] [rbp-28h]

  v7 = *a1;
  v8 = *(void **)(*a1 + 24);
  if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v8 <= 9u )
  {
    v7 = sub_16E7EE0(*a1, "Function: ", 0xAu);
  }
  else
  {
    qmemcpy(v8, "Function: ", 10);
    *(_QWORD *)(v7 + 24) += 10LL;
  }
  v9 = sub_1649960(a4);
  v11 = *(_DWORD **)(v7 + 24);
  v12 = (char *)v9;
  v13 = *(_QWORD *)(v7 + 16) - (_QWORD)v11;
  if ( v13 < v10 )
  {
    v17 = sub_16E7EE0(v7, v12, v10);
    v11 = *(_DWORD **)(v17 + 24);
    v7 = v17;
    v13 = *(_QWORD *)(v17 + 16) - (_QWORD)v11;
LABEL_5:
    if ( v13 > 6 )
      goto LABEL_6;
    goto LABEL_13;
  }
  if ( !v10 )
    goto LABEL_5;
  v18 = v10;
  memcpy(v11, v12, v10);
  v11 = (_DWORD *)(v18 + *(_QWORD *)(v7 + 24));
  v16 = *(_QWORD *)(v7 + 16) - (_QWORD)v11;
  *(_QWORD *)(v7 + 24) = v11;
  if ( v16 > 6 )
  {
LABEL_6:
    *v11 = 1935757321;
    *((_WORD *)v11 + 2) = 14963;
    *((_BYTE *)v11 + 6) = 32;
    v14 = (_BYTE *)(*(_QWORD *)(v7 + 24) + 7LL);
    result = *(_QWORD *)(v7 + 16);
    *(_QWORD *)(v7 + 24) = v14;
    if ( result - (__int64)v14 >= a3 )
      goto LABEL_7;
LABEL_14:
    v7 = sub_16E7EE0(v7, a2, a3);
    result = *(_QWORD *)(v7 + 16);
    v14 = *(_BYTE **)(v7 + 24);
    goto LABEL_8;
  }
LABEL_13:
  v7 = sub_16E7EE0(v7, "\tPass: ", 7u);
  v14 = *(_BYTE **)(v7 + 24);
  result = *(_QWORD *)(v7 + 16);
  if ( result - (__int64)v14 < a3 )
    goto LABEL_14;
LABEL_7:
  if ( a3 )
  {
    memcpy(v14, a2, a3);
    result = *(_QWORD *)(v7 + 16);
    v14 = (_BYTE *)(a3 + *(_QWORD *)(v7 + 24));
    *(_QWORD *)(v7 + 24) = v14;
    if ( (_BYTE *)result != v14 )
      goto LABEL_9;
    return sub_16E7EE0(v7, "\n", 1u);
  }
LABEL_8:
  if ( (_BYTE *)result != v14 )
  {
LABEL_9:
    *v14 = 10;
    ++*(_QWORD *)(v7 + 24);
    return result;
  }
  return sub_16E7EE0(v7, "\n", 1u);
}
