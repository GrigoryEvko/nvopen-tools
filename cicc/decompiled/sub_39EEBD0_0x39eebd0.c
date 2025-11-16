// Function: sub_39EEBD0
// Address: 0x39eebd0
//
_BYTE *__fastcall sub_39EEBD0(__int64 a1, _BYTE *a2, unsigned __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r15
  void *v6; // r14
  size_t v7; // rax
  void *v8; // rdi
  size_t v9; // r13
  size_t v10; // r13
  _BYTE *result; // rax
  __int64 v12; // r14
  char *v13; // rsi
  void *v14; // rdi
  __int64 v15; // rdi

  sub_38DC4E0(a1, (__int64)a2, a3);
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v4 = *(_QWORD *)(a1 + 280);
  v5 = *(_QWORD *)(a1 + 272);
  v6 = *(void **)(v4 + 64);
  if ( !v6 )
    goto LABEL_5;
  v7 = strlen(*(const char **)(v4 + 64));
  v8 = *(void **)(v5 + 24);
  v9 = v7;
  if ( v7 <= *(_QWORD *)(v5 + 16) - (_QWORD)v8 )
  {
    if ( v7 )
    {
      memcpy(v8, v6, v7);
      *(_QWORD *)(v5 + 24) += v9;
    }
LABEL_5:
    v10 = *(unsigned int *)(a1 + 312);
    if ( !*(_DWORD *)(a1 + 312) )
      goto LABEL_6;
LABEL_9:
    v12 = *(_QWORD *)(a1 + 272);
    v13 = *(char **)(a1 + 304);
    v14 = *(void **)(v12 + 24);
    if ( v10 > *(_QWORD *)(v12 + 16) - (_QWORD)v14 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v13, v10);
    }
    else
    {
      memcpy(v14, v13, v10);
      *(_QWORD *)(v12 + 24) += v10;
    }
    goto LABEL_6;
  }
  sub_16E7EE0(v5, (char *)v6, v7);
  v10 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
    goto LABEL_9;
LABEL_6:
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v15 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v15 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v15 + 16) )
    return (_BYTE *)sub_16E7DE0(v15, 10);
  *(_QWORD *)(v15 + 24) = result + 1;
  *result = 10;
  return result;
}
