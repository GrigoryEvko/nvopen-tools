// Function: sub_39E4130
// Address: 0x39e4130
//
_BYTE *__fastcall sub_39E4130(__int64 a1, char *a2, size_t a3, _BYTE *a4)
{
  __int64 v7; // rdi
  _QWORD *v8; // rdx
  __int64 v9; // r13
  _WORD *v10; // rdx
  void *v11; // rdi
  size_t v12; // r13
  _BYTE *result; // rax
  __int64 v14; // r14
  char *v15; // rsi
  void *v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rax

  v7 = *(_QWORD *)(a1 + 272);
  v8 = *(_QWORD **)(v7 + 24);
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 7u )
  {
    sub_16E7EE0(v7, ".symver ", 8u);
  }
  else
  {
    *v8 = 0x207265766D79732ELL;
    *(_QWORD *)(v7 + 24) += 8LL;
  }
  sub_38E2490(a4, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v9 = *(_QWORD *)(a1 + 272);
  v10 = *(_WORD **)(v9 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 1u )
  {
    v18 = sub_16E7EE0(*(_QWORD *)(a1 + 272), ", ", 2u);
    v11 = *(void **)(v18 + 24);
    v9 = v18;
  }
  else
  {
    *v10 = 8236;
    v11 = (void *)(*(_QWORD *)(v9 + 24) + 2LL);
    *(_QWORD *)(v9 + 24) = v11;
  }
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v11 < a3 )
  {
    sub_16E7EE0(v9, a2, a3);
LABEL_7:
    v12 = *(unsigned int *)(a1 + 312);
    if ( !*(_DWORD *)(a1 + 312) )
      goto LABEL_8;
LABEL_11:
    v14 = *(_QWORD *)(a1 + 272);
    v15 = *(char **)(a1 + 304);
    v16 = *(void **)(v14 + 24);
    if ( v12 > *(_QWORD *)(v14 + 16) - (_QWORD)v16 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v15, v12);
    }
    else
    {
      memcpy(v16, v15, v12);
      *(_QWORD *)(v14 + 24) += v12;
    }
    goto LABEL_8;
  }
  if ( !a3 )
    goto LABEL_7;
  memcpy(v11, a2, a3);
  *(_QWORD *)(v9 + 24) += a3;
  v12 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
    goto LABEL_11;
LABEL_8:
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v17 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v17 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v17 + 16) )
    return (_BYTE *)sub_16E7DE0(v17, 10);
  *(_QWORD *)(v17 + 24) = result + 1;
  *result = 10;
  return result;
}
