// Function: sub_2467520
// Address: 0x2467520
//
_BYTE *__fastcall sub_2467520(__int64 a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r13
  _WORD *v12; // rax
  __int64 v13; // rdi
  _BYTE *result; // rax
  unsigned __int64 v15; // rax

  v6 = a3(a4, "MemorySanitizerPass]", 19);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_QWORD *)(a2 + 24);
  v11 = v7;
  if ( v10 - (unsigned __int64)v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v8 = *(_BYTE **)(a2 + 32);
    v10 = *(_QWORD *)(a2 + 24);
LABEL_3:
    if ( v10 > (unsigned __int64)v8 )
      goto LABEL_4;
LABEL_12:
    sub_CB5D20(a2, 60);
    v12 = *(_WORD **)(a2 + 32);
    if ( !*(_BYTE *)(a1 + 8) )
      goto LABEL_5;
    goto LABEL_13;
  }
  if ( !v7 )
    goto LABEL_3;
  memcpy(v8, v9, v7);
  v15 = *(_QWORD *)(a2 + 24);
  v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
  *(_QWORD *)(a2 + 32) = v8;
  if ( v15 <= (unsigned __int64)v8 )
    goto LABEL_12;
LABEL_4:
  *(_QWORD *)(a2 + 32) = v8 + 1;
  *v8 = 60;
  v12 = *(_WORD **)(a2 + 32);
  if ( !*(_BYTE *)(a1 + 8) )
    goto LABEL_5;
LABEL_13:
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v12 <= 7u )
  {
    sub_CB6200(a2, "recover;", 8u);
    v12 = *(_WORD **)(a2 + 32);
  }
  else
  {
    *(_QWORD *)v12 = 0x3B7265766F636572LL;
    v12 = (_WORD *)(*(_QWORD *)(a2 + 32) + 8LL);
    *(_QWORD *)(a2 + 32) = v12;
  }
LABEL_5:
  if ( *(_BYTE *)a1 )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v12 <= 6u )
    {
      sub_CB6200(a2, "kernel;", 7u);
      v12 = *(_WORD **)(a2 + 32);
    }
    else
    {
      *(_DWORD *)v12 = 1852990827;
      v12[2] = 27749;
      *((_BYTE *)v12 + 6) = 59;
      v12 = (_WORD *)(*(_QWORD *)(a2 + 32) + 7LL);
      *(_QWORD *)(a2 + 32) = v12;
    }
  }
  if ( *(_BYTE *)(a1 + 9) )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v12 <= 0xCu )
    {
      sub_CB6200(a2, "eager-checks;", 0xDu);
      v12 = *(_WORD **)(a2 + 32);
    }
    else
    {
      qmemcpy(v12, "eager-checks;", 13);
      v12 = (_WORD *)(*(_QWORD *)(a2 + 32) + 13LL);
      *(_QWORD *)(a2 + 32) = v12;
    }
  }
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v12 <= 0xDu )
  {
    v13 = sub_CB6200(a2, "track-origins=", 0xEu);
  }
  else
  {
    v13 = a2;
    qmemcpy(v12, "track-origins=", 14);
    *(_QWORD *)(a2 + 32) += 14LL;
  }
  sub_CB59F0(v13, *(int *)(a1 + 4));
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 62;
  return result;
}
