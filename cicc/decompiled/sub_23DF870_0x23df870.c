// Function: sub_23DF870
// Address: 0x23df870
//
_WORD *__fastcall sub_23DF870(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r13
  _WORD *result; // rax
  unsigned __int64 v13; // rax

  v6 = a3(a4, "AddressSanitizerPass]", 20);
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
LABEL_9:
    sub_CB5D20(a2, 60);
    result = *(_WORD **)(a2 + 32);
    if ( !*a1 )
      goto LABEL_5;
    goto LABEL_10;
  }
  if ( !v7 )
    goto LABEL_3;
  memcpy(v8, v9, v7);
  v13 = *(_QWORD *)(a2 + 24);
  v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
  *(_QWORD *)(a2 + 32) = v8;
  if ( v13 <= (unsigned __int64)v8 )
    goto LABEL_9;
LABEL_4:
  *(_QWORD *)(a2 + 32) = v8 + 1;
  *v8 = 60;
  result = *(_WORD **)(a2 + 32);
  if ( !*a1 )
    goto LABEL_5;
LABEL_10:
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)result <= 6u )
  {
    sub_CB6200(a2, "kernel;", 7u);
    result = *(_WORD **)(a2 + 32);
  }
  else
  {
    *(_DWORD *)result = 1852990827;
    result[2] = 27749;
    *((_BYTE *)result + 6) = 59;
    result = (_WORD *)(*(_QWORD *)(a2 + 32) + 7LL);
    *(_QWORD *)(a2 + 32) = result;
  }
LABEL_5:
  if ( a1[2] )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)result <= 0xEu )
    {
      sub_CB6200(a2, (unsigned __int8 *)"use-after-scope", 0xFu);
      result = *(_WORD **)(a2 + 32);
    }
    else
    {
      qmemcpy(result, "use-after-scope", 15);
      result = (_WORD *)(*(_QWORD *)(a2 + 32) + 15LL);
      *(_QWORD *)(a2 + 32) = result;
    }
  }
  if ( *(_QWORD *)(a2 + 24) <= (unsigned __int64)result )
    return (_WORD *)sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = (char *)result + 1;
  *(_BYTE *)result = 62;
  return result;
}
