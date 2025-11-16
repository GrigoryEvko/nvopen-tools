// Function: sub_28BFC00
// Address: 0x28bfc00
//
_BYTE *__fastcall sub_28BFC00(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r13
  _BYTE *v12; // rax
  __int64 v13; // rdi
  _BYTE *result; // rax
  unsigned __int64 v15; // rax

  v6 = a3(a4, "MergedLoadStoreMotionPass]", 25);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_QWORD *)(a2 + 24);
  v11 = v7;
  if ( v10 - (unsigned __int64)v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v8 = *(_BYTE **)(a2 + 32);
    v10 = *(_QWORD *)(a2 + 24);
  }
  else if ( v7 )
  {
    memcpy(v8, v9, v7);
    v15 = *(_QWORD *)(a2 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 32) = v8;
    if ( v15 > (unsigned __int64)v8 )
      goto LABEL_4;
    goto LABEL_13;
  }
  if ( v10 > (unsigned __int64)v8 )
  {
LABEL_4:
    *(_QWORD *)(a2 + 32) = v8 + 1;
    *v8 = 60;
    goto LABEL_5;
  }
LABEL_13:
  sub_CB5D20(a2, 60);
LABEL_5:
  v12 = *(_BYTE **)(a2 + 32);
  v13 = a2;
  if ( !*a1 )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v12 > 2u )
    {
      v12[2] = 45;
      *(_WORD *)v12 = 28526;
      v12 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 3LL);
      *(_QWORD *)(a2 + 32) = v12;
    }
    else
    {
      v13 = sub_CB6200(a2, "no-", 3u);
      v12 = *(_BYTE **)(v13 + 32);
    }
  }
  if ( *(_QWORD *)(v13 + 24) - (_QWORD)v12 <= 0xEu )
  {
    sub_CB6200(v13, "split-footer-bb", 0xFu);
  }
  else
  {
    qmemcpy(v12, "split-footer-bb", 15);
    *(_QWORD *)(v13 + 32) += 15LL;
  }
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 62;
  return result;
}
