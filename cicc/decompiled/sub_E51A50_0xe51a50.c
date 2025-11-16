// Function: sub_E51A50
// Address: 0xe51a50
//
_BYTE *__fastcall sub_E51A50(__int64 a1, char *a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  unsigned __int64 v7; // r13
  bool v8; // zf
  _BYTE *result; // rax
  __int64 v10; // rdi
  __int64 v11; // r14
  unsigned __int8 *v12; // rsi
  size_t v13; // rdx
  void *v14; // rdi

  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(_QWORD *)(v5 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 24) - v6) <= 6 )
  {
    sub_CB6200(v5, "\t.file\t", 7u);
  }
  else
  {
    *(_DWORD *)v6 = 1768304137;
    *(_WORD *)(v6 + 4) = 25964;
    *(_BYTE *)(v6 + 6) = 9;
    *(_QWORD *)(v5 + 32) += 7LL;
  }
  sub_E51560(a1, a2, a3, *(_QWORD *)(a1 + 304));
  v7 = *(_QWORD *)(a1 + 344);
  if ( v7 )
  {
    v11 = *(_QWORD *)(a1 + 304);
    v12 = *(unsigned __int8 **)(a1 + 336);
    v13 = *(_QWORD *)(a1 + 344);
    v14 = *(void **)(v11 + 32);
    if ( v7 > *(_QWORD *)(v11 + 24) - (_QWORD)v14 )
    {
      sub_CB6200(*(_QWORD *)(a1 + 304), v12, v13);
    }
    else
    {
      memcpy(v14, v12, v13);
      *(_QWORD *)(v11 + 32) += v7;
    }
  }
  v8 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v8 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v10 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v10 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v10 + 24) )
    return (_BYTE *)sub_CB5D20(v10, 10);
  *(_QWORD *)(v10 + 32) = result + 1;
  *result = 10;
  return result;
}
