// Function: sub_E52E40
// Address: 0xe52e40
//
_BYTE *__fastcall sub_E52E40(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  void *v4; // rdx
  unsigned __int64 v5; // r13
  bool v6; // zf
  _BYTE *result; // rax
  __int64 v8; // rdi
  __int64 v9; // r14
  unsigned __int8 *v10; // rsi
  size_t v11; // rdx
  void *v12; // rdi

  v3 = *(_QWORD *)(a1 + 304);
  v4 = *(void **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0xEu )
  {
    sub_CB6200(v3, "\t.cfi_startproc", 0xFu);
  }
  else
  {
    qmemcpy(v4, "\t.cfi_startproc", 15);
    *(_QWORD *)(v3 + 32) += 15LL;
  }
  if ( *(_BYTE *)(a2 + 81) )
    sub_904010(*(_QWORD *)(a1 + 304), " simple");
  v5 = *(_QWORD *)(a1 + 344);
  if ( v5 )
  {
    v9 = *(_QWORD *)(a1 + 304);
    v10 = *(unsigned __int8 **)(a1 + 336);
    v11 = *(_QWORD *)(a1 + 344);
    v12 = *(void **)(v9 + 32);
    if ( v5 > *(_QWORD *)(v9 + 24) - (_QWORD)v12 )
    {
      sub_CB6200(*(_QWORD *)(a1 + 304), v10, v11);
    }
    else
    {
      memcpy(v12, v10, v11);
      *(_QWORD *)(v9 + 32) += v5;
    }
  }
  v6 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v6 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v8 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v8 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v8 + 24) )
    return (_BYTE *)sub_CB5D20(v8, 10);
  *(_QWORD *)(v8 + 32) = result + 1;
  *result = 10;
  return result;
}
