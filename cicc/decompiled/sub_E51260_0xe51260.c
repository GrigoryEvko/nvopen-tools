// Function: sub_E51260
// Address: 0xe51260
//
_BYTE *__fastcall sub_E51260(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // rax
  __int64 v6; // r13
  unsigned __int8 *v7; // rsi
  size_t v8; // r15
  void *v9; // rdi
  unsigned __int64 v10; // r13
  bool v11; // zf
  _BYTE *result; // rax
  __int64 v13; // rdi
  __int64 v14; // r14
  unsigned __int8 *v15; // rsi
  size_t v16; // rdx
  void *v17; // rdi
  __int64 v18; // rdi
  _BYTE *v19; // rax

  if ( a3 )
  {
    v18 = *(_QWORD *)(a1 + 304);
    v19 = *(_BYTE **)(v18 + 32);
    if ( (unsigned __int64)v19 >= *(_QWORD *)(v18 + 24) )
    {
      sub_CB5D20(v18, 9);
    }
    else
    {
      *(_QWORD *)(v18 + 32) = v19 + 1;
      *v19 = 9;
    }
  }
  v5 = *(_QWORD *)(a1 + 312);
  v6 = *(_QWORD *)(a1 + 304);
  v7 = *(unsigned __int8 **)(v5 + 48);
  v8 = *(_QWORD *)(v5 + 56);
  v9 = *(void **)(v6 + 32);
  if ( v8 > *(_QWORD *)(v6 + 24) - (_QWORD)v9 )
  {
    v6 = sub_CB6200(*(_QWORD *)(a1 + 304), v7, v8);
  }
  else if ( v8 )
  {
    memcpy(v9, v7, v8);
    *(_QWORD *)(v6 + 32) += v8;
  }
  sub_CA0E80(a2, v6);
  v10 = *(_QWORD *)(a1 + 344);
  if ( v10 )
  {
    v14 = *(_QWORD *)(a1 + 304);
    v15 = *(unsigned __int8 **)(a1 + 336);
    v16 = *(_QWORD *)(a1 + 344);
    v17 = *(void **)(v14 + 32);
    if ( v10 > *(_QWORD *)(v14 + 24) - (_QWORD)v17 )
    {
      sub_CB6200(*(_QWORD *)(a1 + 304), v15, v16);
    }
    else
    {
      memcpy(v17, v15, v16);
      *(_QWORD *)(v14 + 32) += v10;
    }
  }
  v11 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v11 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v13 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v13 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v13 + 24) )
    return (_BYTE *)sub_CB5D20(v13, 10);
  *(_QWORD *)(v13 + 32) = result + 1;
  *result = 10;
  return result;
}
