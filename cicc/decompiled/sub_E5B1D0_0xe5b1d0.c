// Function: sub_E5B1D0
// Address: 0xe5b1d0
//
_BYTE *__fastcall sub_E5B1D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r15
  void *v6; // r14
  size_t v7; // rax
  void *v8; // rdi
  size_t v9; // r13
  size_t v10; // r13
  bool v11; // zf
  _BYTE *result; // rax
  __int64 v13; // r14
  unsigned __int8 *v14; // rsi
  void *v15; // rdi
  __int64 v16; // rdi

  sub_E98820(a1, a2, a3);
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v4 = *(_QWORD *)(a1 + 312);
  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(void **)(v4 + 72);
  if ( !v6 )
    goto LABEL_5;
  v7 = strlen(*(const char **)(v4 + 72));
  v8 = *(void **)(v5 + 32);
  v9 = v7;
  if ( v7 <= *(_QWORD *)(v5 + 24) - (_QWORD)v8 )
  {
    if ( v7 )
    {
      memcpy(v8, v6, v7);
      *(_QWORD *)(v5 + 32) += v9;
    }
LABEL_5:
    v10 = *(_QWORD *)(a1 + 344);
    if ( !v10 )
      goto LABEL_6;
LABEL_9:
    v13 = *(_QWORD *)(a1 + 304);
    v14 = *(unsigned __int8 **)(a1 + 336);
    v15 = *(void **)(v13 + 32);
    if ( v10 > *(_QWORD *)(v13 + 24) - (_QWORD)v15 )
    {
      sub_CB6200(*(_QWORD *)(a1 + 304), v14, v10);
    }
    else
    {
      memcpy(v15, v14, v10);
      *(_QWORD *)(v13 + 32) += v10;
    }
    goto LABEL_6;
  }
  sub_CB6200(v5, (unsigned __int8 *)v6, v7);
  v10 = *(_QWORD *)(a1 + 344);
  if ( v10 )
    goto LABEL_9;
LABEL_6:
  v11 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v11 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v16 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v16 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v16 + 24) )
    return (_BYTE *)sub_CB5D20(v16, 10);
  *(_QWORD *)(v16 + 32) = result + 1;
  *result = 10;
  return result;
}
