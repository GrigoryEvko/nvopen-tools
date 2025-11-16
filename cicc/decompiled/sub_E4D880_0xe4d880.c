// Function: sub_E4D880
// Address: 0xe4d880
//
_BYTE *__fastcall sub_E4D880(__int64 a1)
{
  unsigned __int64 v2; // r13
  bool v3; // zf
  _BYTE *result; // rax
  __int64 v5; // rdi
  __int64 v6; // r14
  unsigned __int8 *v7; // rsi
  size_t v8; // rdx
  void *v9; // rdi

  v2 = *(_QWORD *)(a1 + 344);
  if ( v2 )
  {
    v6 = *(_QWORD *)(a1 + 304);
    v7 = *(unsigned __int8 **)(a1 + 336);
    v8 = *(_QWORD *)(a1 + 344);
    v9 = *(void **)(v6 + 32);
    if ( v2 > *(_QWORD *)(v6 + 24) - (_QWORD)v9 )
    {
      sub_CB6200(v6, v7, v8);
    }
    else
    {
      memcpy(v9, v7, v8);
      *(_QWORD *)(v6 + 32) += v2;
    }
  }
  v3 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v3 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v5 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v5 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v5 + 24) )
    return (_BYTE *)sub_CB5D20(v5, 10);
  *(_QWORD *)(v5 + 32) = result + 1;
  *result = 10;
  return result;
}
