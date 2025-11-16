// Function: sub_289B890
// Address: 0x289b890
//
_BYTE *__fastcall sub_289B890(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r13
  _BYTE *result; // rax
  unsigned __int64 v13; // rax

  v6 = a3(a4, "LowerMatrixIntrinsicsPass]", 25);
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
    goto LABEL_8;
  }
  if ( !v7 )
    goto LABEL_3;
  memcpy(v8, v9, v7);
  v13 = *(_QWORD *)(a2 + 24);
  v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
  *(_QWORD *)(a2 + 32) = v8;
  if ( v13 > (unsigned __int64)v8 )
  {
LABEL_4:
    *(_QWORD *)(a2 + 32) = v8 + 1;
    *v8 = 60;
    if ( !*a1 )
      goto LABEL_5;
LABEL_9:
    sub_904010(a2, "minimal");
    result = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)result < *(_QWORD *)(a2 + 24) )
      goto LABEL_6;
    return (_BYTE *)sub_CB5D20(a2, 62);
  }
LABEL_8:
  sub_CB5D20(a2, 60);
  if ( *a1 )
    goto LABEL_9;
LABEL_5:
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result < *(_QWORD *)(a2 + 24) )
  {
LABEL_6:
    *(_QWORD *)(a2 + 32) = result + 1;
    *result = 62;
    return result;
  }
  return (_BYTE *)sub_CB5D20(a2, 62);
}
