// Function: sub_E96DC0
// Address: 0xe96dc0
//
_BYTE *__fastcall sub_E96DC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v4; // rdx
  _BYTE *v5; // rdi
  __int64 v6; // rax
  size_t *v7; // rsi
  size_t v8; // r13
  void *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdi
  _BYTE *result; // rax
  __int64 v13; // rax

  v2 = a2;
  v4 = *(_QWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 7u )
  {
    v13 = sub_CB6200(a2, "\t.csect ", 8u);
    v5 = *(_BYTE **)(v13 + 32);
    v2 = v13;
  }
  else
  {
    *v4 = 0x2074636573632E09LL;
    v5 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 8LL);
    *(_QWORD *)(a2 + 32) = v5;
  }
  v6 = *(_QWORD *)(a1 + 152);
  if ( (*(_BYTE *)(v6 + 8) & 1) != 0 )
  {
    v7 = *(size_t **)(v6 - 8);
    v8 = *v7;
    v9 = v7 + 3;
    if ( *(_QWORD *)(v2 + 24) - (_QWORD)v5 >= v8 )
    {
      if ( v8 )
      {
        memcpy(v5, v9, v8);
        v5 = (_BYTE *)(v8 + *(_QWORD *)(v2 + 32));
        *(_QWORD *)(v2 + 32) = v5;
        if ( v5 != *(_BYTE **)(v2 + 24) )
          goto LABEL_7;
        goto LABEL_12;
      }
    }
    else
    {
      v10 = sub_CB6200(v2, (unsigned __int8 *)v9, v8);
      v5 = *(_BYTE **)(v10 + 32);
      v2 = v10;
    }
  }
  if ( v5 != *(_BYTE **)(v2 + 24) )
  {
LABEL_7:
    *v5 = 44;
    ++*(_QWORD *)(v2 + 32);
    goto LABEL_8;
  }
LABEL_12:
  v2 = sub_CB6200(v2, (unsigned __int8 *)",", 1u);
LABEL_8:
  v11 = sub_CB59D0(v2, *(unsigned __int8 *)(a1 + 32));
  result = *(_BYTE **)(v11 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v11 + 24) )
    return (_BYTE *)sub_CB5D20(v11, 10);
  *(_QWORD *)(v11 + 32) = result + 1;
  *result = 10;
  return result;
}
