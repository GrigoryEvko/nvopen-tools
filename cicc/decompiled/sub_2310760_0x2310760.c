// Function: sub_2310760
// Address: 0x2310760
//
_BYTE *__fastcall sub_2310760(unsigned __int8 *src, size_t n, __int64 a3)
{
  __int64 v4; // r12
  _WORD *v5; // rdx
  _BYTE *v6; // rdi
  _BYTE *result; // rax
  __int64 v8; // rax

  v4 = a3;
  v5 = *(_WORD **)(a3 + 32);
  if ( *(_QWORD *)(v4 + 24) - (_QWORD)v5 <= 1u )
  {
    v8 = sub_CB6200(v4, (unsigned __int8 *)"  ", 2u);
    v6 = *(_BYTE **)(v8 + 32);
    v4 = v8;
  }
  else
  {
    *v5 = 8224;
    v6 = (_BYTE *)(*(_QWORD *)(v4 + 32) + 2LL);
    *(_QWORD *)(v4 + 32) = v6;
  }
  result = *(_BYTE **)(v4 + 24);
  if ( result - v6 < n )
  {
    v4 = sub_CB6200(v4, src, n);
    result = *(_BYTE **)(v4 + 24);
    v6 = *(_BYTE **)(v4 + 32);
  }
  else if ( n )
  {
    memcpy(v6, src, n);
    result = *(_BYTE **)(v4 + 24);
    v6 = (_BYTE *)(n + *(_QWORD *)(v4 + 32));
    *(_QWORD *)(v4 + 32) = v6;
    if ( v6 != result )
      goto LABEL_6;
    return (_BYTE *)sub_CB6200(v4, (unsigned __int8 *)"\n", 1u);
  }
  if ( v6 != result )
  {
LABEL_6:
    *v6 = 10;
    ++*(_QWORD *)(v4 + 32);
    return result;
  }
  return (_BYTE *)sub_CB6200(v4, (unsigned __int8 *)"\n", 1u);
}
