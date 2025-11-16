// Function: sub_BBB9D0
// Address: 0xbbb9d0
//
__int64 __fastcall sub_BBB9D0(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v3; // r12
  __int64 v4; // rax
  size_t v5; // rdx
  _BYTE *v6; // rdi
  const void *v7; // rsi
  __int64 result; // rax
  size_t v9; // r13

  v2 = *(void **)(a1 + 32);
  v3 = a1;
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v2 <= 9u )
  {
    v3 = sub_CB6200(a1, "function \"", 10);
  }
  else
  {
    qmemcpy(v2, "function \"", 10);
    *(_QWORD *)(a1 + 32) += 10LL;
  }
  v4 = sub_BD5D20(a2);
  v6 = *(_BYTE **)(v3 + 32);
  v7 = (const void *)v4;
  result = *(_QWORD *)(v3 + 24);
  v9 = v5;
  if ( result - (__int64)v6 < v5 )
  {
    v3 = sub_CB6200(v3, v7, v5);
    result = *(_QWORD *)(v3 + 24);
    v6 = *(_BYTE **)(v3 + 32);
  }
  else if ( v5 )
  {
    memcpy(v6, v7, v5);
    result = *(_QWORD *)(v3 + 24);
    v6 = (_BYTE *)(v9 + *(_QWORD *)(v3 + 32));
    *(_QWORD *)(v3 + 32) = v6;
    if ( (_BYTE *)result != v6 )
      goto LABEL_6;
    return sub_CB6200(v3, "\"", 1);
  }
  if ( (_BYTE *)result != v6 )
  {
LABEL_6:
    *v6 = 34;
    ++*(_QWORD *)(v3 + 32);
    return result;
  }
  return sub_CB6200(v3, "\"", 1);
}
