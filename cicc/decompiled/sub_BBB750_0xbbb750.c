// Function: sub_BBB750
// Address: 0xbbb750
//
__int64 __fastcall sub_BBB750(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v4; // rdx
  _BYTE *v5; // rdi
  __int64 result; // rax
  unsigned __int64 v7; // r13
  const void *v8; // rsi
  __int64 v9; // rax

  v2 = a1;
  v4 = *(_QWORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v4 <= 7u )
  {
    v9 = sub_CB6200(a1, "module \"", 8);
    v5 = *(_BYTE **)(v9 + 32);
    v2 = v9;
  }
  else
  {
    *v4 = 0x2220656C75646F6DLL;
    v5 = (_BYTE *)(*(_QWORD *)(a1 + 32) + 8LL);
    *(_QWORD *)(v2 + 32) = v5;
  }
  result = *(_QWORD *)(v2 + 24);
  v7 = *(_QWORD *)(a2 + 176);
  v8 = *(const void **)(a2 + 168);
  if ( v7 > result - (__int64)v5 )
  {
    v2 = sub_CB6200(v2, v8, *(_QWORD *)(a2 + 176));
    result = *(_QWORD *)(v2 + 24);
    v5 = *(_BYTE **)(v2 + 32);
  }
  else if ( v7 )
  {
    memcpy(v5, v8, *(_QWORD *)(a2 + 176));
    result = *(_QWORD *)(v2 + 24);
    v5 = (_BYTE *)(v7 + *(_QWORD *)(v2 + 32));
    *(_QWORD *)(v2 + 32) = v5;
    if ( (_BYTE *)result != v5 )
      goto LABEL_6;
    return sub_CB6200(v2, "\"", 1);
  }
  if ( (_BYTE *)result != v5 )
  {
LABEL_6:
    *v5 = 34;
    ++*(_QWORD *)(v2 + 32);
    return result;
  }
  return sub_CB6200(v2, "\"", 1);
}
