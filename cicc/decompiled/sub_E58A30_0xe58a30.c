// Function: sub_E58A30
// Address: 0xe58a30
//
_BYTE *__fastcall sub_E58A30(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r9
  char v5; // al
  unsigned __int64 v6; // rsi
  _BYTE *v7; // rdx
  unsigned __int64 v8; // r13
  bool v9; // zf
  _BYTE *result; // rax
  __int64 v11; // rdi
  __int64 v12; // r14
  unsigned __int8 *v13; // rsi
  size_t v14; // rdx
  void *v15; // rdi
  _QWORD v16[6]; // [rsp+0h] [rbp-30h] BYREF

  sub_E9DBF0();
  v16[0] = 46;
  v5 = a2 & 0x7F;
  v16[1] = 0;
  if ( a2 >> 7 )
  {
    v6 = a2 >> 7;
    v7 = (char *)v16 + 1;
    do
    {
      *v7++ = v5 | 0x80;
      v5 = v6 & 0x7F;
      v6 >>= 7;
    }
    while ( v6 );
  }
  else
  {
    v7 = (char *)v16 + 1;
  }
  *v7 = v5;
  sub_E4D060(
    *(_QWORD *)(a1 + 304),
    (char *)v16,
    (unsigned int)v7 - ((unsigned int)v16 + 1) + 2,
    (__int64)v16 + 1,
    v3,
    v4);
  v8 = *(_QWORD *)(a1 + 344);
  if ( v8 )
  {
    v12 = *(_QWORD *)(a1 + 304);
    v13 = *(unsigned __int8 **)(a1 + 336);
    v14 = *(_QWORD *)(a1 + 344);
    v15 = *(void **)(v12 + 32);
    if ( v8 > *(_QWORD *)(v12 + 24) - (_QWORD)v15 )
    {
      sub_CB6200(*(_QWORD *)(a1 + 304), v13, v14);
    }
    else
    {
      memcpy(v15, v13, v14);
      *(_QWORD *)(v12 + 32) += v8;
    }
  }
  v9 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v9 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v11 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v11 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v11 + 24) )
    return (_BYTE *)sub_CB5D20(v11, 10);
  *(_QWORD *)(v11 + 32) = result + 1;
  *result = 10;
  return result;
}
