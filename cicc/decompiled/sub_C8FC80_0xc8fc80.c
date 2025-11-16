// Function: sub_C8FC80
// Address: 0xc8fc80
//
_QWORD *__fastcall sub_C8FC80(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // r12
  _BYTE *v6; // rsi
  _QWORD *v7; // [rsp+8h] [rbp-48h]
  char v8[49]; // [rsp+1Fh] [rbp-31h] BYREF

  result = (_QWORD *)sub_22077B0(24);
  if ( result )
  {
    *result = 0;
    result[1] = 0;
    result[2] = 0;
  }
  v3 = *(_QWORD *)(a2 + 8);
  v4 = 0;
  v5 = *(_QWORD *)(a2 + 16) - v3;
  if ( v5 )
  {
    do
    {
      while ( *(_BYTE *)(v3 + v4) != 10 )
      {
LABEL_5:
        if ( v5 == ++v4 )
          goto LABEL_11;
      }
      v8[0] = v4;
      v6 = (_BYTE *)result[1];
      if ( v6 == (_BYTE *)result[2] )
      {
        v7 = result;
        sub_C8FB10((__int64)result, v6, v8);
        result = v7;
        goto LABEL_5;
      }
      if ( v6 )
      {
        *v6 = v4;
        v6 = (_BYTE *)result[1];
      }
      ++v4;
      result[1] = v6 + 1;
    }
    while ( v5 != v4 );
  }
LABEL_11:
  *a1 = result;
  return result;
}
