// Function: sub_2AEE280
// Address: 0x2aee280
//
__int64 *__fastcall sub_2AEE280(__int64 a1, __int64 a2)
{
  __int64 i; // rsi
  __int64 v4; // r15
  _BYTE *v5; // rbx
  __int64 *v6; // rdi
  __int64 v7; // rax
  __int64 *result; // rax
  _BYTE *v9; // [rsp+0h] [rbp-B0h]
  __int64 v10[2]; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v11; // [rsp+18h] [rbp-98h] BYREF
  void *v12; // [rsp+20h] [rbp-90h] BYREF
  __int16 v13; // [rsp+40h] [rbp-70h]
  __int64 v14; // [rsp+50h] [rbp-60h] BYREF
  __int64 v15; // [rsp+58h] [rbp-58h]
  __int64 v16; // [rsp+60h] [rbp-50h]
  __int64 v17; // [rsp+68h] [rbp-48h]
  _BYTE *v18; // [rsp+70h] [rbp-40h]
  __int64 v19; // [rsp+78h] [rbp-38h]
  _BYTE v20[48]; // [rsp+80h] [rbp-30h] BYREF

  v10[0] = a2;
  for ( i = *(_QWORD *)(a2 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    if ( (unsigned __int8)(**(_BYTE **)(i + 24) - 30) <= 0xAu )
      break;
  }
  v4 = 0;
  v18 = v20;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v19 = 0;
  sub_2AEE0B0((__int64)&v14, i, 0);
  v5 = v18;
  v9 = &v18[8 * (unsigned int)v19];
  if ( v9 == v18 )
  {
LABEL_12:
    if ( v5 != v20 )
      _libc_free((unsigned __int64)v5);
    sub_C7D6A0(v15, 8LL * (unsigned int)v17, 8);
    result = sub_2AD5700(a1 + 96, v10);
    *result = v4;
  }
  else
  {
    while ( 1 )
    {
      v7 = sub_2AD5230(a1, *(_QWORD *)v5, v10[0]);
      if ( !v7 )
        break;
      if ( v4 )
      {
        v6 = *(__int64 **)(a1 + 56);
        v13 = 257;
        v11 = 0;
        v4 = sub_2AB0F70(v6, v4, v7, &v11, &v12);
        if ( v11 )
          sub_B91220((__int64)&v11, v11);
        v5 += 8;
        if ( v9 == v5 )
        {
LABEL_11:
          v5 = v18;
          goto LABEL_12;
        }
      }
      else
      {
        v4 = v7;
        v5 += 8;
        if ( v9 == v5 )
          goto LABEL_11;
      }
    }
    *sub_2AD5700(a1 + 96, v10) = 0;
    if ( v18 != v20 )
      _libc_free((unsigned __int64)v18);
    return (__int64 *)sub_C7D6A0(v15, 8LL * (unsigned int)v17, 8);
  }
  return result;
}
