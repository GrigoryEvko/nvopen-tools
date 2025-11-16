// Function: sub_DA4270
// Address: 0xda4270
//
_QWORD *__fastcall sub_DA4270(__int64 a1, __int64 a2, int a3)
{
  __int64 *v5; // rsi
  _QWORD *v6; // rax
  _QWORD *v7; // r15
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  void *v12; // [rsp+0h] [rbp-E0h]
  __int64 v13; // [rsp+8h] [rbp-D8h]
  __int64 *v14; // [rsp+18h] [rbp-C8h] BYREF
  _QWORD v15[2]; // [rsp+20h] [rbp-C0h] BYREF
  int v16; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v17; // [rsp+34h] [rbp-ACh]
  int v18; // [rsp+3Ch] [rbp-A4h]

  v17 = a2;
  v5 = v15;
  v18 = a3;
  v15[0] = &v16;
  v15[1] = 0x2000000004LL;
  v16 = 2;
  v14 = 0;
  v6 = sub_C65B40(a1 + 1048, (__int64)v15, (__int64 *)&v14, (__int64)&off_49DEA60);
  if ( v6 )
  {
    v7 = v6 - 1;
  }
  else
  {
    v12 = sub_C65D30((__int64)v15, (unsigned __int64 *)(a1 + 1064));
    v13 = v9;
    v10 = sub_A777F0(0x38u, (__int64 *)(a1 + 1064));
    v7 = (_QWORD *)v10;
    if ( v10 )
    {
      sub_D9AB80(v10, (__int64)v12, v13, a2, a3);
      v11 = v14;
      v5 = v7 + 1;
    }
    else
    {
      v11 = v14;
      v5 = 0;
    }
    sub_C657C0((__int64 *)(a1 + 1048), v5, v11, (__int64)&off_49DEA60);
  }
  if ( (int *)v15[0] != &v16 )
    _libc_free(v15[0], v5);
  return v7;
}
