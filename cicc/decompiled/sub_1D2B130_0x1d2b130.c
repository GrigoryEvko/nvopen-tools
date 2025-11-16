// Function: sub_1D2B130
// Address: 0x1d2b130
//
__int64 *__fastcall sub_1D2B130(
        _QWORD *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        unsigned int a8)
{
  __int64 *v9; // rax
  __int64 *v10; // r12
  int v12; // r11d
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int8 v15; // [rsp+0h] [rbp-F0h]
  __int64 v16; // [rsp+8h] [rbp-E8h]
  int v17; // [rsp+8h] [rbp-E8h]
  __int64 *v18; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v19; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v20; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v21[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v22[176]; // [rsp+40h] [rbp-B0h] BYREF

  v15 = a3;
  v21[1] = 0x2000000000LL;
  v20 = a6;
  v19 = a5;
  v21[0] = (unsigned __int64)v22;
  v16 = sub_1D29190((__int64)a1, a3, a4, a4, a5, a6);
  sub_16BD430((__int64)v21, 159);
  sub_16BD4C0((__int64)v21, v16);
  sub_16BD4C0((__int64)v21, v19);
  sub_16BD430((__int64)v21, v20);
  sub_16BD430((__int64)v21, a7);
  sub_16BD430((__int64)v21, a8);
  v18 = 0;
  v9 = sub_1D17920((__int64)a1, (__int64)v21, a2, (__int64 *)&v18);
  if ( v9 )
  {
    v10 = v9;
  }
  else
  {
    v10 = (__int64 *)a1[26];
    v12 = *(_DWORD *)(a2 + 8);
    v13 = v15;
    if ( v10 )
    {
      a1[26] = *v10;
    }
    else
    {
      v17 = *(_DWORD *)(a2 + 8);
      v14 = sub_145CBF0(a1 + 27, 112, 8);
      v12 = v17;
      v13 = v15;
      v10 = (__int64 *)v14;
    }
    sub_1D29050((__int64)v10, v12, (__int64 *)a2, v13, a4, a7, a8);
    sub_1D23B60((__int64)a1, (__int64)v10, (__int64)&v19, 1);
    sub_16BDA20(a1 + 40, v10, v18);
    sub_1D172A0((__int64)a1, (__int64)v10);
  }
  if ( (_BYTE *)v21[0] != v22 )
    _libc_free(v21[0]);
  return v10;
}
