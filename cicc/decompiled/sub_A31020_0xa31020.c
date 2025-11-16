// Function: sub_A31020
// Address: 0xa31020
//
__int64 __fastcall sub_A31020(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 v6; // rsi
  __int64 result; // rax
  _BYTE *v9; // [rsp+10h] [rbp-110h] BYREF
  __int64 v10; // [rsp+18h] [rbp-108h]
  __int64 v11; // [rsp+20h] [rbp-100h]
  _BYTE v12[8]; // [rsp+28h] [rbp-F8h] BYREF
  __int64 v13[30]; // [rsp+30h] [rbp-F0h] BYREF

  v9 = v12;
  v10 = 0;
  v11 = 0;
  sub_C8D290(&v9, v12, 0x40000, 1);
  sub_A18170((__int64)v13, (__int64)&v9);
  sub_A300A0(v13, a1, a3, a4);
  sub_A1BA60((__int64)v13);
  sub_A1BEE0((__int64)v13);
  v6 = (__int64)v9;
  sub_CB6200(a2, v9, v10);
  result = sub_A18460((__int64)v13, v6);
  if ( v9 != v12 )
    return _libc_free(v9, v6);
  return result;
}
