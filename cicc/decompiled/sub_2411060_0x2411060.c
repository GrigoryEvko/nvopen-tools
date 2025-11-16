// Function: sub_2411060
// Address: 0x2411060
//
void __fastcall sub_2411060(__int64 a1, unsigned int **a2, __int64 a3)
{
  char *v5; // rax
  signed __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  __int64 v8; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v9; // [rsp+20h] [rbp-80h] BYREF
  __int64 v10; // [rsp+28h] [rbp-78h]
  __int64 v11; // [rsp+30h] [rbp-70h]
  _QWORD v12[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v13; // [rsp+60h] [rbp-40h]

  v12[0] = a3;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  sub_240DEA0((__int64)&v9, v12);
  v13 = 257;
  v5 = (char *)sub_BD5D20(a3);
  v8 = sub_B33830((__int64)a2, v5, v6, (__int64)v12, 0, 0, 1);
  sub_240DEA0((__int64)&v9, &v8);
  v7 = *(_QWORD *)(a1 + 328);
  v13 = 257;
  sub_921880(a2, v7, *(_QWORD *)(a1 + 336), v9, (__int64)(v10 - v9) >> 3, (__int64)v12, 0);
  if ( v9 )
    j_j___libc_free_0(v9);
}
