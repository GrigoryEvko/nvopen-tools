// Function: sub_324E3B0
// Address: 0x324e3b0
//
void __fastcall sub_324E3B0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // [rsp+8h] [rbp-48h] BYREF
  __int64 v11[8]; // [rsp+10h] [rbp-40h] BYREF

  v4 = sub_324C6D0(a1, 33, a2, 0);
  v5 = sub_324E290(a1);
  sub_32494F0(a1, v4, 73, v5);
  v11[0] = (__int64)a1;
  v10 = sub_3247B20((__int64)a1);
  v11[1] = v4;
  v11[2] = (__int64)&v10;
  v6 = sub_AF2800(a3);
  sub_324A6E0(v11, 34, v6);
  v7 = sub_AF2780(a3);
  sub_324A6E0(v11, 55, v7);
  v8 = sub_AF2880(a3);
  sub_324A6E0(v11, 47, v8);
  v9 = sub_AF2900(a3);
  sub_324A6E0(v11, 81, v9);
}
