// Function: sub_1499A20
// Address: 0x1499a20
//
__int64 __fastcall sub_1499A20(__int64 a1, __int64 a2, _QWORD *a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r12
  __int64 v7; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v8[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v9; // [rsp+20h] [rbp-40h]
  __int64 v10; // [rsp+28h] [rbp-38h]
  int v11; // [rsp+30h] [rbp-30h]
  int v12; // [rsp+38h] [rbp-28h]
  bool (__fastcall *v13)(__int64 *, __int64); // [rsp+40h] [rbp-20h]
  __int64 *v14; // [rsp+48h] [rbp-18h]

  v7 = a2;
  v13 = sub_1498EE0;
  v8[0] = a3;
  v8[1] = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 1;
  v14 = &v7;
  v5 = sub_1498F00(v8, a1, a4, a5);
  j___libc_free_0(v9);
  return v5;
}
