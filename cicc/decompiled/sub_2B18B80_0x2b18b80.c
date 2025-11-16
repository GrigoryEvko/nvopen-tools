// Function: sub_2B18B80
// Address: 0x2b18b80
//
__int64 __fastcall sub_2B18B80(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // [rsp-10h] [rbp-50h]
  __int64 v14; // [rsp+8h] [rbp-38h]

  v7 = (__int64)(0x8E38E38E38E38E39LL * ((a2 - (__int64)a1) >> 3) + 1) / 2;
  v14 = 72 * v7;
  v8 = &a1[9 * v7];
  if ( v7 <= a4 )
  {
    sub_2B11460(a1, (__int64)&a1[9 * v7], a3, a4, a5, 72 * v7);
    sub_2B11460(v8, a2, a3, v10, v11, v12);
  }
  else
  {
    sub_2B18B80(a1, &a1[9 * v7], a3);
    sub_2B18B80(v8, a2, a3);
  }
  sub_2B18360(
    (__int64)a1,
    (__int64)v8,
    a2,
    0x8E38E38E38E38E39LL * (v14 >> 3),
    0x8E38E38E38E38E39LL * ((a2 - (__int64)v8) >> 3),
    (__int64)a3,
    a4);
  return v13;
}
