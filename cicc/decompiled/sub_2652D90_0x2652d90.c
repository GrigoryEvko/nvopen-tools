// Function: sub_2652D90
// Address: 0x2652d90
//
__int64 __fastcall sub_2652D90(char *a1, char *a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  char *v8; // rbx
  __int64 v9; // r9
  __int64 v10; // r10
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]

  v7 = (__int64)(0x8E38E38E38E38E39LL * ((a2 - a1) >> 3) + 1) / 2;
  v13 = 72 * v7;
  v8 = &a1[72 * v7];
  if ( v7 <= a4 )
  {
    sub_2652B00((__int64)a1, &a1[72 * v7], a3, a5);
    sub_2652B00((__int64)v8, a2, a3, a5);
    v10 = v13;
    v9 = a5;
  }
  else
  {
    sub_2652D90(a1, &a1[72 * v7], a3);
    sub_2652D90(v8, a2, a3);
    v9 = a5;
    v10 = v13;
  }
  sub_2651C40(
    a1,
    v8,
    (__int64)a2,
    0x8E38E38E38E38E39LL * (v10 >> 3),
    0x8E38E38E38E38E39LL * ((a2 - v8) >> 3),
    (__int64)a3,
    a4,
    v9);
  return v12;
}
