// Function: sub_27AF270
// Address: 0x27af270
//
__int64 __fastcall sub_27AF270(char *a1, __int64 a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  char *v8; // rbx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // [rsp-10h] [rbp-50h]
  __int64 v14; // [rsp+8h] [rbp-38h]

  v7 = (__int64)(0x8E38E38E38E38E39LL * ((a2 - (__int64)a1) >> 3) + 1) / 2;
  v14 = 72 * v7;
  v8 = &a1[72 * v7];
  if ( v7 <= a4 )
  {
    sub_27ACF00(a1, (__int64)&a1[72 * v7], a3, a4, a5, 72 * v7);
    sub_27ACF00(v8, a2, a3, v10, v11, v12);
  }
  else
  {
    sub_27AF270(a1, &a1[72 * v7], a3);
    sub_27AF270(v8, a2, a3);
  }
  sub_27AE860(
    (__int64)a1,
    (__int64)v8,
    a2,
    0x8E38E38E38E38E39LL * (v14 >> 3),
    0x8E38E38E38E38E39LL * ((a2 - (__int64)v8) >> 3),
    (__int64)a3,
    a4);
  return v13;
}
