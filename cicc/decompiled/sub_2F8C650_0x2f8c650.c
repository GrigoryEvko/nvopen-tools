// Function: sub_2F8C650
// Address: 0x2f8c650
//
__int64 __fastcall sub_2F8C650(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 *v9; // r15
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // [rsp-10h] [rbp-50h]

  v7 = (0x2E8BA2E8BA2E8BA3LL * ((a2 - (__int64)a1) >> 3) + 1) / 2;
  v8 = 88 * v7;
  v9 = &a1[11 * v7];
  if ( v7 <= a4 )
  {
    sub_2F8B330(a1, (__int64)&a1[11 * v7], a3, a4, a5, a6);
    sub_2F8B330(v9, a2, a3, v11, v12, v13);
  }
  else
  {
    sub_2F8C650(a1, &a1[11 * v7], a3);
    sub_2F8C650(v9, a2, a3);
  }
  sub_2F8BC60(
    (__int64)a1,
    (__int64)v9,
    a2,
    0x2E8BA2E8BA2E8BA3LL * (v8 >> 3),
    0x2E8BA2E8BA2E8BA3LL * ((a2 - (__int64)v9) >> 3),
    (__int64)a3,
    a4);
  return v14;
}
