// Function: sub_1DE9F70
// Address: 0x1de9f70
//
__int64 __fastcall sub_1DE9F70(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // r9
  __int64 *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = 2 * ((((a2 - (__int64)a1) >> 4) + 1) / 2);
  v10 = v6 * 8;
  v7 = &a1[v6];
  if ( (((a2 - (__int64)a1) >> 4) + 1) / 2 <= a4 )
  {
    sub_1DE3DB0(a1, (__int64)&a1[v6], a3);
    sub_1DE3DB0(v7, a2, a3);
  }
  else
  {
    sub_1DE9F70(a1, &a1[v6], a3);
    sub_1DE9F70(v7, a2, a3);
  }
  sub_1DE9BB0((__int64)a1, (__int64)v7, a2, v10 >> 4, (a2 - (__int64)v7) >> 4, (__int64)a3, a4);
  return v9;
}
