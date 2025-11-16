// Function: sub_2B5D640
// Address: 0x2b5d640
//
__int64 __fastcall sub_2B5D640(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // r9
  __int64 *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = 2 * (((((char *)a2 - (char *)a1) >> 4) + 1) / 2);
  v10 = v6 * 8;
  v7 = &a1[v6];
  if ( ((((char *)a2 - (char *)a1) >> 4) + 1) / 2 <= a4 )
  {
    sub_2B0E610(a1, &a1[v6], a3);
    sub_2B0E610(v7, a2, a3);
  }
  else
  {
    sub_2B5D640(a1, &a1[v6], a3);
    sub_2B5D640(v7, a2, a3);
  }
  sub_2B5D270((__int64)a1, (__int64)v7, (__int64)a2, v10 >> 4, ((char *)a2 - (char *)v7) >> 4, a3, a4);
  return v9;
}
