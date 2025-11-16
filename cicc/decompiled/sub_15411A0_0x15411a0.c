// Function: sub_15411A0
// Address: 0x15411a0
//
__int64 __fastcall sub_15411A0(__int64 a1, __int64 **a2, __int64 **a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r9
  char *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  __int64 v14; // [rsp+8h] [rbp-38h]

  v8 = 16 * (((((__int64)a2 - a1) >> 4) + 1) / 2);
  v13 = v8;
  v14 = a1 + v8;
  if ( ((((__int64)a2 - a1) >> 4) + 1) / 2 <= a4 )
  {
    sub_153D060(a1, (__int64 **)(a1 + v8), a3, a5);
    sub_153D060(v14, a2, a3, a5);
    v10 = v13;
    v9 = (char *)v14;
  }
  else
  {
    sub_15411A0(a1, a1 + v8, a3);
    sub_15411A0(v14, a2, a3);
    v9 = (char *)v14;
    v10 = v13;
  }
  sub_1540D70(a1, v9, (__int64)a2, v10 >> 4, ((char *)a2 - v9) >> 4, (__int64 *)a3, a4, a5);
  return v12;
}
