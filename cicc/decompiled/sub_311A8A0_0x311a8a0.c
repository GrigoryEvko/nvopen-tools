// Function: sub_311A8A0
// Address: 0x311a8a0
//
__int64 __fastcall sub_311A8A0(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5)
{
  __int64 v8; // rax
  char *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  unsigned __int64 *v14; // [rsp+8h] [rbp-38h]

  v8 = (a2 - a1 + 1) / 2;
  v13 = 8 * v8;
  v14 = &a1[v8];
  if ( v8 <= a4 )
  {
    sub_3119760(a1, &a1[v8], a3, a5);
    sub_3119760(v14, a2, a3, a5);
    v10 = v13;
    v9 = (char *)v14;
  }
  else
  {
    sub_311A8A0(a1, &a1[v8], a3);
    sub_311A8A0(v14, a2, a3);
    v9 = (char *)v14;
    v10 = v13;
  }
  sub_3119860((__int64)a1, v9, (__int64)a2, v10 >> 3, ((char *)a2 - v9) >> 3, a3, a4, a5);
  return v12;
}
