// Function: sub_1D94C00
// Address: 0x1d94c00
//
__int64 __fastcall sub_1D94C00(char *a1, char *a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  char *v14; // [rsp+8h] [rbp-38h]

  v8 = (((a2 - a1) >> 3) + 1) / 2;
  v13 = 8 * v8;
  v14 = &a1[8 * v8];
  if ( v8 <= a4 )
  {
    sub_1D94030(a1, &a1[8 * v8], a3, a5);
    sub_1D94030(v14, a2, a3, a5);
    v10 = v13;
    v9 = (__int64 *)v14;
  }
  else
  {
    sub_1D94C00(a1, &a1[8 * v8], a3);
    sub_1D94C00(v14, a2, a3);
    v9 = (__int64 *)v14;
    v10 = v13;
  }
  sub_1D94260((__int64 *)a1, v9, (__int64)a2, v10 >> 3, (a2 - (char *)v9) >> 3, (char *)a3, a4, a5);
  return v12;
}
