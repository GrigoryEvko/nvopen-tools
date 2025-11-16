// Function: sub_3873790
// Address: 0x3873790
//
__int64 __fastcall sub_3873790(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r9
  __int64 *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  __int64 *v14; // [rsp+8h] [rbp-38h]

  v8 = 2 * (((((char *)a2 - (char *)a1) >> 4) + 1) / 2);
  v13 = v8 * 8;
  v14 = &a1[v8];
  if ( ((((char *)a2 - (char *)a1) >> 4) + 1) / 2 <= a4 )
  {
    sub_386FD40(a1, &a1[v8], a3, a5);
    sub_386FD40(v14, a2, a3, a5);
    v10 = v13;
    v9 = v14;
  }
  else
  {
    sub_3873790(a1, &a1[v8], a3);
    sub_3873790(v14, a2, a3);
    v9 = v14;
    v10 = v13;
  }
  sub_3873350(a1, v9, (__int64)a2, v10 >> 4, ((char *)a2 - (char *)v9) >> 4, a3, a4, a5);
  return v12;
}
