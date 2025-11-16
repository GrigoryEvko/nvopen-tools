// Function: sub_A428E0
// Address: 0xa428e0
//
__int64 __fastcall sub_A428E0(__m128i *a1, __m128i *a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r9
  char *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  __m128i *v14; // [rsp+8h] [rbp-38h]

  v8 = (a2 - a1 + 1) / 2;
  v13 = v8 * 16;
  v14 = &a1[v8];
  if ( (a2 - a1 + 1) / 2 <= a4 )
  {
    sub_A3D5A0(a1, &a1[v8], a3, a5);
    sub_A3D5A0(v14, a2, a3, a5);
    v10 = v13;
    v9 = (char *)v14;
  }
  else
  {
    sub_A428E0(a1, &a1[v8], a3);
    sub_A428E0(v14, a2, a3);
    v9 = (char *)v14;
    v10 = v13;
  }
  sub_A424B0((__int64)a1, v9, (__int64)a2, v10 >> 4, ((char *)a2 - v9) >> 4, (__int64 *)a3, a4, a5);
  return v12;
}
