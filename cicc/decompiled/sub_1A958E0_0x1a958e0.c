// Function: sub_1A958E0
// Address: 0x1a958e0
//
__int64 __fastcall sub_1A958E0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 *v3; // r15
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 *v6; // r14
  __int64 v7; // rax
  _QWORD v9[2]; // [rsp+18h] [rbp-108h] BYREF
  __int64 v10; // [rsp+28h] [rbp-F8h] BYREF
  __m128i v11; // [rsp+30h] [rbp-F0h] BYREF
  _QWORD *v12; // [rsp+48h] [rbp-D8h]
  __m128i v13; // [rsp+90h] [rbp-90h] BYREF
  int v14; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD *v15; // [rsp+A8h] [rbp-78h]
  int *v16; // [rsp+B0h] [rbp-70h]
  int *v17; // [rsp+B8h] [rbp-68h]
  __int64 v18; // [rsp+C0h] [rbp-60h]
  __int64 v19; // [rsp+C8h] [rbp-58h]
  __int64 v20; // [rsp+D0h] [rbp-50h]
  __int64 v21; // [rsp+D8h] [rbp-48h]
  __int64 v22; // [rsp+E0h] [rbp-40h]
  __int64 v23; // [rsp+E8h] [rbp-38h]

  v1 = 0;
  v9[0] = a1;
  if ( a1 )
  {
    v2 = sub_1560250(v9);
    sub_1563030(&v11, v2);
    sub_1560700(&v11, 36);
    sub_1560700(&v11, 37);
    v10 = sub_1560250(v9);
    v3 = (__int64 *)sub_155EE30(&v10);
    v4 = sub_155EE40(&v10);
    while ( (__int64 *)v4 != v3 )
    {
      while ( 1 )
      {
        v5 = *v3;
        if ( sub_1642E20(*v3) )
          break;
        if ( (__int64 *)v4 == ++v3 )
          goto LABEL_7;
      }
      v13.m128i_i64[0] = 0;
      v14 = 0;
      ++v3;
      v15 = 0;
      v16 = &v14;
      v17 = &v14;
      v18 = 0;
      v19 = 0;
      v20 = 0;
      v21 = 0;
      v22 = 0;
      v23 = 0;
      sub_1562E30(&v13, v5);
      sub_1561FA0((__int64)&v11, &v13);
      sub_1A95860(v15);
    }
LABEL_7:
    v6 = (__int64 *)sub_1560170((__int64)v9);
    v7 = sub_1560BF0(v6, &v11);
    sub_1563030(&v13, v7);
    v1 = sub_1560CD0(v6, -1, &v13);
    sub_1A95860(v15);
    sub_1A95860(v12);
  }
  return v1;
}
