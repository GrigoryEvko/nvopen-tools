// Function: sub_20C83A0
// Address: 0x20c83a0
//
__int64 __fastcall sub_20C83A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char *a5)
{
  char *v5; // rbx
  unsigned int v6; // r14d
  char v8; // [rsp+Fh] [rbp-E1h] BYREF
  __m128i v9; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD *v10; // [rsp+28h] [rbp-C8h]
  __m128i v11; // [rsp+70h] [rbp-80h] BYREF
  _QWORD *v12; // [rsp+88h] [rbp-68h]

  v5 = a5;
  if ( !a5 )
    v5 = &v8;
  *v5 = 1;
  sub_1562F70(&v9, *(_QWORD *)(a1 + 112), 0);
  sub_1562F70(&v11, *(_QWORD *)(a2 + 56), 0);
  sub_1560700(&v9, 20);
  sub_1560700(&v11, 20);
  if ( (v9.m128i_i64[0] & 0x400000000000000LL) == 0 )
  {
    if ( (v9.m128i_i64[0] & 0x10000000000LL) != 0 )
    {
      v6 = 0;
      if ( (v11.m128i_i8[5] & 1) == 0 )
        goto LABEL_6;
      *v5 = 0;
      sub_1560700(&v9, 40);
      sub_1560700(&v11, 40);
    }
    v6 = sub_1561B70(&v9, &v11);
    goto LABEL_6;
  }
  v6 = 0;
  if ( (v11.m128i_i8[7] & 4) != 0 )
  {
    *v5 = 0;
    sub_1560700(&v9, 58);
    sub_1560700(&v11, 58);
    v6 = sub_1561B70(&v9, &v11);
  }
LABEL_6:
  sub_20C7A40(v12);
  sub_20C7A40(v10);
  return v6;
}
