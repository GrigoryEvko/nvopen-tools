// Function: sub_254EFB0
// Address: 0x254efb0
//
__int64 __fastcall sub_254EFB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  char v4; // [rsp+2Bh] [rbp-95h] BYREF
  int v5; // [rsp+2Ch] [rbp-94h] BYREF
  __m128i v6; // [rsp+30h] [rbp-90h] BYREF
  __m128i v7; // [rsp+40h] [rbp-80h]
  __int64 *v8; // [rsp+50h] [rbp-70h] BYREF
  __int64 v9; // [rsp+58h] [rbp-68h]
  _BYTE v10[16]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v11[10]; // [rsp+70h] [rbp-50h] BYREF

  v9 = 0x100000000LL;
  v4 = 0;
  v8 = (__int64 *)v10;
  LODWORD(v11[0]) = 81;
  sub_2515D00(a2, (__m128i *)(a1 + 72), (int *)v11, 1, (__int64)&v8, 1);
  if ( (_DWORD)v9
    && (unsigned __int8)sub_2523890(
                          a2,
                          (__int64 (__fastcall *)(__int64, __int64 *))sub_253A490,
                          (__int64)v11,
                          a1,
                          1u,
                          &v4) )
  {
    v3 = sub_A72A60(v8);
    v7.m128i_i8[8] = 1;
    v7.m128i_i64[0] = v3;
  }
  else
  {
    v6 = 0;
    v5 = sub_250CB50((__int64 *)(a1 + 72), 0);
    v11[0] = &v5;
    v11[1] = a2;
    v11[2] = a1;
    v11[3] = &v6;
    if ( (unsigned __int8)sub_2523890(
                            a2,
                            (__int64 (__fastcall *)(__int64, __int64 *))sub_2587840,
                            (__int64)v11,
                            a1,
                            1u,
                            &v4) )
    {
      v7 = _mm_loadu_si128(&v6);
    }
    else
    {
      v7.m128i_i64[0] = 0;
      v7.m128i_i8[8] = 1;
    }
  }
  if ( v8 != (__int64 *)v10 )
    _libc_free((unsigned __int64)v8);
  return v7.m128i_i64[0];
}
