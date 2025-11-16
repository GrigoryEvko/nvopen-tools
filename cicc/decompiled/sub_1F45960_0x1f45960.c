// Function: sub_1F45960
// Address: 0x1f45960
//
__int64 __fastcall sub_1F45960(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 result; // rax
  __int64 v7; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v8; // [rsp+8h] [rbp-B8h]
  __m128i v9; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v10; // [rsp+20h] [rbp-A0h]
  __m128i v11; // [rsp+30h] [rbp-90h] BYREF
  __int16 v12; // [rsp+40h] [rbp-80h]
  __m128i v13[2]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v14; // [rsp+70h] [rbp-50h] BYREF
  char v15; // [rsp+80h] [rbp-40h]
  char v16; // [rsp+81h] [rbp-3Fh]
  __m128i v17[3]; // [rsp+90h] [rbp-30h] BYREF

  v7 = a1;
  v8 = a2;
  if ( !a2 )
    return 0;
  v2 = sub_163A1D0();
  result = sub_163A430(v2, v7, v8, v3, v4, v5);
  if ( !result )
  {
    v16 = 1;
    v14.m128i_i64[0] = (__int64)"\" pass is not registered.";
    v15 = 3;
    v11.m128i_i64[0] = (__int64)&v7;
    v12 = 261;
    v9.m128i_i64[0] = 34;
    v10 = 264;
    sub_14EC200(v13, &v9, &v11);
    sub_14EC200(v17, v13, &v14);
    sub_16BCFB0((__int64)v17, 1u);
  }
  return result;
}
