// Function: sub_2559000
// Address: 0x2559000
//
__int64 __fastcall sub_2559000(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  unsigned __int8 *v5; // r12
  __m128i v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  char v10; // [rsp+1Fh] [rbp-71h] BYREF
  unsigned __int64 v11; // [rsp+20h] [rbp-70h]
  __int64 v12; // [rsp+28h] [rbp-68h]
  __m128i v13[6]; // [rsp+30h] [rbp-60h] BYREF

  v5 = (unsigned __int8 *)a3;
  v10 = 0;
  v6.m128i_i64[0] = sub_250D2C0(a3, 0);
  v13[0] = v6;
  v7 = sub_2527850(a2, v13, a1, &v10, 2u);
  v11 = v7;
  v12 = v8;
  if ( !v10 )
  {
    if ( !(_BYTE)v8 )
    {
      sub_BED950((__int64)v13, a1 + 104, a4);
      v13[0].m128i_i8[8] = 0;
      return v13[0].m128i_i64[0];
    }
    v5 = (unsigned __int8 *)v7;
    if ( !v7 )
    {
      v13[0].m128i_i64[0] = 0;
      v13[0].m128i_i8[8] = 1;
      return v13[0].m128i_i64[0];
    }
  }
  if ( (unsigned int)*v5 - 12 > 1 )
  {
    v13[0].m128i_i64[0] = (__int64)v5;
    v13[0].m128i_i8[8] = 1;
  }
  else
  {
    sub_AE6EC0(a1 + 104, a4);
    v13[0].m128i_i8[8] = 0;
  }
  return v13[0].m128i_i64[0];
}
