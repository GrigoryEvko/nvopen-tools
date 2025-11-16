// Function: sub_A73860
// Address: 0xa73860
//
unsigned __int64 __fastcall sub_A73860(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  unsigned __int64 v4; // rax
  __int32 v5; // edx
  __int64 v6[2]; // [rsp+20h] [rbp-30h] BYREF
  __m128i v7; // [rsp+30h] [rbp-20h] BYREF
  __int64 v8; // [rsp+40h] [rbp-10h]
  char v9; // [rsp+4Ch] [rbp-4h]

  v1 = sub_A733A0(a1, 88);
  v6[1] = v2;
  v6[0] = v1;
  if ( (_BYTE)v2 )
  {
    v4 = sub_A71E50(v6);
    v7.m128i_i8[12] = 1;
    v7.m128i_i64[0] = v4;
    v7.m128i_i32[2] = v5;
    return _mm_loadu_si128(&v7).m128i_u64[0];
  }
  else
  {
    v9 = 0;
    return v8;
  }
}
