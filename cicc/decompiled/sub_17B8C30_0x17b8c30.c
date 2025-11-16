// Function: sub_17B8C30
// Address: 0x17b8c30
//
__int64 sub_17B8C30()
{
  int v0; // edx
  __int64 v2; // rcx
  __m128i *v3; // rax
  __int64 v4; // rcx
  _QWORD v5[2]; // [rsp+10h] [rbp-50h] BYREF
  __m128i v6; // [rsp+20h] [rbp-40h] BYREF
  char *v7; // [rsp+30h] [rbp-30h] BYREF
  char v8; // [rsp+38h] [rbp-28h]
  char v9; // [rsp+39h] [rbp-27h]
  char v10; // [rsp+40h] [rbp-20h] BYREF

  if ( qword_4FA2B48 != 4 )
  {
    v7 = &v10;
    sub_17B71F0((__int64 *)&v7, "Invalid -default-gcov-version: ", (__int64)"");
    v3 = (__m128i *)sub_2241490(&v7, qword_4FA2B40, qword_4FA2B48, v2);
    v5[0] = &v6;
    if ( (__m128i *)v3->m128i_i64[0] == &v3[1] )
    {
      v6 = _mm_loadu_si128(v3 + 1);
    }
    else
    {
      v5[0] = v3->m128i_i64[0];
      v6.m128i_i64[0] = v3[1].m128i_i64[0];
    }
    v4 = v3->m128i_i64[1];
    v3[1].m128i_i8[0] = 0;
    v5[1] = v4;
    v3->m128i_i64[0] = (__int64)v3[1].m128i_i64;
    v3->m128i_i64[1] = 0;
    sub_16BD160((__int64)v5, 1u);
  }
  v0 = *(_DWORD *)qword_4FA2B40;
  v9 = byte_4FA2A60;
  LOWORD(v7) = 257;
  *(_DWORD *)((char *)&v7 + 2) = v0;
  HIWORD(v7) = 0;
  v8 = 1;
  return (__int64)v7;
}
