// Function: sub_E6C840
// Address: 0xe6c840
//
__int64 __fastcall sub_E6C840(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  char v6; // al
  __m128i v8; // xmm1
  __m128i v9; // [rsp+0h] [rbp-60h] BYREF
  __m128i v10; // [rsp+10h] [rbp-50h] BYREF
  __int64 v11; // [rsp+20h] [rbp-40h]
  __m128i v12; // [rsp+30h] [rbp-30h] BYREF
  __m128i v13; // [rsp+40h] [rbp-20h]
  __int64 v14; // [rsp+50h] [rbp-10h]

  v4 = *(_QWORD *)(a1 + 152);
  LOWORD(v11) = 773;
  v5 = *(_QWORD *)(v4 + 88);
  v9.m128i_i64[1] = *(_QWORD *)(v4 + 96);
  v10.m128i_i64[0] = (__int64)"__ehtable$";
  v6 = *(_BYTE *)(a2 + 32);
  v9.m128i_i64[0] = v5;
  if ( !v6 )
  {
    LOWORD(v14) = 256;
    return sub_E6C460(a1, (const char **)&v12);
  }
  if ( v6 == 1 )
  {
    v8 = _mm_loadu_si128(&v10);
    v12 = _mm_loadu_si128(&v9);
    v14 = v11;
    v13 = v8;
    return sub_E6C460(a1, (const char **)&v12);
  }
  if ( *(_BYTE *)(a2 + 33) == 1 )
  {
    a4 = *(_QWORD *)(a2 + 8);
    a2 = *(_QWORD *)a2;
  }
  else
  {
    v6 = 2;
  }
  v13.m128i_i64[0] = a2;
  v12.m128i_i64[0] = (__int64)&v9;
  v13.m128i_i64[1] = a4;
  LOBYTE(v14) = 2;
  BYTE1(v14) = v6;
  return sub_E6C460(a1, (const char **)&v12);
}
