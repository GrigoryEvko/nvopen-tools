// Function: sub_2353C00
// Address: 0x2353c00
//
unsigned __int64 __fastcall sub_2353C00(unsigned __int64 *a1, const __m128i *a2)
{
  __int32 v2; // eax
  __int64 v3; // rax
  __int64 v4; // rcx
  __int32 v5; // edx
  __m128i v6; // xmm0
  unsigned __int64 result; // rax
  __m128i v8; // [rsp+0h] [rbp-50h] BYREF
  __int64 v9; // [rsp+10h] [rbp-40h]
  __int32 v10; // [rsp+18h] [rbp-38h]
  __m128i v11; // [rsp+20h] [rbp-30h] BYREF
  __int64 v12; // [rsp+30h] [rbp-20h]
  __int32 v13; // [rsp+38h] [rbp-18h]

  v9 = a2[1].m128i_i64[0];
  v2 = a2[1].m128i_i32[2];
  v8 = _mm_loadu_si128(a2);
  v10 = v2;
  v3 = sub_22077B0(0x28u);
  if ( v3 )
  {
    v4 = v9;
    v5 = v10;
    v6 = _mm_loadu_si128(&v8);
    v12 = v9;
    v13 = v10;
    *(_QWORD *)v3 = &unk_4A119B8;
    *(_QWORD *)(v3 + 24) = v4;
    *(_DWORD *)(v3 + 32) = v5;
    v11 = v6;
    *(__m128i *)(v3 + 8) = v6;
  }
  v11.m128i_i64[0] = v3;
  result = sub_2353900(a1, (unsigned __int64 *)&v11);
  if ( v11.m128i_i64[0] )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11.m128i_i64[0] + 8LL))(v11.m128i_i64[0]);
  return result;
}
