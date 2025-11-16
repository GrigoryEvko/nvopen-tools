// Function: sub_7F8930
// Address: 0x7f8930
//
__int64 __fastcall sub_7F8930(char *a1, __int64 *a2)
{
  __int64 v2; // rax
  size_t v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __m128i v8[5]; // [rsp+10h] [rbp-50h] BYREF

  v2 = *a2;
  if ( !*a2 )
  {
    v8[0].m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v8[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    v8[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    v8[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
    v8[0].m128i_i64[1] = *(_QWORD *)&dword_4F077C8;
    v4 = strlen(a1);
    sub_878540(a1, v4);
    v2 = *(_QWORD *)(sub_7D5DD0(v8, 0, v5, v6, v7) + 88);
    *a2 = v2;
  }
  *(_BYTE *)(v2 + 198) |= 0x10u;
  return *a2;
}
