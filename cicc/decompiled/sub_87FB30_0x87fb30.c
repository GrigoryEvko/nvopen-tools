// Function: sub_87FB30
// Address: 0x87fb30
//
_QWORD *__fastcall sub_87FB30(char *src, __int64 a2, __int64 **a3)
{
  size_t v4; // rax
  __int64 *v5; // rax
  __m128i *v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 *v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 *v14; // r9
  size_t v16; // rax
  __int64 *v17; // rax
  __int64 v18; // [rsp+0h] [rbp-60h] BYREF
  __int64 v19; // [rsp+8h] [rbp-58h]
  __m128i v20; // [rsp+10h] [rbp-50h]
  __m128i v21; // [rsp+20h] [rbp-40h]
  __m128i v22; // [rsp+30h] [rbp-30h]

  if ( a2 )
  {
    sub_8602E0(4u, a2);
    v18 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v20 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v21 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v22 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v19 = *(_QWORD *)&dword_4F077C8;
    v4 = strlen(src);
    sub_878540(src, v4, &v18);
    v5 = sub_87EBB0(0x17u, v18, &dword_4F077C8);
    *a3 = v5;
    *((_DWORD *)v5 + 10) = 0;
    (*a3)[8] = a2;
    v6 = sub_726DA0(0);
    sub_877D80((__int64)v6, *a3);
    v6[5].m128i_i8[8] = v6[5].m128i_i8[8] & 0x8F | 0x20;
    (*a3)[11] = (__int64)v6;
    sub_7331A0((__int64)v6);
    sub_8602E0(3u, (__int64)v6);
    sub_863FC0(3, (__int64)v6, v7, v8, v9, v10);
  }
  else
  {
    v18 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v20 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v21 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v22 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v19 = *(_QWORD *)&dword_4F077C8;
    v16 = strlen(src);
    sub_878540(src, v16, &v18);
    v17 = sub_87EBB0(0x17u, v18, &dword_4F077C8);
    *a3 = v17;
    *((_DWORD *)v17 + 10) = 0;
    v6 = sub_726DA0(0);
    sub_877D80((__int64)v6, *a3);
    v6[5].m128i_i8[8] = v6[5].m128i_i8[8] & 0x8F | 0x20;
    (*a3)[11] = (__int64)v6;
    sub_7331A0((__int64)v6);
    sub_8602E0(3u, (__int64)v6);
  }
  return sub_863FC0(3, (__int64)v6, v11, v12, v13, v14);
}
