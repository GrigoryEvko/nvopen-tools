// Function: sub_2354080
// Address: 0x2354080
//
void __fastcall sub_2354080(unsigned __int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rdx
  __m128i *v5; // rax
  __int64 v6; // rdx
  __m128i *v7; // [rsp+8h] [rbp-48h] BYREF
  __int64 v8; // [rsp+10h] [rbp-40h]
  __m128i *v9; // [rsp+18h] [rbp-38h]
  __int64 v10; // [rsp+20h] [rbp-30h]
  __m128i v11[2]; // [rsp+28h] [rbp-28h] BYREF

  v2 = *(_QWORD *)a2;
  v3 = *(_QWORD *)(a2 + 8);
  v9 = v11;
  v8 = v2;
  if ( v3 == a2 + 24 )
  {
    v11[0] = _mm_loadu_si128((const __m128i *)(a2 + 24));
  }
  else
  {
    v9 = (__m128i *)v3;
    v11[0].m128i_i64[0] = *(_QWORD *)(a2 + 24);
  }
  v4 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = a2 + 24;
  *(_QWORD *)(a2 + 16) = 0;
  *(_BYTE *)(a2 + 24) = 0;
  v10 = v4;
  v5 = (__m128i *)sub_22077B0(0x30u);
  if ( v5 )
  {
    v5->m128i_i64[0] = (__int64)&unk_4A10438;
    v5->m128i_i64[1] = v8;
    v5[1].m128i_i64[0] = (__int64)v5[2].m128i_i64;
    if ( v9 == v11 )
    {
      v5[2] = _mm_loadu_si128(v11);
    }
    else
    {
      v5[1].m128i_i64[0] = (__int64)v9;
      v5[2].m128i_i64[0] = v11[0].m128i_i64[0];
    }
    v6 = v10;
    v9 = v11;
    v10 = 0;
    v5[1].m128i_i64[1] = v6;
    v11[0].m128i_i8[0] = 0;
  }
  v7 = v5;
  sub_2353900(a1, (unsigned __int64 *)&v7);
  if ( v7 )
    (*(void (__fastcall **)(__m128i *))(v7->m128i_i64[0] + 8))(v7);
  if ( v9 != v11 )
    j_j___libc_free_0((unsigned __int64)v9);
}
