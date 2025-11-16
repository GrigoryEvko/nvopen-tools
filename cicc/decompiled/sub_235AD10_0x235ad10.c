// Function: sub_235AD10
// Address: 0x235ad10
//
void __fastcall sub_235AD10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __m128i *v9; // rax
  __int64 v10; // rdx
  __m128i *v11; // [rsp+8h] [rbp-58h] BYREF
  __int64 v12; // [rsp+10h] [rbp-50h]
  __m128i *v13; // [rsp+18h] [rbp-48h]
  __int64 v14; // [rsp+20h] [rbp-40h]
  __m128i v15[3]; // [rsp+28h] [rbp-38h] BYREF

  sub_2332320(a1, 0, a3, a4, a5, a6);
  v6 = *(_QWORD *)a2;
  v7 = *(_QWORD *)(a2 + 8);
  v13 = v15;
  v12 = v6;
  if ( v7 == a2 + 24 )
  {
    v15[0] = _mm_loadu_si128((const __m128i *)(a2 + 24));
  }
  else
  {
    v13 = (__m128i *)v7;
    v15[0].m128i_i64[0] = *(_QWORD *)(a2 + 24);
  }
  v8 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = a2 + 24;
  *(_QWORD *)(a2 + 16) = 0;
  *(_BYTE *)(a2 + 24) = 0;
  v14 = v8;
  v9 = (__m128i *)sub_22077B0(0x30u);
  if ( v9 )
  {
    v9->m128i_i64[0] = (__int64)&unk_4A122F8;
    v9->m128i_i64[1] = v12;
    v9[1].m128i_i64[0] = (__int64)v9[2].m128i_i64;
    if ( v13 == v15 )
    {
      v9[2] = _mm_loadu_si128(v15);
    }
    else
    {
      v9[1].m128i_i64[0] = (__int64)v13;
      v9[2].m128i_i64[0] = v15[0].m128i_i64[0];
    }
    v10 = v14;
    v13 = v15;
    v14 = 0;
    v9[1].m128i_i64[1] = v10;
    v15[0].m128i_i8[0] = 0;
  }
  v11 = v9;
  sub_235ACD0((unsigned __int64 *)(a1 + 72), (unsigned __int64 *)&v11);
  if ( v11 )
    (*(void (__fastcall **)(__m128i *))(v11->m128i_i64[0] + 8))(v11);
  if ( v13 != v15 )
    j_j___libc_free_0((unsigned __int64)v13);
}
