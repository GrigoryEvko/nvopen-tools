// Function: sub_30D1240
// Address: 0x30d1240
//
__int64 __fastcall sub_30D1240(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  const __m128i *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __m128i v11; // xmm6
  __m128i *v12; // rdx
  __m128i v13; // xmm7
  __m128i v14; // xmm5
  __int64 v15; // rax
  __m128i *v16; // rax
  __m128i v17; // xmm6
  __m128i v18; // xmm7
  _QWORD *v19; // rcx
  __m128i v20; // xmm5
  __m128i v21; // xmm6
  __m128i v22; // xmm7
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  __m128i v25; // xmm7
  __m128i *v26; // rax
  _OWORD v27[4]; // [rsp+0h] [rbp-110h] BYREF
  __int64 v28; // [rsp+40h] [rbp-D0h]
  __m128i v29; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v30; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v31; // [rsp+70h] [rbp-A0h] BYREF
  __m128i v32; // [rsp+80h] [rbp-90h] BYREF
  __int64 v33; // [rsp+90h] [rbp-80h]
  char v34; // [rsp+98h] [rbp-78h] BYREF
  __m128i v35; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v36; // [rsp+B0h] [rbp-60h] BYREF
  __m128i v37; // [rsp+C0h] [rbp-50h] BYREF
  __m128i v38; // [rsp+D0h] [rbp-40h] BYREF
  __m128i v39; // [rsp+E0h] [rbp-30h]

  v6 = (const __m128i *)(*(__int64 (__fastcall **)(_QWORD))a4)(*(_QWORD *)(a4 + 8));
  v35 = _mm_loadu_si128(v6);
  v36 = _mm_loadu_si128(v6 + 1);
  v37 = _mm_loadu_si128(v6 + 2);
  v38 = _mm_loadu_si128(v6 + 3);
  v39 = _mm_loadu_si128(v6 + 4);
  if ( byte_5030DA8 || sub_DFDF40(a3, a1, a2) )
  {
    v7 = (*(__int64 (__fastcall **)(_QWORD, __int64))a4)(*(_QWORD *)(a4 + 8), a1);
    v8 = v7;
    if ( (_BYTE)qword_502FC28 )
    {
      v11 = _mm_loadu_si128((const __m128i *)(v7 + 24));
      v12 = &v29;
      v13 = _mm_loadu_si128((const __m128i *)(v7 + 40));
      v29 = _mm_loadu_si128((const __m128i *)(v7 + 8));
      v14 = _mm_loadu_si128((const __m128i *)(v7 + 56));
      v15 = *(_QWORD *)(v7 + 72);
      v30 = v11;
      v33 = v15;
      v16 = &v29;
      v31 = v13;
      v32 = v14;
      do
      {
        v16->m128i_i64[0] = ~v16->m128i_i64[0];
        v16 = (__m128i *)((char *)v16 + 8);
      }
      while ( v16 != (__m128i *)&v34 );
      v17 = _mm_loadu_si128(&v29);
      v18 = _mm_loadu_si128(&v30);
      v19 = v27;
      v20 = _mm_loadu_si128(&v31);
      v33 &= 0x7FFu;
      v27[0] = v17;
      v21 = _mm_loadu_si128(&v32);
      v27[1] = v18;
      v22 = _mm_loadu_si128((const __m128i *)&v35.m128i_u64[1]);
      v28 = v33;
      v27[2] = v20;
      v23 = _mm_loadu_si128((const __m128i *)&v36.m128i_u64[1]);
      v27[3] = v21;
      v24 = _mm_loadu_si128((const __m128i *)&v37.m128i_u64[1]);
      v29 = v22;
      v25 = _mm_loadu_si128((const __m128i *)&v38.m128i_u64[1]);
      v33 = v39.m128i_i64[1];
      v26 = &v29;
      v30 = v23;
      v31 = v24;
      v32 = v25;
      do
      {
        v26->m128i_i64[0] &= *v19;
        v26 = (__m128i *)((char *)v26 + 8);
        ++v19;
      }
      while ( v26 != (__m128i *)&v34 );
      while ( !v12->m128i_i64[0] )
      {
        v12 = (__m128i *)((char *)v12 + 8);
        if ( v12 == (__m128i *)&v34 )
          return sub_A75420(a1, a2);
      }
    }
    else
    {
      v9 = 0;
      while ( *(_QWORD *)(v8 + 8 * v9 + 8) == v35.m128i_i64[v9 + 1] )
      {
        if ( ++v9 == 9 )
          return sub_A75420(a1, a2);
      }
    }
  }
  return 0;
}
