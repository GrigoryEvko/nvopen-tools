// Function: sub_1FDAF90
// Address: 0x1fdaf90
//
void __fastcall sub_1FDAF90(_QWORD *a1, __int64 a2)
{
  int v2; // eax
  __int32 v3; // eax
  __m128i v4; // xmm1
  int v5; // edx
  __int32 v6; // eax
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rax
  __m128i v10; // xmm3
  __m128i *v11; // rax
  __int64 v12; // rdx
  __int32 v13; // eax
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  __m128i v17; // xmm5
  __m128i *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // r12
  __int64 v22; // r8
  __int64 v23; // r13
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rax
  const __m128i *v27; // rbx
  const __m128i *v28; // r12
  const __m128i *v29; // rdx
  __int64 v30; // [rsp+8h] [rbp-1B8h]
  __m128i v31; // [rsp+10h] [rbp-1B0h] BYREF
  __m128i v32; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v33; // [rsp+30h] [rbp-190h]
  const __m128i *v34; // [rsp+40h] [rbp-180h] BYREF
  __int64 v35; // [rsp+48h] [rbp-178h]
  __m128i v36[2]; // [rsp+50h] [rbp-170h] BYREF
  __int64 v37; // [rsp+70h] [rbp-150h]

  v35 = 0x800000000LL;
  v2 = *(_DWORD *)(a2 + 20);
  v34 = v36;
  v3 = sub_1FD8F60(a1, *(_QWORD *)(a2 - 24LL * (v2 & 0xFFFFFFF)));
  v31.m128i_i64[0] = 0;
  v31.m128i_i32[2] = v3;
  v32 = 0u;
  v33 = 0;
  v4 = _mm_loadu_si128(&v32);
  v36[0] = _mm_loadu_si128(&v31);
  v37 = 0;
  v36[1] = v4;
  v5 = *(_DWORD *)(a2 + 20);
  LODWORD(v35) = 1;
  v6 = sub_1FD8F60(a1, *(_QWORD *)(a2 + 24 * (1LL - (v5 & 0xFFFFFFF))));
  v31.m128i_i64[0] = 0;
  v31.m128i_i32[2] = v6;
  v9 = (unsigned int)v35;
  v32 = 0u;
  v33 = 0;
  if ( (unsigned int)v35 >= HIDWORD(v35) )
  {
    sub_16CD150((__int64)&v34, v36, 0, 40, v7, v8);
    v9 = (unsigned int)v35;
  }
  v10 = _mm_loadu_si128(&v32);
  v11 = (__m128i *)((char *)v34 + 40 * v9);
  v12 = v33;
  *v11 = _mm_loadu_si128(&v31);
  v11[2].m128i_i64[0] = v12;
  v11[1] = v10;
  LODWORD(v12) = *(_DWORD *)(a2 + 20);
  LODWORD(v35) = v35 + 1;
  v13 = sub_1FD8F60(a1, *(_QWORD *)(a2 + 24 * (2 - (v12 & 0xFFFFFFF))));
  v31.m128i_i64[0] = 0;
  v31.m128i_i32[2] = v13;
  v16 = (unsigned int)v35;
  v32 = 0u;
  v33 = 0;
  if ( (unsigned int)v35 >= HIDWORD(v35) )
  {
    sub_16CD150((__int64)&v34, v36, 0, 40, v14, v15);
    v16 = (unsigned int)v35;
  }
  v17 = _mm_loadu_si128(&v32);
  v18 = (__m128i *)((char *)v34 + 40 * v16);
  v19 = v33;
  *v18 = _mm_loadu_si128(&v31);
  v18[2].m128i_i64[0] = v19;
  v18[1] = v17;
  v20 = a1[5];
  LODWORD(v35) = v35 + 1;
  v21 = *(__int64 **)(v20 + 792);
  v22 = *(_QWORD *)(v20 + 784);
  v23 = *(_QWORD *)(v22 + 56);
  v30 = v22;
  v24 = (__int64)sub_1E0B640(v23, *(_QWORD *)(a1[13] + 8LL) + 2048LL, a1 + 10, 0);
  sub_1DD5BA0((__int64 *)(v30 + 16), v24);
  v25 = *v21;
  v26 = *(_QWORD *)v24;
  *(_QWORD *)(v24 + 8) = v21;
  v25 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v24 = v25 | v26 & 7;
  *(_QWORD *)(v25 + 8) = v24;
  *v21 = v24 | *v21 & 7;
  v27 = v34;
  v28 = (const __m128i *)((char *)v34 + 40 * (unsigned int)v35);
  if ( v34 != v28 )
  {
    do
    {
      v29 = v27;
      v27 = (const __m128i *)((char *)v27 + 40);
      sub_1E1A9C0(v24, v23, v29);
    }
    while ( v28 != v27 );
    v28 = v34;
  }
  if ( v28 != v36 )
    _libc_free((unsigned __int64)v28);
}
