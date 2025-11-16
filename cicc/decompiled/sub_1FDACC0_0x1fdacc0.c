// Function: sub_1FDACC0
// Address: 0x1fdacc0
//
void __fastcall sub_1FDACC0(_QWORD *a1, __int64 a2)
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
  __int64 v13; // rax
  __int64 *v14; // r12
  __int64 v15; // r8
  __int64 v16; // r13
  __int64 v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rax
  const __m128i *v20; // rbx
  const __m128i *v21; // r12
  const __m128i *v22; // rdx
  __int64 v23; // [rsp+8h] [rbp-1B8h]
  __m128i v24; // [rsp+10h] [rbp-1B0h] BYREF
  __m128i v25; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v26; // [rsp+30h] [rbp-190h]
  const __m128i *v27; // [rsp+40h] [rbp-180h] BYREF
  __int64 v28; // [rsp+48h] [rbp-178h]
  __m128i v29[2]; // [rsp+50h] [rbp-170h] BYREF
  __int64 v30; // [rsp+70h] [rbp-150h]

  v28 = 0x800000000LL;
  v2 = *(_DWORD *)(a2 + 20);
  v27 = v29;
  v3 = sub_1FD8F60(a1, *(_QWORD *)(a2 - 24LL * (v2 & 0xFFFFFFF)));
  v24.m128i_i64[0] = 0;
  v24.m128i_i32[2] = v3;
  v25 = 0u;
  v26 = 0;
  v4 = _mm_loadu_si128(&v25);
  v29[0] = _mm_loadu_si128(&v24);
  v30 = 0;
  v29[1] = v4;
  v5 = *(_DWORD *)(a2 + 20);
  LODWORD(v28) = 1;
  v6 = sub_1FD8F60(a1, *(_QWORD *)(a2 + 24 * (1LL - (v5 & 0xFFFFFFF))));
  v24.m128i_i64[0] = 0;
  v24.m128i_i32[2] = v6;
  v9 = (unsigned int)v28;
  v25 = 0u;
  v26 = 0;
  if ( (unsigned int)v28 >= HIDWORD(v28) )
  {
    sub_16CD150((__int64)&v27, v29, 0, 40, v7, v8);
    v9 = (unsigned int)v28;
  }
  v10 = _mm_loadu_si128(&v25);
  v11 = (__m128i *)((char *)v27 + 40 * v9);
  v12 = v26;
  *v11 = _mm_loadu_si128(&v24);
  v11[2].m128i_i64[0] = v12;
  v11[1] = v10;
  v13 = a1[5];
  LODWORD(v28) = v28 + 1;
  v14 = *(__int64 **)(v13 + 792);
  v15 = *(_QWORD *)(v13 + 784);
  v16 = *(_QWORD *)(v15 + 56);
  v23 = v15;
  v17 = (__int64)sub_1E0B640(v16, *(_QWORD *)(a1[13] + 8LL) + 1984LL, a1 + 10, 0);
  sub_1DD5BA0((__int64 *)(v23 + 16), v17);
  v18 = *v14;
  v19 = *(_QWORD *)v17;
  *(_QWORD *)(v17 + 8) = v14;
  v18 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v17 = v18 | v19 & 7;
  *(_QWORD *)(v18 + 8) = v17;
  *v14 = v17 | *v14 & 7;
  v20 = v27;
  v21 = (const __m128i *)((char *)v27 + 40 * (unsigned int)v28);
  if ( v27 != v21 )
  {
    do
    {
      v22 = v20;
      v20 = (const __m128i *)((char *)v20 + 40);
      sub_1E1A9C0(v17, v16, v22);
    }
    while ( v21 != v20 );
    v21 = v27;
  }
  if ( v21 != v29 )
    _libc_free((unsigned __int64)v21);
}
