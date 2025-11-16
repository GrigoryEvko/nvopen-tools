// Function: sub_2A1A1F0
// Address: 0x2a1a1f0
//
__int64 __fastcall sub_2A1A1F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // r8
  __int64 v12; // r9
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 v16; // rax
  __int64 *v17; // rdi
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v22; // [rsp+8h] [rbp-248h]
  __int64 v23; // [rsp+18h] [rbp-238h] BYREF
  __m128i v24; // [rsp+20h] [rbp-230h] BYREF
  _QWORD v25[4]; // [rsp+30h] [rbp-220h] BYREF
  __int64 v26; // [rsp+50h] [rbp-200h] BYREF
  _QWORD v27[10]; // [rsp+70h] [rbp-1E0h] BYREF
  unsigned __int64 *v28; // [rsp+C0h] [rbp-190h]
  unsigned int v29; // [rsp+C8h] [rbp-188h]
  char v30; // [rsp+D0h] [rbp-180h] BYREF

  v7 = **(_QWORD **)a2;
  v22 = **(_QWORD **)(v7 + 32);
  sub_D4BD20(&v23, v7, a3, a4, a5, v22);
  sub_B157E0((__int64)&v24, &v23);
  sub_B17430((__int64)v27, (__int64)"loop-unroll-and-jam", (__int64)"PartialUnrolled", 15, &v24, v22);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  sub_B18290((__int64)v27, "unroll and jammed loop by a factor of ", 0x26u);
  sub_B169E0(v24.m128i_i64, "UnrollCount", 11, **(_DWORD **)(a2 + 8));
  v10 = sub_23FD640((__int64)v27, (__int64)&v24);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(v10 + 8);
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(v10 + 12);
  v13 = _mm_loadu_si128((const __m128i *)(v10 + 24));
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(v10 + 16);
  *(__m128i *)(a1 + 24) = v13;
  v14 = _mm_loadu_si128((const __m128i *)(v10 + 48));
  v15 = _mm_loadu_si128((const __m128i *)(v10 + 64));
  *(_QWORD *)a1 = &unk_49D9D40;
  v16 = *(_QWORD *)(v10 + 40);
  *(__m128i *)(a1 + 48) = v14;
  *(_QWORD *)(a1 + 40) = v16;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(__m128i *)(a1 + 64) = v15;
  if ( *(_DWORD *)(v10 + 88) )
    sub_2A19F70(a1 + 80, v10 + 80, v8, v9, v11, v12);
  v17 = (__int64 *)v25[2];
  *(_BYTE *)(a1 + 416) = *(_BYTE *)(v10 + 416);
  *(_DWORD *)(a1 + 420) = *(_DWORD *)(v10 + 420);
  *(_QWORD *)(a1 + 424) = *(_QWORD *)(v10 + 424);
  *(_QWORD *)a1 = &unk_49D9D78;
  if ( v17 != &v26 )
    j_j___libc_free_0((unsigned __int64)v17);
  if ( (_QWORD *)v24.m128i_i64[0] != v25 )
    j_j___libc_free_0(v24.m128i_u64[0]);
  v18 = v28;
  v27[0] = &unk_49D9D40;
  v19 = &v28[10 * v29];
  if ( v28 != v19 )
  {
    do
    {
      v19 -= 10;
      v20 = v19[4];
      if ( (unsigned __int64 *)v20 != v19 + 6 )
        j_j___libc_free_0(v20);
      if ( (unsigned __int64 *)*v19 != v19 + 2 )
        j_j___libc_free_0(*v19);
    }
    while ( v18 != v19 );
    v19 = v28;
  }
  if ( v19 != (unsigned __int64 *)&v30 )
    _libc_free((unsigned __int64)v19);
  return a1;
}
