// Function: sub_2882C00
// Address: 0x2882c00
//
void __fastcall sub_2882C00(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  unsigned __int64 *v17; // r15
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 *v23; // rbx
  __int64 v24; // r8
  unsigned __int64 v25; // rdi
  __int64 v26; // [rsp+8h] [rbp-3A8h]
  __int64 v27; // [rsp+18h] [rbp-398h] BYREF
  __m128i v28; // [rsp+20h] [rbp-390h] BYREF
  __int64 v29; // [rsp+30h] [rbp-380h] BYREF
  __m128i v30; // [rsp+38h] [rbp-378h]
  __int64 v31; // [rsp+48h] [rbp-368h]
  _OWORD v32[2]; // [rsp+50h] [rbp-360h] BYREF
  unsigned __int64 *v33; // [rsp+70h] [rbp-340h] BYREF
  __int64 v34; // [rsp+78h] [rbp-338h]
  _BYTE v35[320]; // [rsp+80h] [rbp-330h] BYREF
  char v36; // [rsp+1C0h] [rbp-1F0h]
  int v37; // [rsp+1C4h] [rbp-1ECh]
  __int64 v38; // [rsp+1C8h] [rbp-1E8h]
  void *v39; // [rsp+1D0h] [rbp-1E0h] BYREF
  __int32 v40; // [rsp+1D8h] [rbp-1D8h]
  __int8 v41; // [rsp+1DCh] [rbp-1D4h]
  __int64 v42; // [rsp+1E0h] [rbp-1D0h]
  __m128i v43; // [rsp+1E8h] [rbp-1C8h] BYREF
  __int64 v44; // [rsp+1F8h] [rbp-1B8h]
  __m128i v45; // [rsp+200h] [rbp-1B0h] BYREF
  __m128i v46; // [rsp+210h] [rbp-1A0h] BYREF
  unsigned __int64 *v47; // [rsp+220h] [rbp-190h] BYREF
  unsigned int v48; // [rsp+228h] [rbp-188h]
  char v49; // [rsp+230h] [rbp-180h] BYREF
  char v50; // [rsp+370h] [rbp-40h]
  int v51; // [rsp+374h] [rbp-3Ch]
  __int64 v52; // [rsp+378h] [rbp-38h]

  sub_D4BD20(&v27, a2, a3, a4, a5, a6);
  if ( a1 )
  {
    v8 = *a1;
    v26 = **(_QWORD **)(a2 + 32);
    v9 = sub_B2BE50(*a1);
    if ( sub_B6EA50(v9)
      || (v21 = sub_B2BE50(v8),
          v22 = sub_B6F970(v21),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v22 + 48LL))(v22)) )
    {
      sub_B157E0((__int64)&v28, &v27);
      sub_B17430((__int64)&v39, (__int64)"loop-unroll", (__int64)"computeRuntimeUnrollCount", 25, &v28, v26);
      sub_B18290((__int64)&v39, *(__int8 **)a3, *(_QWORD *)(a3 + 8));
      sub_B16430((__int64)&v28, "RuntimeUnrollVariable", 0x15u, byte_3F871B3, 0);
      sub_23FD640((__int64)&v39, (__int64)&v28);
      if ( (_OWORD *)v30.m128i_i64[1] != v32 )
        j_j___libc_free_0(v30.m128i_u64[1]);
      if ( (__int64 *)v28.m128i_i64[0] != &v29 )
        j_j___libc_free_0(v28.m128i_u64[0]);
      v14 = _mm_loadu_si128(&v43);
      v15 = _mm_loadu_si128(&v45);
      v16 = _mm_loadu_si128(&v46);
      v28.m128i_i32[2] = v40;
      v30 = v14;
      v28.m128i_i8[12] = v41;
      v32[0] = v15;
      v29 = v42;
      v32[1] = v16;
      v28.m128i_i64[0] = (__int64)&unk_49D9D40;
      v31 = v44;
      v33 = (unsigned __int64 *)v35;
      v34 = 0x400000000LL;
      if ( v48 )
      {
        sub_2882450((__int64)&v33, (__int64)&v47, v10, v11, v12, v13);
        v39 = &unk_49D9D40;
        v23 = v47;
        v36 = v50;
        v37 = v51;
        v38 = v52;
        v28.m128i_i64[0] = (__int64)&unk_49D9D78;
        v24 = 10LL * v48;
        v17 = &v47[v24];
        if ( v47 != &v47[v24] )
        {
          do
          {
            v17 -= 10;
            v25 = v17[4];
            if ( (unsigned __int64 *)v25 != v17 + 6 )
              j_j___libc_free_0(v25);
            if ( (unsigned __int64 *)*v17 != v17 + 2 )
              j_j___libc_free_0(*v17);
          }
          while ( v23 != v17 );
          v17 = v47;
        }
      }
      else
      {
        v17 = v47;
        v36 = v50;
        v37 = v51;
        v38 = v52;
        v28.m128i_i64[0] = (__int64)&unk_49D9D78;
      }
      if ( v17 != (unsigned __int64 *)&v49 )
        _libc_free((unsigned __int64)v17);
      sub_1049740(a1, (__int64)&v28);
      v18 = v33;
      v28.m128i_i64[0] = (__int64)&unk_49D9D40;
      v19 = &v33[10 * (unsigned int)v34];
      if ( v33 != v19 )
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
        v19 = v33;
      }
      if ( v19 != (unsigned __int64 *)v35 )
        _libc_free((unsigned __int64)v19);
    }
  }
  if ( v27 )
    sub_B91220((__int64)&v27, v27);
}
