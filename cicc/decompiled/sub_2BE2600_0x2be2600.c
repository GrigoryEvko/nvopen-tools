// Function: sub_2BE2600
// Address: 0x2be2600
//
__int64 __fastcall sub_2BE2600(__int64 a1, __int64 a2, __int64 a3)
{
  const __m128i *v4; // rcx
  const __m128i *v5; // r8
  unsigned __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __m128i *v9; // rdx
  const __m128i *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // r14
  unsigned __int64 v22; // r14
  void *v23; // rcx
  size_t v24; // rdx
  unsigned int v25; // r13d
  __int64 v26; // rbx
  unsigned __int64 v27; // r12
  unsigned __int64 v28; // rdi
  unsigned __int64 v30; // rsi
  __int64 v31; // rdx
  unsigned __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rsi
  unsigned __int64 v35; // [rsp+0h] [rbp-D0h] BYREF
  unsigned __int64 v36; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v37; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v38[3]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v39; // [rsp+38h] [rbp-98h]
  __int64 v40; // [rsp+40h] [rbp-90h]
  __int64 v41; // [rsp+48h] [rbp-88h]
  __int64 v42; // [rsp+50h] [rbp-80h]
  __int64 v43; // [rsp+58h] [rbp-78h]
  unsigned __int64 *v44; // [rsp+60h] [rbp-70h]
  unsigned __int64 v45; // [rsp+68h] [rbp-68h]
  __int64 v46; // [rsp+70h] [rbp-60h]
  __int64 v47; // [rsp+78h] [rbp-58h]
  unsigned __int64 v48; // [rsp+80h] [rbp-50h]
  __int64 v49; // [rsp+88h] [rbp-48h]
  __int64 v50; // [rsp+90h] [rbp-40h]
  unsigned __int64 v51; // [rsp+98h] [rbp-38h]
  __int64 v52; // [rsp+A0h] [rbp-30h]
  int v53; // [rsp+A8h] [rbp-28h]

  v4 = *(const __m128i **)(a1 + 8);
  v5 = *(const __m128i **)a1;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v6 = (char *)v4 - (char *)v5;
  if ( v4 == v5 )
  {
    v8 = 0;
  }
  else
  {
    if ( v6 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a2, a3);
    v7 = sub_22077B0((char *)v4 - (char *)v5);
    v4 = *(const __m128i **)(a1 + 8);
    v5 = *(const __m128i **)a1;
    v8 = v7;
  }
  v35 = v8;
  v36 = v8;
  v37 = v8 + v6;
  if ( v4 != v5 )
  {
    v9 = (__m128i *)v8;
    v10 = v5;
    do
    {
      if ( v9 )
      {
        *v9 = _mm_loadu_si128(v10);
        v9[1].m128i_i64[0] = v10[1].m128i_i64[0];
      }
      v10 = (const __m128i *)((char *)v10 + 24);
      v9 = (__m128i *)((char *)v9 + 24);
    }
    while ( v10 != v4 );
    v8 += 8 * ((unsigned __int64)((char *)&v10[-2].m128i_u64[1] - (char *)v5) >> 3) + 24;
  }
  v11 = *(_QWORD *)(a1 + 48);
  v12 = *(_QWORD *)(a1 + 24);
  v13 = *(_QWORD *)(a1 + 40);
  v39 = 0;
  v40 = v12;
  v14 = *(_DWORD *)(a1 + 136);
  v41 = v13;
  v42 = v11;
  v15 = *(_QWORD *)(v11 + 16);
  v44 = &v35;
  v43 = v15;
  v16 = *(_QWORD *)(v15 + 64) - *(_QWORD *)(v15 + 56);
  v36 = v8;
  memset(v38, 0, sizeof(v38));
  v17 = 0xAAAAAAAAAAAAAAABLL * (v16 >> 4);
  if ( v16 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v45 = 0;
  v18 = 0;
  v46 = 0;
  v19 = 16 * v17;
  v47 = 0;
  if ( v17 )
  {
    v20 = sub_22077B0(16 * v17);
    v18 = v20 + v19;
    v45 = v20;
    v47 = v20 + v19;
    do
    {
      if ( v20 )
      {
        *(_QWORD *)v20 = 0;
        *(_DWORD *)(v20 + 8) = 0;
      }
      v20 += 16;
    }
    while ( v18 != v20 );
    v15 = v43;
  }
  v46 = v18;
  v21 = *(_QWORD *)(v15 + 64) - *(_QWORD *)(v15 + 56);
  v48 = 0;
  v49 = 0;
  v22 = 0xAAAAAAAAAAAAAAABLL * (v21 >> 4);
  v50 = 0;
  v23 = (void *)sub_2207820(v22);
  if ( v23 && (__int64)(v22 - 1) >= 0 )
  {
    v24 = 1;
    if ( (__int64)(v22 - 2) >= -1 )
      v24 = v22;
    v23 = memset(v23, 0, v24);
  }
  v51 = (unsigned __int64)v23;
  v52 = a2;
  if ( (v14 & 0x80u) != 0 )
    v14 &= 0xFFFFFFFA;
  v53 = v14;
  v39 = v40;
  v25 = sub_2BE23C0((__int64)v38, 1u);
  if ( (_BYTE)v25 )
  {
    v30 = v35;
    v31 = 0;
    v32 = 0;
    if ( v36 != v35 )
    {
      do
      {
        v33 = v30 + v31;
        if ( *(_BYTE *)(v30 + v31 + 16) )
        {
          v34 = v31 + *(_QWORD *)a1;
          *(_QWORD *)v34 = *(_QWORD *)v33;
          *(_QWORD *)(v34 + 8) = *(_QWORD *)(v33 + 8);
          *(_BYTE *)(v34 + 16) = *(_BYTE *)(v33 + 16);
          v30 = v35;
        }
        ++v32;
        v31 += 24;
      }
      while ( v32 < 0xAAAAAAAAAAAAAAABLL * ((__int64)(v36 - v30) >> 3) );
    }
  }
  if ( v51 )
    j_j___libc_free_0_0(v51);
  v26 = v49;
  v27 = v48;
  if ( v49 != v48 )
  {
    do
    {
      v28 = *(_QWORD *)(v27 + 8);
      if ( v28 )
        j_j___libc_free_0(v28);
      v27 += 32LL;
    }
    while ( v26 != v27 );
    v27 = v48;
  }
  if ( v27 )
    j_j___libc_free_0(v27);
  if ( v45 )
    j_j___libc_free_0(v45);
  if ( v38[0] )
    j_j___libc_free_0(v38[0]);
  if ( v35 )
    j_j___libc_free_0(v35);
  return v25;
}
