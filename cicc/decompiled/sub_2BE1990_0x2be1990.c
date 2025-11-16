// Function: sub_2BE1990
// Address: 0x2be1990
//
__int64 __fastcall sub_2BE1990(__int64 a1, __int64 a2, __int64 a3)
{
  const __m128i *v5; // rcx
  const __m128i *v6; // r8
  unsigned __int64 v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __m128i *v10; // rdx
  const __m128i *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  const __m128i **v14; // rsi
  __int64 v15; // rdx
  int v16; // r14d
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  unsigned int v23; // r13d
  unsigned __int64 v25; // rsi
  __int64 v26; // rdx
  unsigned __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rsi
  unsigned __int64 v30; // [rsp+0h] [rbp-C0h] BYREF
  unsigned __int64 v31; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v32; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v33[3]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v34; // [rsp+38h] [rbp-88h]
  __int64 v35; // [rsp+40h] [rbp-80h]
  __int64 v36; // [rsp+48h] [rbp-78h]
  __int64 v37; // [rsp+50h] [rbp-70h]
  __int64 v38; // [rsp+58h] [rbp-68h]
  unsigned __int64 *v39; // [rsp+60h] [rbp-60h]
  unsigned __int64 v40; // [rsp+68h] [rbp-58h]
  __int64 v41; // [rsp+70h] [rbp-50h]
  __int64 v42; // [rsp+78h] [rbp-48h]
  __int64 v43; // [rsp+80h] [rbp-40h]
  __int64 v44; // [rsp+88h] [rbp-38h]
  int v45; // [rsp+90h] [rbp-30h]
  unsigned __int8 v46; // [rsp+94h] [rbp-2Ch]

  v5 = *(const __m128i **)(a1 + 8);
  v6 = *(const __m128i **)a1;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v7 = (char *)v5 - (char *)v6;
  if ( v5 == v6 )
  {
    v9 = 0;
  }
  else
  {
    if ( v7 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a2, a3);
    v8 = sub_22077B0((char *)v5 - (char *)v6);
    v5 = *(const __m128i **)(a1 + 8);
    v6 = *(const __m128i **)a1;
    v9 = v8;
  }
  v30 = v9;
  v31 = v9;
  v32 = v9 + v7;
  if ( v5 != v6 )
  {
    v10 = (__m128i *)v9;
    v11 = v6;
    do
    {
      if ( v10 )
      {
        *v10 = _mm_loadu_si128(v11);
        v10[1].m128i_i64[0] = v11[1].m128i_i64[0];
      }
      v11 = (const __m128i *)((char *)v11 + 24);
      v10 = (__m128i *)((char *)v10 + 24);
    }
    while ( v11 != v5 );
    v9 += 8 * ((unsigned __int64)((char *)&v11[-2].m128i_u64[1] - (char *)v6) >> 3) + 24;
  }
  v12 = *(_QWORD *)(a1 + 48);
  v13 = *(_QWORD *)(a1 + 24);
  v14 = (const __m128i **)&v30;
  v15 = *(_QWORD *)(a1 + 40);
  v16 = *(_DWORD *)(a1 + 112);
  v34 = 0;
  v35 = v13;
  v36 = v15;
  v37 = v12;
  v17 = *(_QWORD *)(v12 + 16);
  v39 = &v30;
  v38 = v17;
  v18 = *(_QWORD *)(v17 + 64) - *(_QWORD *)(v17 + 56);
  v31 = v9;
  memset(v33, 0, sizeof(v33));
  v19 = 0xAAAAAAAAAAAAAAABLL * (v18 >> 4);
  if ( v18 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v40 = 0;
  v20 = 0;
  v41 = 0;
  v21 = 16 * v19;
  v42 = 0;
  if ( v19 )
  {
    v22 = sub_22077B0(16 * v19);
    v20 = v22 + v21;
    v40 = v22;
    v42 = v22 + v21;
    do
    {
      if ( v22 )
      {
        *(_QWORD *)v22 = 0;
        *(_DWORD *)(v22 + 8) = 0;
      }
      v22 += 16;
    }
    while ( v20 != v22 );
    v14 = (const __m128i **)v39;
  }
  v41 = v20;
  v43 = a2;
  if ( (v16 & 0x80u) != 0 )
    v16 &= 0xFFFFFFFA;
  v44 = 0;
  v46 = 0;
  v34 = v35;
  v45 = v16;
  sub_2BDBCA0(v33, v14, v20);
  sub_2BE13A0((__int64)v33, 1u, v43);
  v23 = v46;
  if ( v46 )
  {
    v25 = v30;
    v26 = 0;
    v27 = 0;
    if ( v31 != v30 )
    {
      do
      {
        v28 = v25 + v26;
        if ( *(_BYTE *)(v25 + v26 + 16) )
        {
          v29 = v26 + *(_QWORD *)a1;
          *(_QWORD *)v29 = *(_QWORD *)v28;
          *(_QWORD *)(v29 + 8) = *(_QWORD *)(v28 + 8);
          *(_BYTE *)(v29 + 16) = *(_BYTE *)(v28 + 16);
          v25 = v30;
        }
        ++v27;
        v26 += 24;
      }
      while ( v27 < 0xAAAAAAAAAAAAAAABLL * ((__int64)(v31 - v25) >> 3) );
    }
  }
  if ( v40 )
    j_j___libc_free_0(v40);
  if ( v33[0] )
    j_j___libc_free_0(v33[0]);
  if ( v30 )
    j_j___libc_free_0(v30);
  return v23;
}
