// Function: sub_38E8150
// Address: 0x38e8150
//
void __fastcall sub_38E8150(__int64 a1, unsigned __int8 *a2, size_t a3, const __m128i *a4)
{
  __int64 v4; // r15
  __int64 v7; // rax
  unsigned __int64 v8; // r12
  __int64 v9; // rbx
  __m128i v10; // xmm0
  unsigned int v11; // r9d
  __int64 *v12; // r10
  unsigned __int64 v13; // r14
  __int64 v14; // r15
  unsigned __int64 v15; // r13
  unsigned __int64 v16; // rdi
  __int64 v17; // rax
  unsigned int v18; // r9d
  __int64 *v19; // r10
  __int64 v20; // rcx
  _BYTE *v21; // rdi
  __m128i v22; // xmm2
  __m128i v23; // xmm3
  __int64 v24; // rax
  _BYTE *v25; // rax
  __int64 *v26; // [rsp+0h] [rbp-B0h]
  __int64 v27; // [rsp+0h] [rbp-B0h]
  unsigned int v28; // [rsp+8h] [rbp-A8h]
  __int64 v29; // [rsp+8h] [rbp-A8h]
  __int64 *v30; // [rsp+8h] [rbp-A8h]
  __int64 *v31; // [rsp+10h] [rbp-A0h]
  unsigned int v32; // [rsp+10h] [rbp-A0h]
  unsigned int v33; // [rsp+20h] [rbp-90h]
  __int64 v34; // [rsp+28h] [rbp-88h]
  __m128i v35; // [rsp+40h] [rbp-70h] BYREF
  __m128i v36[6]; // [rsp+50h] [rbp-60h] BYREF

  v4 = a1 + 1488;
  v7 = a4[3].m128i_i64[0];
  v8 = a4[2].m128i_u64[0];
  a4[3].m128i_i64[0] = 0;
  v9 = a4[2].m128i_i64[1];
  v10 = _mm_loadu_si128(a4);
  a4[2].m128i_i64[1] = 0;
  a4[2].m128i_i64[0] = 0;
  v34 = v7;
  v35 = v10;
  v36[0] = _mm_loadu_si128(a4 + 1);
  v11 = sub_16D19C0(a1 + 1488, a2, a3);
  v12 = (__int64 *)(*(_QWORD *)(a1 + 1488) + 8LL * v11);
  if ( !*v12 )
  {
LABEL_17:
    v26 = v12;
    v28 = v11;
    v17 = malloc(a3 + 65);
    v18 = v28;
    v19 = v26;
    v20 = v17;
    if ( !v17 )
    {
      if ( a3 == -65 )
      {
        v24 = malloc(1u);
        v18 = v28;
        v19 = v26;
        v20 = 0;
        if ( v24 )
        {
          v21 = (_BYTE *)(v24 + 64);
          v20 = v24;
          goto LABEL_22;
        }
      }
      v27 = v20;
      v30 = v19;
      v32 = v18;
      sub_16BD1C0("Allocation failed", 1u);
      v18 = v32;
      v19 = v30;
      v20 = v27;
    }
    v21 = (_BYTE *)(v20 + 64);
    if ( a3 + 1 <= 1 )
    {
LABEL_19:
      v22 = _mm_loadu_si128(&v35);
      v23 = _mm_loadu_si128(v36);
      v21[a3] = 0;
      *(_QWORD *)v20 = a3;
      *(_QWORD *)(v20 + 40) = v8;
      *(_QWORD *)(v20 + 48) = v9;
      *(_QWORD *)(v20 + 56) = v34;
      *(__m128i *)(v20 + 8) = v22;
      *(__m128i *)(v20 + 24) = v23;
      *v19 = v20;
      ++*(_DWORD *)(a1 + 1500);
      sub_16D1CD0(v4, v18);
      return;
    }
LABEL_22:
    v29 = v20;
    v31 = v19;
    v33 = v18;
    v25 = memcpy(v21, a2, a3);
    v20 = v29;
    v19 = v31;
    v18 = v33;
    v21 = v25;
    goto LABEL_19;
  }
  if ( *v12 == -8 )
  {
    --*(_DWORD *)(a1 + 1504);
    goto LABEL_17;
  }
  if ( v8 != v9 )
  {
    v13 = v8;
    do
    {
      v14 = *(_QWORD *)(v13 + 24);
      v15 = *(_QWORD *)(v13 + 16);
      if ( v14 != v15 )
      {
        do
        {
          if ( *(_DWORD *)(v15 + 32) > 0x40u )
          {
            v16 = *(_QWORD *)(v15 + 24);
            if ( v16 )
              j_j___libc_free_0_0(v16);
          }
          v15 += 40LL;
        }
        while ( v14 != v15 );
        v15 = *(_QWORD *)(v13 + 16);
      }
      if ( v15 )
        j_j___libc_free_0(v15);
      v13 += 48LL;
    }
    while ( v9 != v13 );
  }
  if ( v8 )
    j_j___libc_free_0(v8);
}
