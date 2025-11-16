// Function: sub_2445FE0
// Address: 0x2445fe0
//
void __fastcall sub_2445FE0(__int64 a1, __int64 a2, __m128i *a3)
{
  unsigned int v5; // eax
  __int8 v6; // cl
  int v7; // ecx
  __int64 v8; // rdx
  __m128i *v9; // rax
  __m128i *v10; // rbx
  __m128i *v11; // r12
  __int64 v12; // rdx
  __m128i *v13; // rax
  __m128i *v14; // r13
  const __m128i *v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r15
  __int64 *v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rcx
  unsigned __int64 v22; // rax
  __m128i *v23; // rsi
  __int64 v24; // rdi
  unsigned __int64 v25; // rcx
  __int64 *v26; // rdx
  const __m128i *v27; // rax
  __m128i v28; // xmm0
  __m128i *v29; // rbx
  __int64 v30; // rdx
  __m128i *v31; // [rsp+0h] [rbp-80h]
  __int64 v32; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+10h] [rbp-70h]
  const __m128i *v34; // [rsp+10h] [rbp-70h]
  __m128i v36; // [rsp+20h] [rbp-60h] BYREF
  void *src; // [rsp+30h] [rbp-50h] BYREF
  __m128i *v38; // [rsp+38h] [rbp-48h]
  __m128i *v39; // [rsp+40h] [rbp-40h]

  if ( LOBYTE(qword_4F8A488[8]) != 1 || !a2 || (*(_BYTE *)(a2 + 7) & 0x20) == 0 || !sub_B91C10(a2, 2) )
    return;
  sub_B99FD0(a2, 2u, 0);
  v5 = a3->m128i_u32[2];
  src = 0;
  v38 = 0;
  v6 = a3->m128i_i8[8];
  v39 = 0;
  if ( v5 >> 1 )
  {
    v7 = v6 & 1;
    if ( v7 )
    {
      v10 = a3 + 1;
      v11 = a3 + 17;
    }
    else
    {
      v8 = a3[1].m128i_u32[2];
      v9 = (__m128i *)a3[1].m128i_i64[0];
      v10 = v9;
      v11 = &v9[v8];
      if ( v9 == v11 )
        goto LABEL_12;
    }
    do
    {
      if ( v10->m128i_i64[0] <= 0xFFFFFFFFFFFFFFFDLL )
        break;
      ++v10;
    }
    while ( v10 != v11 );
  }
  else
  {
    v7 = v6 & 1;
    if ( v7 )
    {
      v29 = a3 + 1;
      v30 = 16;
    }
    else
    {
      v29 = (__m128i *)a3[1].m128i_i64[0];
      v30 = a3[1].m128i_u32[2];
    }
    v10 = &v29[v30];
    v11 = v10;
  }
  if ( (_BYTE)v7 )
  {
    v9 = a3 + 1;
    v12 = 16;
    goto LABEL_13;
  }
  v9 = (__m128i *)a3[1].m128i_i64[0];
  v8 = a3[1].m128i_u32[2];
LABEL_12:
  v12 = v8;
LABEL_13:
  v13 = &v9[v12];
  v14 = 0;
  v15 = &v36;
  v16 = 0;
  if ( v13 == v10 )
  {
    v21 = 0;
    LODWORD(v20) = 0;
  }
  else
  {
    do
    {
      v17 = v10->m128i_i64[1];
      if ( v17 )
      {
        v36 = *v10;
        if ( v39 == v14 )
        {
          v31 = v13;
          v32 = v16;
          v34 = v15;
          sub_EDCF30((__int64)&src, v14, v15);
          v14 = v38;
          v13 = v31;
          v16 = v32;
          v15 = v34;
        }
        else
        {
          if ( v14 )
          {
            *v14 = _mm_loadu_si128(&v36);
            v14 = v38;
          }
          v38 = ++v14;
        }
        v16 += v17;
      }
      for ( ++v10; v11 != v10; ++v10 )
      {
        if ( v10->m128i_i64[0] <= 0xFFFFFFFFFFFFFFFDLL )
          break;
      }
    }
    while ( v13 != v10 );
    v18 = (__int64 *)src;
    v19 = (char *)v14 - (_BYTE *)src;
    v20 = ((char *)v14 - (_BYTE *)src) >> 4;
    v21 = v20;
    if ( src != v14 )
    {
      _BitScanReverse64(&v22, ((char *)v14 - (_BYTE *)src) >> 4);
      v33 = v16;
      sub_2443BE0((__m128i *)src, v14, 2LL * (int)(63 - (v22 ^ 0x3F)));
      if ( v19 <= 256 )
      {
        sub_2444160(v18, v14->m128i_i64);
        v16 = v33;
      }
      else
      {
        sub_2444160(v18, v18 + 32);
        v16 = v33;
        v23 = (__m128i *)(v18 + 32);
        if ( v18 + 32 != (__int64 *)v14 )
        {
          do
          {
            v24 = v23->m128i_i64[0];
            v25 = v23->m128i_u64[1];
            v26 = (__int64 *)v23;
            if ( v23[-1].m128i_i64[1] < v25 )
            {
              v27 = v23 - 1;
              do
              {
                v28 = _mm_loadu_si128(v27);
                v26 = (__int64 *)v27--;
                v27[2] = v28;
              }
              while ( v25 > v27->m128i_i64[1] );
            }
            ++v23;
            *v26 = v24;
            v26[1] = v25;
          }
          while ( v14 != v23 );
        }
      }
      v14 = (__m128i *)src;
      v20 = ((char *)v38 - (_BYTE *)src) >> 4;
      v21 = v20;
    }
  }
  sub_ED2230(*(__int64 ***)(a1 + 8), a2, v14->m128i_i64, v21, v16, 2u, v20);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
}
