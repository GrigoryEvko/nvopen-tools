// Function: sub_2568D00
// Address: 0x2568d00
//
_OWORD *__fastcall sub_2568D00(__int64 a1, int a2)
{
  __int64 v3; // r15
  __int64 v4; // rbx
  unsigned int v5; // eax
  _OWORD *result; // rax
  __int64 v7; // rdx
  _OWORD *v8; // rdx
  __m128i i; // xmm0
  __m128i *v10; // rbx
  __int64 v11; // rdx
  int v12; // ecx
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r8
  int v16; // r13d
  __int64 v17; // r9
  __m128i *v18; // r11
  unsigned int j; // eax
  __m128i *v20; // r12
  __int64 v21; // rdi
  unsigned int v22; // eax
  __m128i *v23; // rax
  __m128i v24; // xmm1
  __m128i *v25; // r13
  unsigned __int32 v26; // r15d
  __int64 v27; // r12
  unsigned __int64 v28; // r13
  void (__fastcall *v29)(unsigned __int64, unsigned __int64, __int64, __int64, __int64, __int64, double); // rax
  __m128i *v30; // rax
  __m128i *v31; // rax
  __int64 v32; // rcx
  __m128i *v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // r12
  __int64 v37; // r13
  void (__fastcall *v38)(__int64, __int64, __int64, __int64, __int64, __int64, double); // rax
  unsigned __int64 v39; // rdi
  __int32 v40; // eax
  __m128i *v41; // rdx
  __m128i v42; // xmm0
  _OWORD *k; // rdx
  __int64 v44; // [rsp+8h] [rbp-88h]
  __int64 v45; // [rsp+10h] [rbp-80h]
  __int64 v46; // [rsp+18h] [rbp-78h]
  __int32 v47; // [rsp+20h] [rbp-70h]
  __m128i *v48; // [rsp+28h] [rbp-68h]
  __int64 v49; // [rsp+30h] [rbp-60h]
  __int64 v50; // [rsp+38h] [rbp-58h]
  __int64 v51; // [rsp+40h] [rbp-50h]
  __m128i *v52; // [rsp+48h] [rbp-48h]
  unsigned __int64 v53[7]; // [rsp+58h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 24);
  v49 = v3;
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_OWORD *)sub_C7D670((unsigned __int64)v5 << 6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v44 = v4 << 6;
    v52 = (__m128i *)((v4 << 6) + v3);
    v8 = &result[4 * v7];
    for ( i = _mm_loadu_si128((const __m128i *)&unk_4FEE4D0); v8 != result; result += 4 )
    {
      if ( result )
        *result = i;
    }
    v51 = unk_4FEE4D0;
    v46 = unk_4FEE4D8;
    v50 = qword_4FEE4C0[0];
    v45 = qword_4FEE4C0[1];
    v10 = (__m128i *)(v3 + 32);
    if ( v52 != (__m128i *)v3 )
    {
      while ( 1 )
      {
        v11 = v10[-2].m128i_i64[0];
        if ( v51 == v11 && v46 == v10[-2].m128i_i64[1] )
          goto LABEL_18;
        if ( v50 != v11 || v45 != v10[-2].m128i_i64[1] )
          break;
        v23 = v10 + 4;
        if ( v52 == &v10[2] )
          return (_OWORD *)sub_C7D6A0(v49, v44, 8);
LABEL_19:
        v10 = v23;
      }
      v12 = *(_DWORD *)(a1 + 24);
      if ( !v12 )
      {
        MEMORY[0] = _mm_loadu_si128(v10 - 2);
        BUG();
      }
      v13 = v10[-2].m128i_i64[1];
      v14 = (unsigned int)(v12 - 1);
      v15 = *(_QWORD *)(a1 + 8);
      v16 = 1;
      v17 = unk_4FEE4D0;
      v18 = 0;
      for ( j = v14
              & (((unsigned int)v13 >> 9)
               ^ ((unsigned int)v13 >> 4)
               ^ (16 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)))); ; j = v14 & v22 )
      {
        v20 = (__m128i *)(v15 + ((unsigned __int64)j << 6));
        v21 = v20->m128i_i64[0];
        if ( v11 == v20->m128i_i64[0] && v13 == v20->m128i_i64[1] )
          break;
        if ( unk_4FEE4D0 == v21 && unk_4FEE4D8 == v20->m128i_i64[1] )
        {
          if ( v18 )
            v20 = v18;
          break;
        }
        if ( qword_4FEE4C0[0] == v21 && v20->m128i_i64[1] == qword_4FEE4C0[1] && !v18 )
          v18 = (__m128i *)(v15 + ((unsigned __int64)j << 6));
        v22 = v16 + j;
        ++v16;
      }
      v24 = _mm_loadu_si128(v10 - 2);
      v25 = v20 + 2;
      v20[1].m128i_i64[1] = 0x100000000LL;
      v20[1].m128i_i64[0] = (__int64)v20[2].m128i_i64;
      *v20 = v24;
      v26 = v10[-1].m128i_u32[2];
      if ( &v20[1] != &v10[-1] && v26 )
      {
        v30 = (__m128i *)v10[-1].m128i_i64[0];
        if ( v30 == v10 )
        {
          v31 = v10;
          v32 = 1;
          if ( v26 != 1 )
          {
            v48 = (__m128i *)sub_C8D7D0((__int64)v20[1].m128i_i64, (__int64)v20[2].m128i_i64, v26, 0x20u, v53, v17);
            sub_255FA70((__int64)v20[1].m128i_i64, v48);
            v39 = v20[1].m128i_u64[0];
            v40 = v53[0];
            v41 = v48;
            if ( v25 != (__m128i *)v39 )
            {
              v47 = v53[0];
              _libc_free(v39);
              v40 = v47;
              v41 = v48;
            }
            v20[1].m128i_i64[0] = (__int64)v41;
            v25 = v41;
            v20[1].m128i_i32[3] = v40;
            v31 = (__m128i *)v10[-1].m128i_i64[0];
            v32 = v10[-1].m128i_u32[2];
          }
          v14 = 32 * v32;
          v33 = (__m128i *)((char *)v25 + v14);
          if ( v14 )
          {
            do
            {
              if ( v25 )
              {
                v25[1].m128i_i64[0] = 0;
                i = _mm_loadu_si128(v31);
                *v31 = _mm_loadu_si128(v25);
                *v25 = i;
                v34 = v31[1].m128i_i64[0];
                v31[1].m128i_i64[0] = 0;
                v14 = v25[1].m128i_i64[1];
                v25[1].m128i_i64[0] = v34;
                v35 = v31[1].m128i_i64[1];
                v31[1].m128i_i64[1] = v14;
                v25[1].m128i_i64[1] = v35;
              }
              v25 += 2;
              v31 += 2;
            }
            while ( v33 != v25 );
          }
          v20[1].m128i_i32[2] = v26;
          v36 = v10[-1].m128i_i64[0];
          v37 = v36 + 32LL * v10[-1].m128i_u32[2];
          while ( v36 != v37 )
          {
            v38 = *(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, double))(v37 - 16);
            v37 -= 32;
            if ( v38 )
              v38(v37, v37, 3, v14, v15, v17, *(double *)i.m128i_i64);
          }
          v10[-1].m128i_i32[2] = 0;
        }
        else
        {
          v20[1].m128i_i64[0] = (__int64)v30;
          v20[1].m128i_i32[2] = v10[-1].m128i_i32[2];
          v20[1].m128i_i32[3] = v10[-1].m128i_i32[3];
          v10[-1].m128i_i64[0] = (__int64)v10;
          v10[-1].m128i_i32[3] = 0;
          v10[-1].m128i_i32[2] = 0;
        }
      }
      ++*(_DWORD *)(a1 + 16);
      v27 = v10[-1].m128i_i64[0];
      v28 = v27 + 32LL * v10[-1].m128i_u32[2];
      if ( v27 != v28 )
      {
        do
        {
          v29 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64, __int64, __int64, __int64, double))(v28 - 16);
          v28 -= 32LL;
          if ( v29 )
            v29(v28, v28, 3, v14, v15, v17, *(double *)i.m128i_i64);
        }
        while ( v27 != v28 );
        v28 = v10[-1].m128i_u64[0];
      }
      if ( v10 != (__m128i *)v28 )
        _libc_free(v28);
LABEL_18:
      v23 = v10 + 4;
      if ( v52 == &v10[2] )
        return (_OWORD *)sub_C7D6A0(v49, v44, 8);
      goto LABEL_19;
    }
    return (_OWORD *)sub_C7D6A0(v49, v44, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v42 = _mm_loadu_si128((const __m128i *)&unk_4FEE4D0);
    for ( k = &result[4 * (unsigned __int64)*(unsigned int *)(a1 + 24)]; k != result; result += 4 )
    {
      if ( result )
        *result = v42;
    }
  }
  return result;
}
