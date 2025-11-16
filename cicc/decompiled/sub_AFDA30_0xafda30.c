// Function: sub_AFDA30
// Address: 0xafda30
//
__int64 __fastcall sub_AFDA30(__int64 a1, const __m128i **a2, const __m128i ***a3)
{
  int v4; // r13d
  const __m128i *v6; // rax
  __int64 v7; // r14
  unsigned __int8 v9; // dl
  __int64 m128i_i64; // rcx
  unsigned __int8 v11; // dl
  __int64 v12; // rcx
  int v13; // eax
  int v14; // r13d
  int v15; // eax
  const __m128i *v16; // rsi
  int v17; // r8d
  const __m128i **v18; // rdi
  unsigned int v19; // eax
  const __m128i **v20; // rcx
  const __m128i *v21; // rdx
  int v22; // [rsp+4h] [rbp-5Ch] BYREF
  __int64 v23; // [rsp+8h] [rbp-58h] BYREF
  __int64 v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 v25; // [rsp+18h] [rbp-48h] BYREF
  __m128i v26; // [rsp+20h] [rbp-40h]
  __int64 v27; // [rsp+30h] [rbp-30h]
  __int64 v28[5]; // [rsp+38h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 8);
  v9 = (*a2)[-1].m128i_u8[0];
  m128i_i64 = (__int64)(*a2)[-1].m128i_i64;
  if ( (v9 & 2) != 0 )
  {
    v24 = *(_QWORD *)v6[-2].m128i_i64[0];
    v11 = v6[-1].m128i_u8[0];
    if ( (v11 & 2) != 0 )
    {
LABEL_5:
      v12 = v6[-2].m128i_i64[0];
      goto LABEL_6;
    }
  }
  else
  {
    v24 = *(_QWORD *)(m128i_i64 - 8LL * ((v9 >> 2) & 0xF));
    v11 = v6[-1].m128i_u8[0];
    if ( (v11 & 2) != 0 )
      goto LABEL_5;
  }
  v12 = m128i_i64 - 8LL * ((v11 >> 2) & 0xF);
LABEL_6:
  v25 = *(_QWORD *)(v12 + 8);
  v26 = _mm_loadu_si128(v6 + 1);
  v27 = v6[2].m128i_i64[0];
  v28[0] = v6[2].m128i_i64[1];
  if ( (_BYTE)v27 )
  {
    v23 = v26.m128i_i64[1];
    v13 = v26.m128i_i32[0];
  }
  else
  {
    v23 = 0;
    v13 = 0;
  }
  v22 = v13;
  v14 = v4 - 1;
  v15 = sub_AFAA60(&v24, &v25, &v22, &v23, v28);
  v16 = *a2;
  v17 = 1;
  v18 = 0;
  v19 = v14 & v15;
  v20 = (const __m128i **)(v7 + 8LL * v19);
  v21 = *v20;
  if ( *v20 == *a2 )
  {
LABEL_18:
    *a3 = v20;
    return 1;
  }
  else
  {
    while ( v21 != (const __m128i *)-4096LL )
    {
      if ( v21 != (const __m128i *)-8192LL || v18 )
        v20 = v18;
      v19 = v14 & (v17 + v19);
      v21 = *(const __m128i **)(v7 + 8LL * v19);
      if ( v21 == v16 )
      {
        v20 = (const __m128i **)(v7 + 8LL * v19);
        goto LABEL_18;
      }
      ++v17;
      v18 = v20;
      v20 = (const __m128i **)(v7 + 8LL * v19);
    }
    if ( !v18 )
      v18 = v20;
    *a3 = v18;
    return 0;
  }
}
