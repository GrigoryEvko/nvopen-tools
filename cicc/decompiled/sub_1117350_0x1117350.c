// Function: sub_1117350
// Address: 0x1117350
//
__int64 __fastcall sub_1117350(const __m128i *a1, int a2, char a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __m128i v6; // xmm1
  unsigned __int64 v7; // xmm2_8
  __m128i v8; // xmm3
  __int64 v9; // rax
  __int64 result; // rax
  __m128i v11; // xmm1
  unsigned __int64 v12; // xmm2_8
  __m128i v13; // xmm3
  __int64 v14; // rax
  __m128i v15; // xmm1
  unsigned __int64 v16; // xmm2_8
  __m128i v17; // xmm3
  __int64 v18; // rax
  __m128i v19; // xmm5
  unsigned __int64 v20; // xmm6_8
  __m128i v21; // xmm7
  __int64 v22; // rax
  __m128i v23; // xmm5
  unsigned __int64 v24; // xmm6_8
  __m128i v25; // xmm7
  __int64 v26; // rax
  __m128i v27; // xmm5
  unsigned __int64 v28; // xmm6_8
  __m128i v29; // xmm7
  __int64 v30; // rax
  unsigned int v31; // [rsp+Ch] [rbp-B4h]
  unsigned int v32; // [rsp+Ch] [rbp-B4h]
  unsigned int v33; // [rsp+Ch] [rbp-B4h]
  unsigned int v34; // [rsp+Ch] [rbp-B4h]
  unsigned __int64 v35; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v36; // [rsp+18h] [rbp-A8h]
  unsigned int v37; // [rsp+20h] [rbp-A0h]
  __int64 v38; // [rsp+28h] [rbp-98h]
  unsigned int v39; // [rsp+30h] [rbp-90h]
  unsigned __int64 v40; // [rsp+40h] [rbp-80h] BYREF
  __int64 v41; // [rsp+48h] [rbp-78h]
  unsigned int v42; // [rsp+50h] [rbp-70h]
  __int64 v43; // [rsp+58h] [rbp-68h]
  unsigned int v44; // [rsp+60h] [rbp-60h]
  __m128i v45; // [rsp+70h] [rbp-50h] BYREF
  __m128i v46; // [rsp+80h] [rbp-40h]
  unsigned __int64 v47; // [rsp+90h] [rbp-30h]
  __int64 v48; // [rsp+98h] [rbp-28h]
  __m128i v49; // [rsp+A0h] [rbp-20h]
  __int64 v50; // [rsp+B0h] [rbp-10h]

  if ( a2 == 15 )
  {
    if ( a3 )
    {
      v15 = _mm_loadu_si128(a1 + 7);
      v16 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v17 = _mm_loadu_si128(a1 + 9);
      v18 = a1[10].m128i_i64[0];
      v45 = _mm_loadu_si128(a1 + 6);
      v47 = v16;
      v50 = v18;
      v48 = a6;
      v46 = v15;
      v49 = v17;
      return sub_9AFB10(a4, a5, &v45);
    }
    else
    {
      v19 = _mm_loadu_si128(a1 + 7);
      v20 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v21 = _mm_loadu_si128(a1 + 9);
      v22 = a1[10].m128i_i64[0];
      v45 = _mm_loadu_si128(a1 + 6);
      v47 = v20;
      v50 = v22;
      v48 = a6;
      v46 = v19;
      v49 = v21;
      return sub_9AC9C0(a4, a5, &v45);
    }
  }
  if ( a2 == 17 )
  {
    if ( a3 )
    {
      v6 = _mm_loadu_si128(a1 + 7);
      v7 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v8 = _mm_loadu_si128(a1 + 9);
      v9 = a1[10].m128i_i64[0];
      v45 = _mm_loadu_si128(a1 + 6);
      v47 = v7;
      v50 = v9;
      v48 = a6;
      v46 = v6;
      v49 = v8;
      return sub_9AF960(a4, a5, &v45);
    }
    else
    {
      v23 = _mm_loadu_si128(a1 + 7);
      v24 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v25 = _mm_loadu_si128(a1 + 9);
      v26 = a1[10].m128i_i64[0];
      v45 = _mm_loadu_si128(a1 + 6);
      v47 = v24;
      v50 = v26;
      v48 = a6;
      v46 = v23;
      v49 = v25;
      return sub_9AC590(a4, a5, &v45, 0);
    }
  }
  if ( a2 != 13 )
    BUG();
  v42 = 1;
  v40 = a5 & 0xFFFFFFFFFFFFFFFBLL;
  v41 = 0;
  v44 = 1;
  v43 = 0;
  v35 = a4 & 0xFFFFFFFFFFFFFFFBLL;
  v37 = 1;
  v36 = 0;
  v39 = 1;
  v38 = 0;
  if ( a3 )
  {
    v11 = _mm_loadu_si128(a1 + 7);
    v12 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
    v13 = _mm_loadu_si128(a1 + 9);
    v14 = a1[10].m128i_i64[0];
    v45 = _mm_loadu_si128(a1 + 6);
    v47 = v12;
    v50 = v14;
    v48 = a6;
    v46 = v11;
    v49 = v13;
    result = sub_9B0100((__int64 *)&v35, (__int64 *)&v40, &v45);
    if ( v39 > 0x40 )
      goto LABEL_9;
  }
  else
  {
    v27 = _mm_loadu_si128(a1 + 7);
    v28 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
    v29 = _mm_loadu_si128(a1 + 9);
    v30 = a1[10].m128i_i64[0];
    v45 = _mm_loadu_si128(a1 + 6);
    v47 = v28;
    v50 = v30;
    v48 = a6;
    v46 = v27;
    v49 = v29;
    result = sub_9AC900((__int64 *)&v35, (__int64 *)&v40, &v45);
    if ( v39 > 0x40 )
    {
LABEL_9:
      if ( v38 )
      {
        v31 = result;
        j_j___libc_free_0_0(v38);
        result = v31;
      }
    }
  }
  if ( v37 > 0x40 && v36 )
  {
    v32 = result;
    j_j___libc_free_0_0(v36);
    result = v32;
  }
  if ( v44 > 0x40 && v43 )
  {
    v33 = result;
    j_j___libc_free_0_0(v43);
    result = v33;
  }
  if ( v42 > 0x40 && v41 )
  {
    v34 = result;
    j_j___libc_free_0_0(v41);
    return v34;
  }
  return result;
}
