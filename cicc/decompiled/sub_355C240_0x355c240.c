// Function: sub_355C240
// Address: 0x355c240
//
__int64 *__fastcall sub_355C240(__int64 *a1, __m128i *a2, __int64 *a3, __int64 *a4)
{
  __int64 v6; // rdx
  __int64 v7; // rdi
  __m128i *v8; // rbx
  __int64 v9; // r15
  __int64 v10; // r11
  __int64 v11; // r8
  __int64 v12; // rsi
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // rsi
  unsigned __int64 v16; // r14
  __int64 v17; // rax
  unsigned __int64 v18; // r15
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __m128i v21; // xmm1
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // r15
  unsigned __int64 *v25; // r12
  unsigned __int64 v26; // rdi
  __m128i v27; // xmm3
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned __int64 *v32; // rbx
  __int64 v33; // r14
  unsigned __int64 v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // [rsp+8h] [rbp-F8h]
  __int64 v38; // [rsp+10h] [rbp-F0h]
  __int64 v39; // [rsp+20h] [rbp-E0h]
  __int64 v40; // [rsp+28h] [rbp-D8h]
  __int64 v41; // [rsp+30h] [rbp-D0h]
  __int64 v42; // [rsp+38h] [rbp-C8h]
  __int64 v43; // [rsp+40h] [rbp-C0h]
  __m128i *v44; // [rsp+40h] [rbp-C0h]
  __int64 v45; // [rsp+48h] [rbp-B8h]
  __int64 v46[4]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+70h] [rbp-90h] BYREF
  __int64 v48; // [rsp+78h] [rbp-88h]
  __int64 v49; // [rsp+80h] [rbp-80h]
  __int64 v50; // [rsp+88h] [rbp-78h]
  __int64 v51; // [rsp+90h] [rbp-70h] BYREF
  __int64 v52; // [rsp+98h] [rbp-68h]
  __int64 v53; // [rsp+A0h] [rbp-60h]
  __int64 v54; // [rsp+A8h] [rbp-58h]
  __m128i v55; // [rsp+B0h] [rbp-50h] BYREF
  __m128i v56; // [rsp+C0h] [rbp-40h] BYREF

  v6 = *a3;
  v7 = *a4;
  if ( v6 == *a4 )
  {
    *a1 = v6;
    a1[1] = a3[1];
    v35 = a3[2];
    v36 = a3[3];
    a1[2] = v35;
    a1[3] = v36;
  }
  else
  {
    v8 = a2;
    v9 = a2[1].m128i_i64[0];
    v10 = a2[2].m128i_i64[1];
    v11 = a2[3].m128i_i64[0];
    v40 = a2[1].m128i_i64[1];
    v45 = a2[2].m128i_i64[0];
    v12 = a2[4].m128i_i64[1];
    v43 = v12;
    if ( v6 == v9 && v7 == v11 )
    {
      if ( v12 + 8 > (unsigned __int64)(v10 + 8) )
      {
        v44 = v8;
        v32 = (unsigned __int64 *)(v10 + 8);
        v33 = v10;
        do
        {
          v34 = *v32++;
          j_j___libc_free_0(v34);
        }
        while ( v12 + 8 > (unsigned __int64)v32 );
        v8 = v44;
        v10 = v33;
      }
      v8[3].m128i_i64[0] = v9;
      v8[4].m128i_i64[1] = v10;
      v8[3].m128i_i64[1] = v40;
      v8[4].m128i_i64[0] = v45;
      *a1 = v9;
      a1[1] = v40;
      a1[2] = v45;
      a1[3] = v10;
    }
    else
    {
      v13 = a3[3];
      v38 = a4[3];
      v41 = a4[1];
      v37 = a3[2];
      v14 = ((((v38 - v13) >> 3) - 1) << 6) + ((v7 - v41) >> 3) + ((v37 - v6) >> 3);
      v42 = a3[1];
      v39 = v8[3].m128i_i64[1];
      v15 = (v45 - v9) >> 3;
      v16 = v15 + ((((v13 - v10) >> 3) - 1) << 6) + ((v6 - v42) >> 3);
      if ( (unsigned __int64)(v15 + ((v11 - v39) >> 3) + ((((v43 - v10) >> 3) - 1) << 6) - v14) >> 1 < v16 )
      {
        v22 = v8[4].m128i_i64[0];
        if ( v7 != v11 )
        {
          v55.m128i_i64[0] = v6;
          v53 = v22;
          v55.m128i_i64[1] = v42;
          v54 = v43;
          v23 = a4[2];
          v56.m128i_i64[0] = v37;
          v48 = v41;
          v47 = v7;
          v52 = v39;
          v49 = v23;
          v51 = v11;
          v50 = v38;
          v56.m128i_i64[1] = v13;
          sub_355AE80(v46, (__int64)&v47, &v51, (__int64)&v55);
          v11 = v8[3].m128i_i64[0];
          v43 = v8[4].m128i_i64[1];
          v39 = v8[3].m128i_i64[1];
          v22 = v8[4].m128i_i64[0];
        }
        v56.m128i_i64[0] = v22;
        v55.m128i_i64[0] = v11;
        v55.m128i_i64[1] = v39;
        v56.m128i_i64[1] = v43;
        sub_353DF70(v55.m128i_i64, -v14);
        v24 = v8[4].m128i_i64[1] + 8;
        v25 = (unsigned __int64 *)(v56.m128i_i64[1] + 8);
        if ( v24 > v56.m128i_i64[1] + 8 )
        {
          do
          {
            v26 = *v25++;
            j_j___libc_free_0(v26);
          }
          while ( v24 > (unsigned __int64)v25 );
        }
        v27 = _mm_loadu_si128(&v56);
        v8[3] = _mm_loadu_si128(&v55);
        v8[4] = v27;
      }
      else
      {
        if ( v6 != v9 )
        {
          v17 = a4[2];
          v55.m128i_i64[0] = *a4;
          v51 = v6;
          v56.m128i_i64[0] = v17;
          v55.m128i_i64[1] = v41;
          v52 = v42;
          v56.m128i_i64[1] = v38;
          v53 = v37;
          v48 = v40;
          v47 = v9;
          v49 = v45;
          v50 = v10;
          v54 = v13;
          sub_355ABF0(v46, &v47, (__int64)&v51, &v55);
          v9 = v8[1].m128i_i64[0];
          v10 = v8[2].m128i_i64[1];
          v40 = v8[1].m128i_i64[1];
          v45 = v8[2].m128i_i64[0];
        }
        v55.m128i_i64[0] = v9;
        v56.m128i_i64[1] = v10;
        v55.m128i_i64[1] = v40;
        v56.m128i_i64[0] = v45;
        sub_353DF70(v55.m128i_i64, v14);
        v18 = v56.m128i_u64[1];
        v19 = (unsigned __int64 *)v8[2].m128i_i64[1];
        if ( v56.m128i_i64[1] > (unsigned __int64)v19 )
        {
          do
          {
            v20 = *v19++;
            j_j___libc_free_0(v20);
          }
          while ( v18 > (unsigned __int64)v19 );
        }
        v21 = _mm_loadu_si128(&v56);
        v8[1] = _mm_loadu_si128(&v55);
        v8[2] = v21;
      }
      v28 = v8[1].m128i_i64[1];
      v29 = v8[2].m128i_i64[0];
      v30 = v8[2].m128i_i64[1];
      *a1 = v8[1].m128i_i64[0];
      a1[1] = v28;
      a1[2] = v29;
      a1[3] = v30;
      sub_353DF70(a1, v16);
    }
  }
  return a1;
}
