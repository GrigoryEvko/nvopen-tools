// Function: sub_2BECC80
// Address: 0x2becc80
//
__int64 __fastcall sub_2BECC80(__int64 a1)
{
  __int64 result; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r14
  unsigned __int64 v6; // rdi
  __int64 v7; // rbx
  unsigned __int64 v8; // rdi
  __int64 v9; // r10
  __int64 v10; // r9
  unsigned __int64 v11; // rdi
  __int64 v12; // r8
  unsigned __int64 *v13; // rax
  __m128i *v14; // rsi
  __m128i v15; // xmm0
  unsigned __int64 v16; // rsi
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  unsigned __int64 *v19; // rbx
  __m128i v20; // xmm3
  __m128i v21; // xmm0
  __m128i v22; // xmm2
  __m128i *v23; // rsi
  __m128i v24; // xmm0
  bool v25; // zf
  unsigned __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // rdx
  __m128i v29; // xmm1
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rdx
  __m128i v33; // xmm1
  __int64 v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // rdx
  _QWORD *v37; // rax
  __int64 *v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  _QWORD *v41; // rax
  __int64 *v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // [rsp+8h] [rbp-E8h]
  __int64 v46; // [rsp+8h] [rbp-E8h]
  __int64 v47; // [rsp+10h] [rbp-E0h]
  __int64 v48; // [rsp+10h] [rbp-E0h]
  __int64 v49; // [rsp+18h] [rbp-D8h]
  __int64 v50; // [rsp+18h] [rbp-D8h]
  __int64 v51; // [rsp+18h] [rbp-D8h]
  __int64 v52; // [rsp+20h] [rbp-D0h]
  __int64 v53; // [rsp+20h] [rbp-D0h]
  unsigned __int64 *v54; // [rsp+20h] [rbp-D0h]
  __int64 v55; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v56; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v57; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v58; // [rsp+28h] [rbp-C8h]
  __m128i v59; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v60; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v61; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v62; // [rsp+60h] [rbp-90h] BYREF
  __m128i v63; // [rsp+70h] [rbp-80h] BYREF
  __m128i v64; // [rsp+80h] [rbp-70h] BYREF
  __m128i v65; // [rsp+90h] [rbp-60h] BYREF
  __m128i v66; // [rsp+A0h] [rbp-50h] BYREF
  __m128i v67; // [rsp+B0h] [rbp-40h] BYREF

  for ( result = (__int64)sub_2BEC920((_QWORD *)a1);
        *(_DWORD *)(a1 + 152) == 19;
        result = (__int64)sub_2BE3490((unsigned __int64 *)(a1 + 304), &v65) )
  {
    result = sub_2BE0030(a1);
    if ( !(_BYTE)result )
      break;
    v3 = *(_QWORD *)(a1 + 352);
    if ( v3 == *(_QWORD *)(a1 + 360) )
    {
      v41 = *(_QWORD **)(*(_QWORD *)(a1 + 376) - 8LL);
      v5 = v41[60];
      v7 = v41[62];
      v55 = v41[61];
      j_j___libc_free_0(v3);
      v42 = (__int64 *)(*(_QWORD *)(a1 + 376) - 8LL);
      *(_QWORD *)(a1 + 376) = v42;
      v43 = *v42;
      v44 = *v42 + 504;
      *(_QWORD *)(a1 + 360) = v43;
      *(_QWORD *)(a1 + 368) = v44;
      *(_QWORD *)(a1 + 352) = v43 + 480;
    }
    else
    {
      v4 = *(_QWORD *)(v3 - 16);
      v5 = *(_QWORD *)(v3 - 24);
      v6 = v3 - 24;
      v7 = *(_QWORD *)(v6 + 16);
      *(_QWORD *)(a1 + 352) = v6;
      v55 = v4;
    }
    sub_2BEC920((_QWORD *)a1);
    v8 = *(_QWORD *)(a1 + 352);
    if ( v8 == *(_QWORD *)(a1 + 360) )
    {
      v37 = *(_QWORD **)(*(_QWORD *)(a1 + 376) - 8LL);
      v47 = v37[60];
      v49 = v37[61];
      v52 = v37[62];
      j_j___libc_free_0(v8);
      v9 = v47;
      v10 = v49;
      v12 = v52;
      v38 = (__int64 *)(*(_QWORD *)(a1 + 376) - 8LL);
      *(_QWORD *)(a1 + 376) = v38;
      v39 = *v38;
      v40 = *v38 + 504;
      *(_QWORD *)(a1 + 360) = v39;
      *(_QWORD *)(a1 + 368) = v40;
      *(_QWORD *)(a1 + 352) = v39 + 480;
    }
    else
    {
      v9 = *(_QWORD *)(v8 - 24);
      v10 = *(_QWORD *)(v8 - 16);
      v11 = v8 - 24;
      v12 = *(_QWORD *)(v11 + 16);
      *(_QWORD *)(a1 + 352) = v11;
    }
    v13 = *(unsigned __int64 **)(a1 + 256);
    v65.m128i_i32[0] = 10;
    v65.m128i_i64[1] = -1;
    v14 = (__m128i *)v13[8];
    if ( v14 == (__m128i *)v13[9] )
    {
      v46 = v12;
      v48 = v10;
      v51 = v9;
      v54 = v13;
      sub_2BE00E0(v13 + 7, v14, &v65);
      v13 = v54;
      v12 = v46;
      v10 = v48;
      v9 = v51;
      v16 = v54[8];
    }
    else
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(&v65);
        v15 = _mm_loadu_si128(&v66);
        v14[1] = v15;
        v14[2] = _mm_loadu_si128(&v67);
        if ( v65.m128i_i32[0] == 11 )
        {
          v14[2].m128i_i64[0] = 0;
          v33 = _mm_loadu_si128(&v66);
          v66 = v15;
          v14[1] = v33;
          v34 = v67.m128i_i64[0];
          v67.m128i_i64[0] = 0;
          v35 = v14[2].m128i_i64[1];
          v14[2].m128i_i64[0] = v34;
          v36 = v67.m128i_i64[1];
          v67.m128i_i64[1] = v35;
          v14[2].m128i_i64[1] = v36;
        }
        v14 = (__m128i *)v13[8];
      }
      v16 = (unsigned __int64)&v14[3];
      v13[8] = v16;
    }
    v17 = v16 - v13[7];
    if ( (unsigned __int64)v17 > 0x493E00 )
      goto LABEL_34;
    v18 = 0xAAAAAAAAAAAAAAABLL * (v17 >> 4) - 1;
    if ( v65.m128i_i32[0] == 11 )
    {
      v45 = v12;
      v50 = v10;
      v53 = v9;
      sub_A17130((__int64)&v66);
      v12 = v45;
      v18 = 0xAAAAAAAAAAAAAAABLL * (v17 >> 4) - 1;
      v10 = v50;
      v9 = v53;
    }
    v59.m128i_i32[0] = 1;
    *(_QWORD *)(*(_QWORD *)(v5 + 56) + 48 * v7 + 8) = v18;
    *(_QWORD *)(*(_QWORD *)(v9 + 56) + 48 * v12 + 8) = v18;
    v19 = *(unsigned __int64 **)(a1 + 256);
    v59.m128i_i64[1] = v10;
    v20 = _mm_loadu_si128(&v61);
    v60.m128i_i64[0] = v55;
    v21 = _mm_loadu_si128(&v59);
    v22 = _mm_loadu_si128(&v60);
    v64 = v20;
    v23 = (__m128i *)v19[8];
    v62 = v21;
    v63 = v22;
    if ( v23 == (__m128i *)v19[9] )
    {
      v58 = v18;
      sub_2BE00E0(v19 + 7, v23, &v62);
      v26 = v19[8];
      v18 = v58;
    }
    else
    {
      if ( v23 )
      {
        *v23 = v21;
        v24 = _mm_loadu_si128(&v63);
        v25 = v62.m128i_i32[0] == 11;
        v23[1] = v24;
        v23[2] = _mm_loadu_si128(&v64);
        if ( v25 )
        {
          v29 = _mm_loadu_si128(&v63);
          v63 = v24;
          v30 = v23[2].m128i_i64[1];
          v23[2].m128i_i64[0] = 0;
          v23[1] = v29;
          v31 = v64.m128i_i64[0];
          v64.m128i_i64[0] = 0;
          v23[2].m128i_i64[0] = v31;
          v32 = v64.m128i_i64[1];
          v64.m128i_i64[1] = v30;
          v23[2].m128i_i64[1] = v32;
        }
        v23 = (__m128i *)v19[8];
      }
      v26 = (unsigned __int64)&v23[3];
      v19[8] = v26;
    }
    v27 = v26 - v19[7];
    if ( (unsigned __int64)v27 > 0x493E00 )
LABEL_34:
      abort();
    if ( v62.m128i_i32[0] == 11 )
    {
      v57 = v18;
      sub_A17130((__int64)&v63);
      v18 = v57;
    }
    if ( v59.m128i_i32[0] == 11 )
    {
      v56 = v18;
      sub_A17130((__int64)&v60);
      v18 = v56;
    }
    v28 = *(_QWORD *)(a1 + 256);
    v65.m128i_i64[1] = 0xAAAAAAAAAAAAAAABLL * (v27 >> 4) - 1;
    v66.m128i_i64[0] = v18;
    v65.m128i_i64[0] = v28;
  }
  return result;
}
