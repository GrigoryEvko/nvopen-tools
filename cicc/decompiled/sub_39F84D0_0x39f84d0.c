// Function: sub_39F84D0
// Address: 0x39f84d0
//
__int64 __fastcall sub_39F84D0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  __int64 v7; // rax
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __m128i v13; // xmm4
  __m128i v14; // xmm5
  __m128i v15; // xmm6
  __m128i v16; // xmm7
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  __m128i v19; // xmm2
  __m128i v20; // xmm3
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  __m128i v23; // xmm6
  __int64 result; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // [rsp+0h] [rbp-228h] BYREF
  __m128i v28; // [rsp+8h] [rbp-220h] BYREF
  __m128i v29; // [rsp+18h] [rbp-210h] BYREF
  __m128i v30; // [rsp+28h] [rbp-200h] BYREF
  __m128i v31; // [rsp+38h] [rbp-1F0h] BYREF
  __m128i v32; // [rsp+48h] [rbp-1E0h] BYREF
  __m128i v33; // [rsp+58h] [rbp-1D0h] BYREF
  __m128i v34; // [rsp+68h] [rbp-1C0h] BYREF
  __m128i v35; // [rsp+78h] [rbp-1B0h] BYREF
  __m128i v36; // [rsp+88h] [rbp-1A0h] BYREF
  __m128i v37; // [rsp+98h] [rbp-190h] BYREF
  __m128i v38; // [rsp+A8h] [rbp-180h] BYREF
  __m128i v39; // [rsp+B8h] [rbp-170h] BYREF
  __m128i v40; // [rsp+C8h] [rbp-160h] BYREF
  __m128i v41; // [rsp+D8h] [rbp-150h] BYREF
  __m128i v42; // [rsp+E8h] [rbp-140h] BYREF
  __m128i v43[9]; // [rsp+F8h] [rbp-130h] BYREF
  __m128i v44; // [rsp+188h] [rbp-A0h]
  __m128i v45; // [rsp+198h] [rbp-90h]
  __m128i v46; // [rsp+1A8h] [rbp-80h]
  __m128i v47; // [rsp+1B8h] [rbp-70h]
  __m128i v48; // [rsp+1C8h] [rbp-60h]
  __m128i v49; // [rsp+1D8h] [rbp-50h]
  __int64 v50; // [rsp+1F0h] [rbp-38h]
  __int64 v51; // [rsp+1F8h] [rbp-30h]
  __int64 retaddr; // [rsp+230h] [rbp+8h]

  v51 = a3;
  v50 = v7;
  sub_39F7A80(&v28, (__int64)&a7, retaddr);
  v9 = _mm_loadu_si128(&v28);
  v10 = _mm_loadu_si128(&v29);
  v11 = _mm_loadu_si128(&v30);
  v12 = _mm_loadu_si128(&v31);
  a1[2] = a2;
  v13 = _mm_loadu_si128(&v32);
  v14 = _mm_loadu_si128(&v33);
  a1[3] = a3;
  v15 = _mm_loadu_si128(&v34);
  v16 = _mm_loadu_si128(&v35);
  v43[0] = v9;
  v17 = _mm_loadu_si128(&v36);
  v43[1] = v10;
  v18 = _mm_loadu_si128(&v37);
  v43[2] = v11;
  v19 = _mm_loadu_si128(&v38);
  v43[3] = v12;
  v20 = _mm_loadu_si128(&v39);
  v43[4] = v13;
  v21 = _mm_loadu_si128(&v40);
  v43[5] = v14;
  v22 = _mm_loadu_si128(&v41);
  v43[6] = v15;
  v23 = _mm_loadu_si128(&v42);
  v43[7] = v16;
  v43[8] = v17;
  v44 = v18;
  v45 = v19;
  v46 = v20;
  v47 = v21;
  v48 = v22;
  v49 = v23;
  result = sub_39F7D50(a1, v43, &v27);
  if ( (_DWORD)result == 7 )
  {
    sub_39F5CF0((__int64)&v28, (__int64)v43);
    v25 = v44.m128i_i64[1];
    nullsub_2004();
    *(__int64 *)((char *)&retaddr + v26) = v25;
    return v50;
  }
  return result;
}
