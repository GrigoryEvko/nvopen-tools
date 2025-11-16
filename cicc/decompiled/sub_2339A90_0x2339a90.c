// Function: sub_2339A90
// Address: 0x2339a90
//
__m128i **__fastcall sub_2339A90(__m128i **a1, __int64 a2)
{
  __m128i v2; // xmm0
  __m128i v3; // xmm1
  __m128i v4; // xmm2
  __m128i v5; // xmm3
  __m128i *v6; // rax
  __m128i *v7; // rbx
  __m128i v8; // xmm5
  __m128i v9; // xmm6
  __m128i v10; // xmm7
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v14; // [rsp+0h] [rbp-140h] BYREF
  __m128i v15; // [rsp+8h] [rbp-138h] BYREF
  __m128i v16; // [rsp+18h] [rbp-128h] BYREF
  __m128i v17; // [rsp+28h] [rbp-118h] BYREF
  __m128i v18; // [rsp+38h] [rbp-108h] BYREF
  __int64 v19; // [rsp+48h] [rbp-F8h]
  __int64 v20; // [rsp+50h] [rbp-F0h]
  __int64 v21; // [rsp+58h] [rbp-E8h]
  __int64 v22; // [rsp+60h] [rbp-E0h]
  __int64 v23; // [rsp+68h] [rbp-D8h]
  __int64 v24; // [rsp+70h] [rbp-D0h]
  __int64 v25; // [rsp+78h] [rbp-C8h]
  __int64 v26; // [rsp+80h] [rbp-C0h]
  __int64 v27; // [rsp+88h] [rbp-B8h]
  __int64 v28; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v29; // [rsp+98h] [rbp-A8h] BYREF
  __m128i v30; // [rsp+A8h] [rbp-98h] BYREF
  __m128i v31; // [rsp+B8h] [rbp-88h] BYREF
  __m128i v32; // [rsp+C8h] [rbp-78h] BYREF
  __int64 v33; // [rsp+D8h] [rbp-68h]
  __int64 v34; // [rsp+E0h] [rbp-60h]
  __int64 v35; // [rsp+E8h] [rbp-58h]
  __int64 v36; // [rsp+F0h] [rbp-50h]
  __int64 v37; // [rsp+F8h] [rbp-48h]
  __int64 v38; // [rsp+100h] [rbp-40h]
  __int64 v39; // [rsp+108h] [rbp-38h]
  __int64 v40; // [rsp+110h] [rbp-30h]
  __int64 v41; // [rsp+118h] [rbp-28h]

  sub_2D513D0(&v14, a2 + 8);
  v2 = _mm_loadu_si128(&v15);
  v3 = _mm_loadu_si128(&v16);
  v4 = _mm_loadu_si128(&v17);
  v28 = v14;
  v5 = _mm_loadu_si128(&v18);
  v29 = v2;
  v33 = v19;
  v30 = v3;
  v34 = v20;
  v31 = v4;
  v35 = v21;
  v32 = v5;
  v36 = v22;
  v19 = 0;
  v37 = v23;
  v20 = 0;
  v38 = v24;
  LODWORD(v21) = 0;
  v39 = v25;
  v22 = 0;
  v23 = 0;
  LODWORD(v24) = 0;
  v40 = v26;
  v25 = 0;
  v41 = v27;
  v26 = 0;
  LODWORD(v27) = 0;
  v6 = (__m128i *)sub_22077B0(0x98u);
  v7 = v6;
  if ( v6 )
  {
    v8 = _mm_loadu_si128(&v30);
    v9 = _mm_loadu_si128(&v31);
    v10 = _mm_loadu_si128(&v32);
    v6[1] = _mm_loadu_si128(&v29);
    v6->m128i_i64[0] = (__int64)&unk_4A0B178;
    v11 = v28;
    v7[2] = v8;
    v7->m128i_i64[1] = v11;
    v12 = v33;
    v7[3] = v9;
    v7[5].m128i_i64[0] = v12;
    v33 = 0;
    v7[5].m128i_i64[1] = v34;
    v34 = 0;
    v7[6].m128i_i64[0] = v35;
    LODWORD(v35) = 0;
    v7[6].m128i_i64[1] = v36;
    v36 = 0;
    v7[7].m128i_i64[0] = v37;
    v37 = 0;
    v7[7].m128i_i64[1] = v38;
    LODWORD(v38) = 0;
    v7[8].m128i_i64[0] = v39;
    v7[4] = v10;
    v39 = 0;
    v7[8].m128i_i64[1] = v40;
    v40 = 0;
    v7[9].m128i_i64[0] = v41;
    LODWORD(v41) = 0;
  }
  sub_2339850((__int64)&v28);
  *a1 = v7;
  sub_2339850((__int64)&v14);
  return a1;
}
