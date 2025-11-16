// Function: sub_355D240
// Address: 0x355d240
//
__int64 __fastcall sub_355D240(unsigned __int64 *a1, _QWORD *a2)
{
  unsigned __int64 v3; // r15
  unsigned __int64 v4; // r14
  __int64 v5; // r10
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // r11
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r13
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  __m128i v18; // xmm1
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int64 v23; // rax
  __m128i v24; // xmm3
  __int64 v25; // [rsp+0h] [rbp-100h]
  unsigned __int64 v26; // [rsp+8h] [rbp-F8h]
  __int64 v27; // [rsp+10h] [rbp-F0h]
  __int64 v28; // [rsp+18h] [rbp-E8h]
  __int64 v29; // [rsp+20h] [rbp-E0h]
  __int64 v30; // [rsp+28h] [rbp-D8h]
  __m128i v31; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v32; // [rsp+40h] [rbp-C0h] BYREF
  _QWORD v33[4]; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v34; // [rsp+70h] [rbp-90h] BYREF
  __m128i v35; // [rsp+80h] [rbp-80h]
  __int64 v36; // [rsp+90h] [rbp-70h] BYREF
  __int64 v37; // [rsp+98h] [rbp-68h]
  __int64 v38; // [rsp+A0h] [rbp-60h]
  __int64 v39; // [rsp+A8h] [rbp-58h]
  __int64 v40; // [rsp+B0h] [rbp-50h] BYREF
  unsigned __int64 v41; // [rsp+B8h] [rbp-48h]
  unsigned __int64 v42; // [rsp+C0h] [rbp-40h]
  unsigned __int64 v43; // [rsp+C8h] [rbp-38h]

  v3 = a2[4];
  v4 = a2[5];
  v25 = a2[8];
  v5 = a1[7];
  v28 = a2[7];
  v29 = a2[6];
  v6 = a1[6];
  v26 = a2[3];
  v7 = a1[9];
  v27 = a2[9];
  v8 = a1[5];
  v9 = a1[4];
  v30 = a2[2];
  v10 = a1[2];
  v11 = a1[8];
  v12 = a1[3];
  v13 = ((((v7 - v8) >> 3) - 1) << 6) + ((v6 - v5) >> 3) + ((v9 - v10) >> 3);
  v14 = ((((__int64)(v27 - v4) >> 3) - 1) << 6) + ((v29 - v28) >> 3) + ((__int64)(v3 - v30) >> 3);
  if ( v6 == v10 )
  {
    v23 = (v10 - v12) >> 3;
    if ( v14 > v23 )
    {
      sub_3550820(a1, v14 - v23);
      v10 = a1[2];
      v12 = a1[3];
      v9 = a1[4];
      v8 = a1[5];
    }
    v31.m128i_i64[0] = v10;
    v31.m128i_i64[1] = v12;
    v32.m128i_i64[0] = v9;
    v32.m128i_i64[1] = v8;
    sub_353DF70(v31.m128i_i64, -(__int64)v14);
    v42 = v3;
    v43 = v4;
    v34 = v31;
    v35 = v32;
    v36 = v29;
    v37 = v28;
    v38 = v25;
    v39 = v27;
    v40 = v30;
    v41 = v26;
    sub_355D160(v33, &v40, &v36, &v34);
    v24 = _mm_loadu_si128(&v32);
    *((__m128i *)a1 + 1) = _mm_loadu_si128(&v31);
    *((__m128i *)a1 + 2) = v24;
  }
  else
  {
    v15 = ((v11 - v6) >> 3) - 1;
    if ( v14 > v15 )
    {
      sub_35508F0(a1, v14 - v15);
      v6 = a1[6];
      v5 = a1[7];
      v11 = a1[8];
      v7 = a1[9];
    }
    v32.m128i_i64[1] = v7;
    v31.m128i_i64[0] = v6;
    v31.m128i_i64[1] = v5;
    v32.m128i_i64[0] = v11;
    sub_353DF70(v31.m128i_i64, v14);
    v16 = a1[6];
    v42 = v3;
    v34.m128i_i64[0] = v16;
    v17 = a1[7];
    v43 = v4;
    v34.m128i_i64[1] = v17;
    v35 = *((__m128i *)a1 + 4);
    v36 = v29;
    v37 = v28;
    v38 = v25;
    v39 = v27;
    v40 = v30;
    v41 = v26;
    sub_355D160(v33, &v40, &v36, &v34);
    v18 = _mm_loadu_si128(&v32);
    *((__m128i *)a1 + 3) = _mm_loadu_si128(&v31);
    *((__m128i *)a1 + 4) = v18;
  }
  v19 = a1[3];
  v20 = a1[4];
  v21 = a1[5];
  v40 = a1[2];
  v41 = v19;
  v42 = v20;
  v43 = v21;
  return sub_353DF70(&v40, v13);
}
