// Function: sub_6F8020
// Address: 0x6f8020
//
void __fastcall sub_6F8020(const __m128i *a1)
{
  bool v1; // zf
  __int8 v2; // al
  __int64 v3; // r13
  __m128i v4; // xmm1
  __m128i v5; // xmm2
  __m128i v6; // xmm3
  __m128i v7; // xmm4
  __m128i v8; // xmm5
  __m128i v9; // xmm6
  __int8 v10; // al
  __m128i v11; // xmm7
  __m128i v12; // xmm0
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __m128i v17; // xmm2
  __m128i v18; // xmm3
  __m128i v19; // xmm4
  __m128i v20; // xmm5
  __m128i v21; // xmm6
  __m128i v22; // xmm7
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __m128i v26; // xmm4
  __m128i v27; // xmm5
  __m128i v28; // xmm6
  int v29; // [rsp+Ch] [rbp-174h] BYREF
  _OWORD v30[9]; // [rsp+10h] [rbp-170h] BYREF
  __m128i v31; // [rsp+A0h] [rbp-E0h]
  __m128i v32; // [rsp+B0h] [rbp-D0h]
  __m128i v33; // [rsp+C0h] [rbp-C0h]
  __m128i v34; // [rsp+D0h] [rbp-B0h]
  __m128i v35; // [rsp+E0h] [rbp-A0h]
  __m128i v36; // [rsp+F0h] [rbp-90h]
  __m128i v37; // [rsp+100h] [rbp-80h]
  __m128i v38; // [rsp+110h] [rbp-70h]
  __m128i v39; // [rsp+120h] [rbp-60h]
  __m128i v40; // [rsp+130h] [rbp-50h]
  __m128i v41; // [rsp+140h] [rbp-40h]
  __m128i v42; // [rsp+150h] [rbp-30h]
  __m128i v43; // [rsp+160h] [rbp-20h]

  v1 = a1[1].m128i_i8[1] == 2;
  v29 = 0;
  if ( !v1 )
    return;
  v2 = a1[1].m128i_i8[0];
  if ( v2 != 2 )
  {
    if ( v2 != 1 )
      return;
    v3 = a1[9].m128i_i64[0];
    goto LABEL_6;
  }
  if ( a1[19].m128i_i8[13] == 12 )
  {
    if ( (unsigned int)sub_6DF6A0((__int64)a1[9].m128i_i64, &v29) )
    {
      sub_6F7FE0((__int64)a1, 0, v13, v14, v15, v16);
      return;
    }
    if ( a1[19].m128i_i8[13] == 12 && a1[20].m128i_i8[0] == 1 )
    {
      v3 = sub_72E9A0(v16);
LABEL_6:
      if ( v3 && ((*(_BYTE *)(v3 + 25) & 1) != 0 || sub_6DF740(v3, &v29)) )
      {
        v4 = _mm_loadu_si128(a1 + 1);
        v5 = _mm_loadu_si128(a1 + 2);
        v6 = _mm_loadu_si128(a1 + 3);
        v7 = _mm_loadu_si128(a1 + 4);
        v8 = _mm_loadu_si128(a1 + 5);
        v30[0] = _mm_loadu_si128(a1);
        v9 = _mm_loadu_si128(a1 + 6);
        v10 = a1[1].m128i_i8[0];
        v30[1] = v4;
        v11 = _mm_loadu_si128(a1 + 7);
        v30[2] = v5;
        v12 = _mm_loadu_si128(a1 + 8);
        v30[3] = v6;
        v30[4] = v7;
        v30[5] = v8;
        v30[6] = v9;
        v30[7] = v11;
        v30[8] = v12;
        if ( v10 == 2 )
        {
          v17 = _mm_loadu_si128(a1 + 10);
          v18 = _mm_loadu_si128(a1 + 11);
          v19 = _mm_loadu_si128(a1 + 12);
          v20 = _mm_loadu_si128(a1 + 13);
          v31 = _mm_loadu_si128(a1 + 9);
          v21 = _mm_loadu_si128(a1 + 14);
          v22 = _mm_loadu_si128(a1 + 15);
          v32 = v17;
          v23 = _mm_loadu_si128(a1 + 16);
          v24 = _mm_loadu_si128(a1 + 17);
          v33 = v18;
          v25 = _mm_loadu_si128(a1 + 18);
          v34 = v19;
          v26 = _mm_loadu_si128(a1 + 19);
          v35 = v20;
          v27 = _mm_loadu_si128(a1 + 20);
          v36 = v21;
          v28 = _mm_loadu_si128(a1 + 21);
          v37 = v22;
          v38 = v23;
          v39 = v24;
          v40 = v25;
          v41 = v26;
          v42 = v27;
          v43 = v28;
        }
        else if ( v10 == 5 || v10 == 1 )
        {
          v31.m128i_i64[0] = a1[9].m128i_i64[0];
        }
        sub_6E7150((__int64 *)v3, (__int64)a1);
        if ( v29 )
          a1[1].m128i_i8[1] = 3;
        sub_6E4BC0((__int64)a1, (__int64)v30);
      }
    }
  }
}
