// Function: sub_6EFA90
// Address: 0x6efa90
//
__int64 __fastcall sub_6EFA90(const __m128i *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r13
  int v7; // eax
  char v8; // dl
  __int64 v9; // rax
  __int64 v10; // rdi
  char v11; // dl
  __int64 v12; // rax
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  __m128i v17; // xmm4
  __m128i v18; // xmm5
  __m128i v19; // xmm6
  __int8 v20; // al
  __m128i v21; // xmm7
  __m128i v22; // xmm0
  __m128i v23; // xmm2
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  __m128i v26; // xmm5
  __m128i v27; // xmm6
  __m128i v28; // xmm7
  __m128i v29; // xmm1
  __m128i v30; // xmm2
  __m128i v31; // xmm3
  __m128i v32; // xmm4
  __m128i v33; // xmm5
  __m128i v34; // xmm6
  int v35; // [rsp+4h] [rbp-17Ch] BYREF
  __int64 *v36; // [rsp+8h] [rbp-178h] BYREF
  _OWORD v37[9]; // [rsp+10h] [rbp-170h] BYREF
  __m128i v38; // [rsp+A0h] [rbp-E0h]
  __m128i v39; // [rsp+B0h] [rbp-D0h]
  __m128i v40; // [rsp+C0h] [rbp-C0h]
  __m128i v41; // [rsp+D0h] [rbp-B0h]
  __m128i v42; // [rsp+E0h] [rbp-A0h]
  __m128i v43; // [rsp+F0h] [rbp-90h]
  __m128i v44; // [rsp+100h] [rbp-80h]
  __m128i v45; // [rsp+110h] [rbp-70h]
  __m128i v46; // [rsp+120h] [rbp-60h]
  __m128i v47; // [rsp+130h] [rbp-50h]
  __m128i v48; // [rsp+140h] [rbp-40h]
  __m128i v49; // [rsp+150h] [rbp-30h]
  __m128i v50; // [rsp+160h] [rbp-20h]

  if ( (unsigned int)sub_8D3410(a1->m128i_i64[0]) && (unsigned int)sub_8319F0(a1, &v36) )
  {
    v14 = _mm_loadu_si128(a1 + 1);
    v15 = _mm_loadu_si128(a1 + 2);
    v16 = _mm_loadu_si128(a1 + 3);
    v17 = _mm_loadu_si128(a1 + 4);
    v18 = _mm_loadu_si128(a1 + 5);
    v37[0] = _mm_loadu_si128(a1);
    v19 = _mm_loadu_si128(a1 + 6);
    v20 = a1[1].m128i_i8[0];
    v37[1] = v14;
    v21 = _mm_loadu_si128(a1 + 7);
    v37[2] = v15;
    v22 = _mm_loadu_si128(a1 + 8);
    v37[3] = v16;
    v37[4] = v17;
    v37[5] = v18;
    v37[6] = v19;
    v37[7] = v21;
    v37[8] = v22;
    if ( v20 == 2 )
    {
      v23 = _mm_loadu_si128(a1 + 10);
      v24 = _mm_loadu_si128(a1 + 11);
      v25 = _mm_loadu_si128(a1 + 12);
      v26 = _mm_loadu_si128(a1 + 13);
      v38 = _mm_loadu_si128(a1 + 9);
      v27 = _mm_loadu_si128(a1 + 14);
      v28 = _mm_loadu_si128(a1 + 15);
      v39 = v23;
      v29 = _mm_loadu_si128(a1 + 16);
      v30 = _mm_loadu_si128(a1 + 17);
      v40 = v24;
      v31 = _mm_loadu_si128(a1 + 18);
      v41 = v25;
      v32 = _mm_loadu_si128(a1 + 19);
      v42 = v26;
      v33 = _mm_loadu_si128(a1 + 20);
      v43 = v27;
      v34 = _mm_loadu_si128(a1 + 21);
      v44 = v28;
      v45 = v29;
      v46 = v30;
      v47 = v31;
      v48 = v32;
      v49 = v33;
      v50 = v34;
    }
    else if ( v20 == 5 || v20 == 1 )
    {
      v38.m128i_i64[0] = a1[9].m128i_i64[0];
    }
    v36 = (__int64 *)sub_6EED10((__int64)v36, &v35, 0, 0, 0, 0);
    sub_6E7150(v36, (__int64)a1);
    return sub_6E4BC0((__int64)a1, (__int64)v37);
  }
  else
  {
    v3 = sub_8D46C0(a2);
    v5 = a1->m128i_i64[0];
    v6 = v3;
    if ( v3 != a1->m128i_i64[0] )
    {
      v7 = sub_8D97D0(a1->m128i_i64[0], v3, 32, v4, v5);
      v5 = v6;
      if ( !v7 )
      {
        v8 = *(_BYTE *)(v6 + 140);
        if ( v8 == 12 )
        {
          v9 = v6;
          do
          {
            v9 = *(_QWORD *)(v9 + 160);
            v8 = *(_BYTE *)(v9 + 140);
          }
          while ( v8 == 12 );
        }
        v5 = 0;
        if ( v8 && a1[1].m128i_i8[0] )
        {
          v10 = a1->m128i_i64[0];
          v11 = *(_BYTE *)(a1->m128i_i64[0] + 140);
          if ( v11 == 12 )
          {
            v12 = a1->m128i_i64[0];
            do
            {
              v12 = *(_QWORD *)(v12 + 160);
              v11 = *(_BYTE *)(v12 + 140);
            }
            while ( v11 == 12 );
          }
          v5 = 0;
          if ( v11 )
            v5 = sub_73CA70(v10, v6);
        }
      }
    }
    return sub_8443E0(a1, v5, 1);
  }
}
