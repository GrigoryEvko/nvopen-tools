// Function: sub_690CC0
// Address: 0x690cc0
//
__int64 __fastcall sub_690CC0(const __m128i *a1, __int64 a2)
{
  unsigned int v3; // r12d
  __int64 v5; // r14
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rsi
  unsigned int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned int v22; // r14d
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __m128i v26; // xmm4
  __m128i v27; // xmm5
  __m128i v28; // xmm6
  __m128i v29; // xmm7
  __m128i v30; // xmm0
  __int8 v31; // al
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  __m128i v34; // xmm4
  __m128i v35; // xmm5
  __m128i v36; // xmm6
  __m128i v37; // xmm7
  __m128i v38; // xmm1
  __m128i v39; // xmm2
  __m128i v40; // xmm3
  __m128i v41; // xmm4
  __m128i v42; // xmm5
  __m128i v43; // xmm6
  unsigned int v44; // [rsp+4h] [rbp-18Ch] BYREF
  __int64 v45; // [rsp+8h] [rbp-188h] BYREF
  _OWORD v46[9]; // [rsp+10h] [rbp-180h] BYREF
  __m128i v47; // [rsp+A0h] [rbp-F0h]
  __m128i v48; // [rsp+B0h] [rbp-E0h]
  __m128i v49; // [rsp+C0h] [rbp-D0h]
  __m128i v50; // [rsp+D0h] [rbp-C0h]
  __m128i v51; // [rsp+E0h] [rbp-B0h]
  __m128i v52; // [rsp+F0h] [rbp-A0h]
  __m128i v53; // [rsp+100h] [rbp-90h]
  __m128i v54; // [rsp+110h] [rbp-80h]
  __m128i v55; // [rsp+120h] [rbp-70h]
  __m128i v56; // [rsp+130h] [rbp-60h]
  __m128i v57; // [rsp+140h] [rbp-50h]
  __m128i v58; // [rsp+150h] [rbp-40h]
  __m128i v59; // [rsp+160h] [rbp-30h]

  if ( unk_4D041F8 && *(_BYTE *)(a2 + 56) == 105 && (v5 = *(_QWORD *)(a2 + 72), v7 = v5, (v12 = sub_72B0F0(v5, 0)) != 0) )
  {
    v13 = *(_QWORD *)(v5 + 16);
    v14 = sub_724DC0(v7, 0, v8, v9, v10, v11);
    v15 = v12;
    v16 = v13;
    v45 = v14;
    v17 = sub_717910(v12, v13, a2, v14, &v44);
    v22 = v44;
    v3 = v17;
    if ( v44 )
    {
      if ( (unsigned int)sub_6E5430(v15, v16, v18, v19, v20, v21) )
        sub_6851C0(v22, &a1[4].m128i_i32[1]);
      sub_6E6840(a1);
    }
    else if ( v17 )
    {
      v23 = _mm_loadu_si128(a1 + 1);
      v24 = _mm_loadu_si128(a1 + 2);
      v25 = _mm_loadu_si128(a1 + 3);
      v26 = _mm_loadu_si128(a1 + 4);
      v27 = _mm_loadu_si128(a1 + 5);
      v46[0] = _mm_loadu_si128(a1);
      v28 = _mm_loadu_si128(a1 + 6);
      v29 = _mm_loadu_si128(a1 + 7);
      v46[1] = v23;
      v30 = _mm_loadu_si128(a1 + 8);
      v31 = a1[1].m128i_i8[0];
      v46[2] = v24;
      v46[3] = v25;
      v46[4] = v26;
      v46[5] = v27;
      v46[6] = v28;
      v46[7] = v29;
      v46[8] = v30;
      if ( v31 == 2 )
      {
        v32 = _mm_loadu_si128(a1 + 10);
        v33 = _mm_loadu_si128(a1 + 11);
        v34 = _mm_loadu_si128(a1 + 12);
        v35 = _mm_loadu_si128(a1 + 13);
        v47 = _mm_loadu_si128(a1 + 9);
        v36 = _mm_loadu_si128(a1 + 14);
        v37 = _mm_loadu_si128(a1 + 15);
        v48 = v32;
        v38 = _mm_loadu_si128(a1 + 16);
        v39 = _mm_loadu_si128(a1 + 17);
        v49 = v33;
        v40 = _mm_loadu_si128(a1 + 18);
        v50 = v34;
        v41 = _mm_loadu_si128(a1 + 19);
        v51 = v35;
        v42 = _mm_loadu_si128(a1 + 20);
        v52 = v36;
        v43 = _mm_loadu_si128(a1 + 21);
        v53 = v37;
        v54 = v38;
        v55 = v39;
        v56 = v40;
        v57 = v41;
        v58 = v42;
        v59 = v43;
      }
      else if ( v31 == 5 || v31 == 1 )
      {
        v47.m128i_i64[0] = a1[9].m128i_i64[0];
      }
      sub_6E6A50(v45, a1);
      sub_6E4BC0(a1, v46);
      if ( *(_BYTE *)(unk_4D03C50 + 16LL) && *(_BYTE *)(v45 + 173) != 12 )
        a1[18].m128i_i64[0] = a2;
    }
    sub_724E30(&v45);
  }
  else
  {
    return 0;
  }
  return v3;
}
