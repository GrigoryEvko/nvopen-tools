// Function: sub_37996A0
// Address: 0x37996a0
//
__m128i *__fastcall sub_37996A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r8
  __int64 v7; // rsi
  __int64 v8; // r9
  __int64 v9; // rdx
  __m128i v10; // xmm1
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __m128i *v16; // r14
  __int64 v17; // rbx
  __int16 v18; // r14
  unsigned __int8 v19; // r15
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v23; // [rsp+8h] [rbp-98h]
  __int64 v24; // [rsp+10h] [rbp-90h]
  __int64 v25; // [rsp+10h] [rbp-90h]
  unsigned __int8 v26; // [rsp+1Fh] [rbp-81h]
  __int16 v27; // [rsp+20h] [rbp-80h]
  __int64 v28; // [rsp+20h] [rbp-80h]
  _QWORD *v29; // [rsp+28h] [rbp-78h]
  __int64 v30; // [rsp+30h] [rbp-70h] BYREF
  int v31; // [rsp+38h] [rbp-68h]
  __int16 v32; // [rsp+40h] [rbp-60h] BYREF
  __int64 v33; // [rsp+48h] [rbp-58h]
  __m128i v34; // [rsp+50h] [rbp-50h] BYREF
  __m128i v35; // [rsp+60h] [rbp-40h]

  v5 = a1;
  v7 = *(_QWORD *)(a2 + 80);
  v30 = v7;
  if ( v7 )
  {
    sub_B96E90((__int64)&v30, v7, 1);
    v5 = a1;
  }
  v8 = *(_QWORD *)(a2 + 112);
  v31 = *(_DWORD *)(a2 + 72);
  v29 = *(_QWORD **)(v5 + 8);
  if ( (*(_BYTE *)(a2 + 33) & 4) != 0 )
  {
    v9 = *(_QWORD *)(a2 + 104);
    v34 = _mm_loadu_si128((const __m128i *)(v8 + 40));
    v10 = _mm_loadu_si128((const __m128i *)(v8 + 56));
    v33 = v9;
    v35 = v10;
    v27 = *(_WORD *)(v8 + 32);
    v26 = *(_BYTE *)(v8 + 34);
    LODWORD(v11) = *(unsigned __int16 *)(a2 + 96);
    v32 = v11;
    if ( (_WORD)v11 )
    {
      LOWORD(v11) = word_4456580[(int)v11 - 1];
      v12 = 0;
    }
    else
    {
      v25 = v5;
      v11 = sub_3009970((__int64)&v32, v7, v9, a4, v5);
      v8 = *(_QWORD *)(a2 + 112);
      v5 = v25;
      v4 = v11;
    }
    v13 = v12;
    LOWORD(v4) = v11;
    v23 = v8;
    v24 = *(_QWORD *)(a2 + 40);
    v14 = sub_37946F0(v5, *(_QWORD *)(v24 + 40), *(_QWORD *)(v24 + 48));
    v16 = sub_33F5040(
            v29,
            **(_QWORD **)(a2 + 40),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
            (__int64)&v30,
            v14,
            v15,
            *(_QWORD *)(v24 + 80),
            *(_QWORD *)(v24 + 88),
            *(_OWORD *)v23,
            *(_QWORD *)(v23 + 16),
            v4,
            v13,
            v26,
            v27,
            (__int64)&v34);
  }
  else
  {
    v17 = *(_QWORD *)(a2 + 40);
    v28 = v8;
    v34 = _mm_loadu_si128((const __m128i *)(v8 + 40));
    v35 = _mm_loadu_si128((const __m128i *)(v8 + 56));
    v18 = *(_WORD *)(v8 + 32);
    v19 = *(_BYTE *)(v8 + 34);
    v20 = sub_37946F0(v5, *(_QWORD *)(v17 + 40), *(_QWORD *)(v17 + 48));
    v16 = sub_33F4560(
            v29,
            **(_QWORD **)(a2 + 40),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
            (__int64)&v30,
            v20,
            v21,
            *(_QWORD *)(v17 + 80),
            *(_QWORD *)(v17 + 88),
            *(_OWORD *)v28,
            *(_QWORD *)(v28 + 16),
            v19,
            v18,
            (__int64)&v34);
  }
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v16;
}
