// Function: sub_33AAC60
// Address: 0x33aac60
//
__int64 __fastcall sub_33AAC60(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r13
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned int v15; // r12d
  __int64 v17; // r13
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __m128i si128; // xmm0
  __int64 v22; // [rsp+8h] [rbp-98h]
  __m128i v23; // [rsp+10h] [rbp-90h] BYREF
  __int64 v24; // [rsp+20h] [rbp-80h] BYREF
  int v25; // [rsp+28h] [rbp-78h]
  unsigned __int64 v26; // [rsp+30h] [rbp-70h]
  __int64 v27; // [rsp+38h] [rbp-68h]
  __int64 v28; // [rsp+40h] [rbp-60h]
  __int64 v29[2]; // [rsp+50h] [rbp-50h] BYREF
  __m128i v30[4]; // [rsp+60h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 8LL);
  v23.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)v4 + 96LL);
  if ( v3 )
  {
    v27 = 0;
    BYTE4(v28) = 0;
    v26 = v3 & 0xFFFFFFFFFFFFFFFBLL;
    v5 = *(_QWORD *)(v3 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
      v5 = **(_QWORD **)(v5 + 16);
    v6 = *(_DWORD *)(v5 + 8) >> 8;
  }
  else
  {
    v26 = 0;
    v6 = 0;
    v27 = 0;
    BYTE4(v28) = 0;
  }
  LODWORD(v28) = v6;
  v7 = sub_338B750(a1, v3);
  v8 = *(_QWORD *)(a1 + 864);
  v24 = 0;
  v9 = v7;
  v11 = v10;
  v12 = *(_QWORD *)a1;
  v25 = *(_DWORD *)(a1 + 848);
  v13 = v8;
  if ( v12 )
  {
    if ( &v24 != (__int64 *)(v12 + 48) )
    {
      v14 = *(_QWORD *)(v12 + 48);
      v24 = v14;
      if ( v14 )
      {
        v22 = v8;
        sub_B96E90((__int64)&v24, v14, 1);
        v13 = *(_QWORD *)(a1 + 864);
        v8 = v22;
      }
    }
  }
  if ( (_OWORD *(__fastcall *)(_OWORD *))v23.m128i_i64[0] == sub_3364FB0 )
  {
    v15 = 0;
    if ( v24 )
      sub_B91220((__int64)&v24, v24);
  }
  else
  {
    ((void (__fastcall *)(__int64 *, __int64, __int64, __int64 *, _QWORD, _QWORD, __int64, __int64, unsigned __int64, __int64, __int64))v23.m128i_i64[0])(
      v29,
      v4,
      v13,
      &v24,
      *(_QWORD *)(v8 + 384),
      *(_QWORD *)(v8 + 392),
      v9,
      v11,
      v26,
      v27,
      v28);
    v17 = v29[0];
    if ( v24 )
      sub_B91220((__int64)&v24, v24);
    if ( v17 )
    {
      sub_33809B0((__int64 *)a1, a2, v29[0], v29[1], 0);
      v20 = *(unsigned int *)(a1 + 136);
      si128 = _mm_load_si128(v30);
      if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
      {
        v23 = si128;
        sub_C8D5F0(a1 + 128, (const void *)(a1 + 144), v20 + 1, 0x10u, v18, v19);
        v20 = *(unsigned int *)(a1 + 136);
        si128 = _mm_load_si128(&v23);
      }
      v15 = 1;
      *(__m128i *)(*(_QWORD *)(a1 + 128) + 16 * v20) = si128;
      ++*(_DWORD *)(a1 + 136);
    }
    else
    {
      return 0;
    }
  }
  return v15;
}
