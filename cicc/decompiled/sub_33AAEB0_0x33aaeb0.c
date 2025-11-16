// Function: sub_33AAEB0
// Address: 0x33aaeb0
//
__int64 __fastcall sub_33AAEB0(__int64 a1, __int64 a2)
{
  int v3; // edx
  __int64 v4; // rdx
  __int64 v5; // r8
  __int64 v6; // rsi
  __int64 v7; // r15
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // r10
  __int64 v20; // rsi
  unsigned int v21; // r12d
  __int64 v23; // r12
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __m128i si128; // xmm0
  __int64 v28; // [rsp+0h] [rbp-B0h]
  __int64 v29; // [rsp+8h] [rbp-A8h]
  __int64 v30; // [rsp+18h] [rbp-98h]
  __int64 v31; // [rsp+18h] [rbp-98h]
  __m128i v32; // [rsp+20h] [rbp-90h] BYREF
  __int64 v33; // [rsp+30h] [rbp-80h] BYREF
  int v34; // [rsp+38h] [rbp-78h]
  unsigned __int64 v35; // [rsp+40h] [rbp-70h]
  __int64 v36; // [rsp+48h] [rbp-68h]
  __int64 v37; // [rsp+50h] [rbp-60h]
  __int64 v38[2]; // [rsp+60h] [rbp-50h] BYREF
  __m128i v39[4]; // [rsp+70h] [rbp-40h] BYREF

  v3 = *(_DWORD *)(a2 + 4);
  BYTE4(v37) = 0;
  v36 = 0;
  v4 = v3 & 0x7FFFFFF;
  v5 = *(_QWORD *)(a2 - 32 * v4);
  v6 = *(_QWORD *)(a2 + 32 * (1 - v4));
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 8LL);
  v32.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)v7 + 104LL);
  v35 = v5 & 0xFFFFFFFFFFFFFFFBLL;
  v8 = 0;
  if ( v5 )
  {
    v9 = *(_QWORD *)(v5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
      v9 = **(_QWORD **)(v9 + 16);
    v8 = *(_DWORD *)(v9 + 8) >> 8;
  }
  v30 = v5;
  LODWORD(v37) = v8;
  v10 = sub_338B750(a1, v6);
  v12 = v11;
  v13 = sub_338B750(a1, v30);
  v15 = *(_QWORD *)(a1 + 864);
  v33 = 0;
  v16 = v13;
  v17 = v14;
  v18 = *(_QWORD *)a1;
  v19 = v15;
  v34 = *(_DWORD *)(a1 + 848);
  if ( v18 )
  {
    if ( &v33 != (__int64 *)(v18 + 48) )
    {
      v20 = *(_QWORD *)(v18 + 48);
      v33 = v20;
      if ( v20 )
      {
        v28 = v16;
        v29 = v14;
        v31 = v15;
        sub_B96E90((__int64)&v33, v20, 1);
        v19 = *(_QWORD *)(a1 + 864);
        v16 = v28;
        v17 = v29;
        v15 = v31;
      }
    }
  }
  if ( (_OWORD *(__fastcall *)(_OWORD *))v32.m128i_i64[0] == sub_3364FC0 )
  {
    v21 = 0;
    if ( v33 )
      sub_B91220((__int64)&v33, v33);
  }
  else
  {
    ((void (__fastcall *)(__int64 *, __int64, __int64, __int64 *, _QWORD, _QWORD, __int64, __int64, __int64, __int64, unsigned __int64, __int64, __int64))v32.m128i_i64[0])(
      v38,
      v7,
      v19,
      &v33,
      *(_QWORD *)(v15 + 384),
      *(_QWORD *)(v15 + 392),
      v16,
      v17,
      v10,
      v12,
      v35,
      v36,
      v37);
    v23 = v38[0];
    if ( v33 )
      sub_B91220((__int64)&v33, v33);
    if ( v23 )
    {
      sub_33809B0((__int64 *)a1, a2, v38[0], v38[1], 0);
      v26 = *(unsigned int *)(a1 + 136);
      si128 = _mm_load_si128(v39);
      if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
      {
        v32 = si128;
        sub_C8D5F0(a1 + 128, (const void *)(a1 + 144), v26 + 1, 0x10u, v24, v25);
        v26 = *(unsigned int *)(a1 + 136);
        si128 = _mm_load_si128(&v32);
      }
      v21 = 1;
      *(__m128i *)(*(_QWORD *)(a1 + 128) + 16 * v26) = si128;
      ++*(_DWORD *)(a1 + 136);
    }
    else
    {
      return 0;
    }
  }
  return v21;
}
