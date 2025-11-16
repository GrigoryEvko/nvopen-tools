// Function: sub_37AC4A0
// Address: 0x37ac4a0
//
unsigned __int8 *__fastcall sub_37AC4A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  int v10; // r9d
  unsigned int v11; // edx
  _QWORD *v12; // rdi
  __int64 v13; // roff
  __m128i v14; // xmm0
  unsigned __int16 *v15; // rsi
  unsigned __int8 *v16; // r12
  __int64 v18; // [rsp+0h] [rbp-B0h]
  __int64 v19; // [rsp+30h] [rbp-80h] BYREF
  int v20; // [rsp+38h] [rbp-78h]
  __m128i v21; // [rsp+40h] [rbp-70h] BYREF
  __int64 v22; // [rsp+50h] [rbp-60h]
  __int64 v23; // [rsp+58h] [rbp-58h]
  __int64 v24; // [rsp+60h] [rbp-50h]
  unsigned __int64 v25; // [rsp+68h] [rbp-48h]
  __m128i v26; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v19 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v19, v3, 1);
  v20 = *(_DWORD *)(a2 + 72);
  v4 = sub_379AB60(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v6 = v5;
  v7 = *(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v5;
  LOWORD(v5) = *(_WORD *)v7;
  v21.m128i_i64[1] = *(_QWORD *)(v7 + 8);
  v21.m128i_i16[0] = v5;
  if ( !(_WORD)v5 )
    sub_3007240((__int64)&v21);
  v8 = *(_QWORD *)(a2 + 40);
  v18 = *(_QWORD *)(v8 + 88);
  v9 = sub_379AB60(a1, *(_QWORD *)(v8 + 80), v18);
  v10 = *(_DWORD *)(a2 + 28);
  v12 = *(_QWORD **)(a1 + 8);
  v13 = *(_QWORD *)(a2 + 40);
  v14 = _mm_loadu_si128((const __m128i *)v13);
  v23 = v6;
  v24 = v9;
  v15 = *(unsigned __int16 **)(a2 + 48);
  v25 = v11 | v18 & 0xFFFFFFFF00000000LL;
  v22 = v4;
  v21 = v14;
  v26 = _mm_loadu_si128((const __m128i *)(v13 + 120));
  v16 = sub_33FBA10(v12, *(unsigned int *)(a2 + 24), (__int64)&v19, *v15, *((_QWORD *)v15 + 1), v10, (__int64)&v21, 4);
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
  return v16;
}
