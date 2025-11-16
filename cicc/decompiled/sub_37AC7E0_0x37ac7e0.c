// Function: sub_37AC7E0
// Address: 0x37ac7e0
//
unsigned __int8 *__fastcall sub_37AC7E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int16 *v11; // rsi
  unsigned int v12; // edx
  __m128i v13; // xmm0
  _QWORD *v14; // rdi
  int v15; // r9d
  unsigned __int8 *v16; // r12
  __int64 v18; // [rsp+8h] [rbp-A8h]
  __int64 v19; // [rsp+30h] [rbp-80h] BYREF
  int v20; // [rsp+38h] [rbp-78h]
  __int16 v21; // [rsp+40h] [rbp-70h] BYREF
  __int64 v22; // [rsp+48h] [rbp-68h]
  _QWORD v23[4]; // [rsp+50h] [rbp-60h] BYREF
  __m128i v24; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v19 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v19, v3, 1);
  v20 = *(_DWORD *)(a2 + 72);
  v4 = sub_379AB60(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v18 = v5;
  v6 = *(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v5;
  LOWORD(v5) = *(_WORD *)v6;
  v7 = *(_QWORD *)(v6 + 8);
  v21 = v5;
  v22 = v7;
  if ( !(_WORD)v5 )
    sub_3007240((__int64)&v21);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(v8 + 48);
  v10 = sub_379AB60(a1, *(_QWORD *)(v8 + 40), v9);
  v23[0] = v4;
  v11 = *(unsigned __int16 **)(a2 + 48);
  v23[2] = v10;
  v13 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 80LL));
  v14 = *(_QWORD **)(a1 + 8);
  v23[1] = v18;
  v15 = *(_DWORD *)(a2 + 28);
  v23[3] = v12 | v9 & 0xFFFFFFFF00000000LL;
  v24 = v13;
  v16 = sub_33FBA10(v14, *(unsigned int *)(a2 + 24), (__int64)&v19, *v11, *((_QWORD *)v11 + 1), v15, (__int64)v23, 3);
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
  return v16;
}
