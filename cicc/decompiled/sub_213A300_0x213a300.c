// Function: sub_213A300
// Address: 0x213a300
//
__int64 *__fastcall sub_213A300(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // r10
  __int64 v6; // r9
  __int128 *v7; // r11
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // r8
  __int64 v12; // rcx
  int v13; // esi
  const __m128i *v14; // r9
  __int64 *v15; // r14
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+10h] [rbp-60h]
  __int128 *v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+20h] [rbp-50h]
  _QWORD *v21; // [rsp+28h] [rbp-48h]
  __int64 v22; // [rsp+30h] [rbp-40h] BYREF
  int v23; // [rsp+38h] [rbp-38h]

  v3 = sub_2138AD0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v4 = *(_QWORD *)(a2 + 72);
  v5 = *(_QWORD **)(a1 + 8);
  v6 = *(_QWORD *)(a2 + 104);
  v7 = *(__int128 **)(a2 + 32);
  v8 = v3;
  v10 = v9;
  v11 = *(_QWORD *)(a2 + 96);
  v12 = *(unsigned __int8 *)(a2 + 88);
  v22 = v4;
  if ( v4 )
  {
    v17 = v12;
    v18 = v6;
    v19 = v7;
    v20 = v11;
    v21 = v5;
    sub_1623A60((__int64)&v22, v4, 2);
    v12 = v17;
    v6 = v18;
    v7 = v19;
    v11 = v20;
    v5 = v21;
  }
  v13 = *(unsigned __int16 *)(a2 + 24);
  v23 = *(_DWORD *)(a2 + 64);
  v15 = sub_1D2B8F0(v5, v13, (__int64)&v22, v12, v11, v6, *v7, *(__int128 *)((char *)v7 + 40), v8, v10);
  if ( v22 )
    sub_161E7C0((__int64)&v22, v22);
  sub_2013400(a1, a2, 1, (__int64)v15, (__m128i *)1, v14);
  return v15;
}
