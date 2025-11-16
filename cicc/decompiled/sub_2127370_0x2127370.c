// Function: sub_2127370
// Address: 0x2127370
//
__int64 *__fastcall sub_2127370(__int64 *a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int16 *v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // r14
  unsigned __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned int v16; // esi
  __int64 *v17; // r12
  __int128 v19; // [rsp-20h] [rbp-A0h]
  unsigned __int64 v20; // [rsp+0h] [rbp-80h]
  __int64 v21; // [rsp+0h] [rbp-80h]
  __int64 v22; // [rsp+8h] [rbp-78h]
  unsigned __int64 v23; // [rsp+10h] [rbp-70h]
  __int16 *v24; // [rsp+18h] [rbp-68h]
  unsigned __int64 v25; // [rsp+20h] [rbp-60h]
  const void **v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+30h] [rbp-50h] BYREF
  int v28; // [rsp+38h] [rbp-48h]
  const void **v29; // [rsp+40h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v27,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v26 = v29;
  v20 = (unsigned __int8)v28;
  v23 = sub_2125740((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v24 = v6;
  v7 = sub_2125740((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v9 = v8;
  v10 = sub_2125740((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v12 = (__int64 *)a1[1];
  v13 = v20;
  v14 = v10;
  v15 = v11;
  v27 = *(_QWORD *)(a2 + 72);
  if ( v27 )
  {
    v22 = v11;
    v25 = v20;
    v21 = v10;
    sub_1623A60((__int64)&v27, v27, 2);
    v13 = v25;
    v14 = v21;
    v15 = v22;
  }
  v16 = *(unsigned __int16 *)(a2 + 24);
  *((_QWORD *)&v19 + 1) = v9;
  *(_QWORD *)&v19 = v7;
  v28 = *(_DWORD *)(a2 + 64);
  v17 = sub_1D3A900(v12, v16, (__int64)&v27, v13, v26, 0, a3, a4, a5, v23, v24, v19, v14, v15);
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v17;
}
