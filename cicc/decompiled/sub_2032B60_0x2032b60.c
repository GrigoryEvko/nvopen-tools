// Function: sub_2032B60
// Address: 0x2032b60
//
__int64 *__fastcall sub_2032B60(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  unsigned __int64 v6; // r12
  __int16 *v7; // rdx
  __int16 *v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // r10
  __int64 v14; // r11
  __int64 *v15; // r15
  unsigned __int64 v16; // rcx
  const void **v17; // r8
  unsigned int v18; // esi
  __int64 *v19; // r12
  __int64 v21; // [rsp+0h] [rbp-70h]
  __int64 v22; // [rsp+8h] [rbp-68h]
  unsigned __int64 v23; // [rsp+10h] [rbp-60h]
  const void **v24; // [rsp+18h] [rbp-58h]
  __int128 v25; // [rsp+20h] [rbp-50h]
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  int v27; // [rsp+38h] [rbp-38h]

  v6 = sub_2032580(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = v7;
  *(_QWORD *)&v25 = sub_2032580(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  *((_QWORD *)&v25 + 1) = v9;
  v10 = sub_2032580(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v12 = *(_QWORD *)(a2 + 72);
  v13 = v10;
  v14 = v11;
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(unsigned __int8 *)(*(_QWORD *)(v6 + 40) + 16LL * (unsigned int)v8);
  v17 = *(const void ***)(*(_QWORD *)(v6 + 40) + 16LL * (unsigned int)v8 + 8);
  v26 = v12;
  if ( v12 )
  {
    v22 = v11;
    v23 = v16;
    v21 = v10;
    v24 = v17;
    sub_1623A60((__int64)&v26, v12, 2);
    v16 = v23;
    v13 = v21;
    v14 = v22;
    v17 = v24;
  }
  v18 = *(unsigned __int16 *)(a2 + 24);
  v27 = *(_DWORD *)(a2 + 64);
  v19 = sub_1D3A900(v15, v18, (__int64)&v26, v16, v17, 0, a3, a4, a5, v6, v8, v25, v13, v14);
  if ( v26 )
    sub_161E7C0((__int64)&v26, v26);
  return v19;
}
