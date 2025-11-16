// Function: sub_21274A0
// Address: 0x21274a0
//
__int64 *__fastcall sub_21274A0(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 *v12; // r15
  __int64 v13; // r10
  __int64 v14; // r9
  __int64 v15; // r11
  unsigned __int64 v16; // rcx
  const void **v17; // r8
  __int64 *v18; // r12
  __int128 v20; // [rsp-20h] [rbp-90h]
  unsigned __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 v22; // [rsp+10h] [rbp-60h]
  __int64 v23; // [rsp+18h] [rbp-58h]
  const void **v24; // [rsp+20h] [rbp-50h]
  __int64 v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  int v27; // [rsp+38h] [rbp-38h]

  v6 = sub_2125740(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v8 = v7;
  v9 = sub_2125740(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v11 = *(_QWORD *)(a2 + 72);
  v12 = *(__int64 **)(a1 + 8);
  v13 = v9;
  v14 = *(_QWORD *)(a2 + 32);
  v15 = v10;
  v16 = **(unsigned __int8 **)(v6 + 40);
  v17 = *(const void ***)(*(_QWORD *)(v6 + 40) + 8LL);
  v26 = v11;
  if ( v11 )
  {
    v23 = v10;
    v21 = v16;
    v22 = v9;
    v24 = v17;
    v25 = v14;
    sub_1623A60((__int64)&v26, v11, 2);
    v16 = v21;
    v13 = v22;
    v15 = v23;
    v17 = v24;
    v14 = v25;
  }
  *((_QWORD *)&v20 + 1) = v8;
  v27 = *(_DWORD *)(a2 + 64);
  *(_QWORD *)&v20 = v6;
  v18 = sub_1D3A900(
          v12,
          0x86u,
          (__int64)&v26,
          v16,
          v17,
          0,
          a3,
          a4,
          a5,
          *(_QWORD *)v14,
          *(__int16 **)(v14 + 8),
          v20,
          v13,
          v15);
  if ( v26 )
    sub_161E7C0((__int64)&v26, v26);
  return v18;
}
