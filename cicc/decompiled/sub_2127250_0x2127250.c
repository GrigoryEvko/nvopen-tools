// Function: sub_2127250
// Address: 0x2127250
//
__int64 *__fastcall sub_2127250(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v6; // r15d
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // r12
  unsigned __int64 v11; // r9
  const void **v12; // r8
  __int64 v13; // r10
  __int64 v14; // r11
  __int64 v15; // rsi
  __int64 *v16; // r12
  __int128 v18; // [rsp-10h] [rbp-90h]
  __int64 v19; // [rsp+0h] [rbp-80h]
  __int64 v20; // [rsp+8h] [rbp-78h]
  unsigned int v21; // [rsp+14h] [rbp-6Ch]
  const void **v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+20h] [rbp-60h]
  unsigned __int64 v24; // [rsp+28h] [rbp-58h]
  __int64 v25; // [rsp+30h] [rbp-50h] BYREF
  int v26; // [rsp+38h] [rbp-48h]
  const void **v27; // [rsp+40h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v25,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = (unsigned __int8)v26;
  v22 = v27;
  v23 = sub_2125740((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v24 = v7;
  v8 = sub_2125740((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v10 = (__int64 *)a1[1];
  v11 = *(unsigned __int16 *)(a2 + 80);
  v12 = v22;
  v13 = v8;
  v14 = v9;
  v25 = *(_QWORD *)(a2 + 72);
  if ( v25 )
  {
    v20 = v9;
    v21 = v11;
    v19 = v8;
    sub_1623A60((__int64)&v25, v25, 2);
    v13 = v19;
    v14 = v20;
    v11 = v21;
    v12 = v22;
  }
  *((_QWORD *)&v18 + 1) = v14;
  *(_QWORD *)&v18 = v13;
  v15 = *(unsigned __int16 *)(a2 + 24);
  v26 = *(_DWORD *)(a2 + 64);
  v16 = sub_1D332F0(v10, v15, (__int64)&v25, v6, v12, v11, a3, a4, a5, v23, v24, v18);
  if ( v25 )
    sub_161E7C0((__int64)&v25, v25);
  return v16;
}
