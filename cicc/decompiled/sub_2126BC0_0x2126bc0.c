// Function: sub_2126BC0
// Address: 0x2126bc0
//
__int64 *__fastcall sub_2126BC0(__int64 *a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int16 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // rsi
  __int64 *v11; // r10
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // r8
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // r12
  __int128 v20; // [rsp-20h] [rbp-A0h]
  unsigned int v21; // [rsp+4h] [rbp-7Ch]
  __int64 *v22; // [rsp+8h] [rbp-78h]
  unsigned __int64 v23; // [rsp+8h] [rbp-78h]
  unsigned __int64 v24; // [rsp+10h] [rbp-70h]
  __int16 *v25; // [rsp+18h] [rbp-68h]
  unsigned __int8 v26; // [rsp+20h] [rbp-60h]
  __int64 *v27; // [rsp+20h] [rbp-60h]
  const void **v28; // [rsp+28h] [rbp-58h]
  __int64 v29; // [rsp+30h] [rbp-50h] BYREF
  int v30; // [rsp+38h] [rbp-48h]
  const void **v31; // [rsp+40h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v29,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v26 = v30;
  v28 = v31;
  v24 = sub_2125740((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v25 = v6;
  v7 = sub_2125740((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v10 = *(_QWORD *)(a2 + 72);
  v11 = (__int64 *)a1[1];
  v12 = v7;
  v13 = v8;
  v14 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL) + 84LL);
  v29 = v10;
  if ( v10 )
  {
    v21 = v14;
    v22 = v11;
    sub_1623A60((__int64)&v29, v10, 2);
    v14 = v21;
    v11 = v22;
  }
  v15 = v26;
  v27 = v11;
  v23 = v15;
  v30 = *(_DWORD *)(a2 + 64);
  v16 = sub_1D28D50(v11, v14, v8, v15, v14, v9);
  *((_QWORD *)&v20 + 1) = v13;
  *(_QWORD *)&v20 = v12;
  v18 = sub_1D3A900(v27, 0x89u, (__int64)&v29, v23, v28, 0, a3, a4, a5, v24, v25, v20, v16, v17);
  if ( v29 )
    sub_161E7C0((__int64)&v29, v29);
  return v18;
}
