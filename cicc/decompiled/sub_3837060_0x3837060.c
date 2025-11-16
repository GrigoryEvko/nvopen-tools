// Function: sub_3837060
// Address: 0x3837060
//
__int64 *__fastcall sub_3837060(__int64 a1, __int64 a2, __m128i a3)
{
  _QWORD *v4; // r15
  __int64 v5; // r14
  unsigned __int64 v6; // r10
  __int64 v7; // rbx
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned __int8 *v14; // rax
  __int64 v15; // rdx
  unsigned __int8 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // [rsp+0h] [rbp-60h]
  unsigned __int64 v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+10h] [rbp-50h]
  __int64 v23; // [rsp+10h] [rbp-50h]
  unsigned __int8 *v24; // [rsp+10h] [rbp-50h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+20h] [rbp-40h] BYREF
  int v27; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD **)(a1 + 8);
  v5 = *(_QWORD *)(a2 + 40);
  if ( *(_DWORD *)(a2 + 24) == 454 )
  {
    v17 = sub_37AF270(a1, *(_QWORD *)v5, *(_QWORD *)(v5 + 8), a3);
    return sub_33EC3B0(
             v4,
             (__int64 *)a2,
             (__int64)v17,
             v18,
             *(_QWORD *)(v5 + 40),
             *(_QWORD *)(v5 + 48),
             *(_OWORD *)(v5 + 80));
  }
  else
  {
    v6 = *(_QWORD *)v5;
    v7 = *(_QWORD *)(v5 + 8);
    v8 = *(_QWORD *)(*(_QWORD *)v5 + 80LL);
    v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * *(unsigned int *)(v5 + 8));
    v10 = *v9;
    v11 = *((_QWORD *)v9 + 1);
    v26 = v8;
    if ( v8 )
    {
      v19 = v10;
      v20 = v6;
      v22 = v11;
      sub_B96E90((__int64)&v26, v8, 1);
      v10 = v19;
      v6 = v20;
      v11 = v22;
    }
    v21 = v10;
    v23 = v11;
    v27 = *(_DWORD *)(v6 + 72);
    v12 = sub_37AE0F0(a1, v6, v7);
    v14 = sub_34070B0(*(_QWORD **)(a1 + 8), v12, v7 & 0xFFFFFFFF00000000LL | v13, (__int64)&v26, v21, v23, a3);
    if ( v26 )
    {
      v24 = v14;
      v25 = v15;
      sub_B91220((__int64)&v26, v26);
      v14 = v24;
      v15 = v25;
    }
    return sub_33EBEE0(v4, (__int64 *)a2, (__int64)v14, v15);
  }
}
