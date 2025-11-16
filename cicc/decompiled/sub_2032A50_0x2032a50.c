// Function: sub_2032A50
// Address: 0x2032a50
//
__int64 *__fastcall sub_2032A50(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r14
  unsigned __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 *v15; // r11
  __int64 v16; // rcx
  const void **v17; // r8
  __int64 v18; // rsi
  __int64 *v19; // r12
  __int128 v21; // [rsp-10h] [rbp-80h]
  __int64 v22; // [rsp+8h] [rbp-68h]
  unsigned int v23; // [rsp+14h] [rbp-5Ch]
  const void **v24; // [rsp+18h] [rbp-58h]
  __int64 *v25; // [rsp+20h] [rbp-50h]
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  int v27; // [rsp+38h] [rbp-38h]

  v6 = sub_2032580(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = v7;
  v9 = sub_2032580(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v10 = *(_QWORD *)(a2 + 72);
  v11 = v9;
  v12 = *(unsigned __int16 *)(a2 + 80);
  v14 = v13;
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(unsigned __int8 *)(*(_QWORD *)(v6 + 40) + 16LL * (unsigned int)v8);
  v17 = *(const void ***)(*(_QWORD *)(v6 + 40) + 16LL * (unsigned int)v8 + 8);
  v26 = v10;
  if ( v10 )
  {
    v22 = v16;
    v23 = v12;
    v24 = v17;
    v25 = v15;
    sub_1623A60((__int64)&v26, v10, 2);
    v16 = v22;
    v12 = v23;
    v17 = v24;
    v15 = v25;
  }
  *((_QWORD *)&v21 + 1) = v14;
  *(_QWORD *)&v21 = v11;
  v18 = *(unsigned __int16 *)(a2 + 24);
  v27 = *(_DWORD *)(a2 + 64);
  v19 = sub_1D332F0(v15, v18, (__int64)&v26, v16, v17, v12, a3, a4, a5, v6, v8, v21);
  if ( v26 )
    sub_161E7C0((__int64)&v26, v26);
  return v19;
}
